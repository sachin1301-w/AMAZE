"""End-to-end Test Inference Pipeline.

Generates:
  1. output/matching_results.tsv (final entity resolution predictions)
  2. output/candidate_pairs.tsv  (blocking candidates fed to the matcher)

Optimized for ultra-fast, high-precision execution on 1.73M test entities.
Validates output using utils/validate_submission.py.
"""
import os
import sys
import time
import json
import multiprocessing as mp
import numpy as np
import pandas as pd
import lightgbm as lgb
from rapidfuzz import fuzz, process
from rapidfuzz.distance import JaroWinkler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from blocking import Index, hash_terms_par, topk_query, _init_worker
from decide import resolve_exclusive, select_sets, group_bounds
from normalize import LEGAL
from terms import (
    CH_ADDR, CH_ALL, CH_NAME, DF_MASK, NGRP,
    addr_parts, core_name
)

def rarity_weight(tab):
    lut = np.maximum(0.0, 23.0 - np.arange(256, dtype=np.float32))
    lut[0] = 23.0
    return lut[tab]


WORK = os.environ.get("BER_WORK", "work")
DATA = os.environ.get("BER_DATA", "dataset")
OUTPUT = os.environ.get("BER_OUTPUT", "output")

_LEGAL_ID = {v: i for i, v in enumerate(sorted(set(LEGAL.values())))}
EPS = np.float32(1e-6)
WORKERS = int(os.environ.get("BER_WORKERS", "-1"))


def _cp(a, b, scorer):
    return process.cpdist(a, b, scorer=scorer, workers=WORKERS, dtype=np.float32)


class Side:
    def __init__(self, nname, naddr, dom, rw_name, rw_addr):
        n = len(nname)
        self.n = n
        self.nm = np.asarray(nname, dtype=object)
        self.ad = np.asarray(naddr, dtype=object)
        self.domf = np.asarray([1.0 if d else 0.0 for d in dom], np.float32)
        cn = np.empty(n, object); cj = np.empty(n, object)
        aw = np.empty(n, object); nu = np.empty(n, object)
        nw = np.zeros(n, np.float32); aww = np.zeros(n, np.float32)
        nlen = np.zeros(n, np.float32); alen = np.zeros(n, np.float32)
        dlen = np.zeros(n, np.float32)
        lgl = np.zeros(n, np.int64)

        sn_sets = [None] * n
        sa_sets = [None] * n
        sd_sets = [None] * n
        max_num = [None] * n

        M = DF_MASK
        for i in range(n):
            ct = core_name(nname[i])
            w, nums = addr_parts(naddr[i])
            if dom[i] and dom[i] not in ct:
                ct = ct + [dom[i]]
            sn, sa, sd = set(ct), set(w), set(nums)
            cn[i] = " ".join(sorted(sn))
            cj[i] = "".join(sorted(sn))
            aw[i] = " ".join(sorted(sa))
            nu[i] = " ".join(sorted(sd))
            nw[i] = sum(rw_name[hash(t) & M] for t in sn)
            aww[i] = sum(rw_addr[hash(t) & M] for t in sa)
            nlen[i] = len(sn); alen[i] = len(sa); dlen[i] = len(sd)
            m = 0
            for t in nname[i].split():
                j = _LEGAL_ID.get(LEGAL.get(t, ""), -1)
                if j >= 0:
                    m |= 1 << j
            lgl[i] = m

            sn_sets[i] = frozenset(sn)
            sa_sets[i] = frozenset(sa)
            sd_sets[i] = frozenset(sd)
            max_num[i] = max(nums, key=len) if nums else ""

        self.cn, self.cj, self.aw, self.nu = cn, cj, aw, nu
        self.nw, self.aww = nw, aww
        self.nlen, self.alen, self.dlen = nlen, alen, dlen
        self.lgl = lgl
        self.rw_name, self.rw_addr = rw_name, rw_addr
        self.sn_sets = sn_sets
        self.sa_sets = sa_sets
        self.sd_sets = sd_sets
        self.max_num = max_num


def cheap_score(Q, D, qi, di):
    nm_ratio = _cp(Q.nm[qi], D.nm[di], fuzz.ratio) / 100.0
    nm_partial = _cp(Q.cn[qi], D.cn[di], fuzz.partial_ratio) / 100.0
    cj_jw = _cp(Q.cj[qi], D.cj[di], JaroWinkler.normalized_similarity)
    a = _cp(Q.aw[qi], D.aw[di], fuzz.token_sort_ratio) / 100.0
    blank = (D.alen[di] + D.dlen[di]) == 0
    ad_tsr = np.where(blank, np.float32(0.45), a)
    score = (
        1.35 * nm_ratio
        + 0.70 * nm_partial
        + 0.30 * cj_jw
        + 0.25 * ad_tsr
    ).astype(np.float32)
    return {
        "score": score,
        "nm_ratio": nm_ratio,
        "nm_partial": nm_partial,
        "cj_jw": cj_jw,
        "ad_tsr": ad_tsr,
    }


def _rank_within(q, s, nq):
    if len(q) == 0:
        return np.zeros(0, np.float32)
    o = np.lexsort((-s, q))
    qs = np.searchsorted(q[o], np.arange(nq + 1))
    starts = np.repeat(qs[:-1], np.diff(qs))
    r = np.empty(len(q), np.float32)
    r[o] = np.arange(len(q), dtype=np.float32) - starts
    return r


def _groupmax(idx, val, n):
    out = np.zeros(n, np.float32)
    if len(idx) == 0:
        return out
    o = np.argsort(idx, kind="stable")
    i2, v2 = idx[o], val[o]
    br = np.flatnonzero(np.r_[True, i2[1:] != i2[:-1]])
    out[i2[br]] = np.maximum.reduceat(v2, br)
    return out


def compute_pair_features(Q, D, qi, di, P_info, src_is3, nq):
    n = len(qi)
    NF = 57
    F = np.empty((n, NF), np.float32)
    c = 0

    def put(v):
        nonlocal c
        F[:, c] = v
        c += 1

    qnm, dnm = Q.nm[qi], D.nm[di]
    qcn, dcn = Q.cn[qi], D.cn[di]
    put(P_info["nm_ratio"])
    put(_cp(qcn, dcn, fuzz.token_sort_ratio) / 100.0)
    put(_cp(qcn, dcn, fuzz.token_set_ratio) / 100.0)
    put(P_info["nm_partial"])
    put(P_info["cj_jw"])
    put(_cp(Q.cj[qi], D.cj[di], fuzz.ratio) / 100.0)
    put(_cp(qnm, dnm, fuzz.token_set_ratio) / 100.0)
    put(_cp(Q.ad[qi], D.ad[di], fuzz.ratio) / 100.0)
    put(P_info["ad_tsr"])
    put(_cp(Q.aw[qi], D.aw[di], fuzz.token_set_ratio) / 100.0)
    put(_cp(Q.nu[qi], D.nu[di], fuzz.token_sort_ratio) / 100.0)

    rn, ra = Q.rw_name, Q.rw_addr
    M = DF_MASK
    Q_sn, D_sn = Q.sn_sets, D.sn_sets
    Q_sa, D_sa = Q.sa_sets, D.sa_sets
    Q_sd, D_sd = Q.sd_sets, D.sd_sets
    Q_mn, D_mn = Q.max_num, D.max_num

    nsh = np.empty(n, np.float32); nshw = np.empty(n, np.float32)
    ash = np.empty(n, np.float32); ashw = np.empty(n, np.float32)
    dsh = np.empty(n, np.float32); dshw = np.empty(n, np.float32)
    fne = np.zeros(n, np.float32)

    for k in range(n):
        a, b = qi[k], di[k]
        s = Q_sn[a] & D_sn[b]
        nsh[k] = len(s)
        nshw[k] = sum(rn[hash(t) & M] for t in s) if s else 0.0

        s = Q_sa[a] & D_sa[b]
        ash[k] = len(s)
        ashw[k] = sum(ra[hash(t) & M] for t in s) if s else 0.0

        s = Q_sd[a] & D_sd[b]
        dsh[k] = len(s)
        dshw[k] = sum(ra[hash(t) & M] for t in s) if s else 0.0

        ma, mb = Q_mn[a], D_mn[b]
        if ma and mb and ma == mb:
            fne[k] = 1.0

    qnw, dnw = Q.nw[qi], D.nw[di]
    qaww, daww = Q.aww[qi], D.aww[di]
    qnl, dnl = Q.nlen[qi], D.nlen[di]
    qal, dal = Q.alen[qi], D.alen[di]
    qdl, ddl = Q.dlen[qi], D.dlen[di]
    put(nsh); put(nshw); put(qnw - nshw); put(dnw - nshw)
    put(nsh / np.maximum(qnl + dnl - nsh, EPS))
    put(nsh / np.maximum(dnl, EPS)); put(nsh / np.maximum(qnl, EPS))
    put(ash); put(ashw); put(qaww - ashw); put(daww - ashw)
    put(ash / np.maximum(qal + dal - ash, EPS))
    put(ash / np.maximum(dal, EPS)); put(ash / np.maximum(qal, EPS))
    put(dsh); put(dshw)
    put(dsh / np.maximum(qdl + ddl - dsh, EPS))
    put(dsh / np.maximum(ddl, EPS)); put(fne)

    put(qnl); put(dnl); put(qal); put(dal); put(qdl); put(ddl)
    put(((dal + ddl) == 0).astype(np.float32))
    la, lb = Q.lgl[qi], D.lgl[di]
    ci = np.bitwise_count(np.bitwise_and(la, lb)).astype(np.float32)
    cu = np.bitwise_count(np.bitwise_or(la, lb)).astype(np.float32)
    put(np.where(cu > 0, ci / np.maximum(cu, EPS), np.float32(1.0)))
    put(D.domf[di])

    # 7 ret base
    for nm in ("sc_all", "sc_name", "sc_addr", "rk_all", "rk_name", "rk_addr", "nchan"):
        put(P_info[nm])

    # 4 grp dummy
    for g in range(4):
        put(np.zeros(n, np.float32))

    # 6 comp
    cs = P_info["cheap"]
    sa = P_info["sc_all"]
    qmax = _groupmax(qi, cs, nq)
    amax = _groupmax(qi, sa, nq)
    ncand = np.bincount(qi, minlength=nq).astype(np.float32)

    put(sa / np.maximum(amax[qi], EPS))
    put(cs / np.maximum(qmax[qi], EPS))
    put(_rank_within(qi, cs, nq))
    put(ncand[qi])
    put(qmax[qi] - cs)
    put(cs)

    # 1 src_is3
    put(np.full(n, np.float32(1.0 if src_is3 else 0.0)))

    assert c == NF, f"Feature count mismatch: {c} != {NF}"
    return F


def main():
    t0 = time.time()
    os.makedirs(OUTPUT, exist_ok=True)
    os.makedirs(WORK, exist_ok=True)

    print("Loading test dataset and model...", flush=True)
    model_path = os.path.join(WORK, "model_final.lgb")
    booster = lgb.Booster(model_file=model_path)
    print(f"Loaded {model_path} ({booster.num_trees()} trees)", flush=True)

    decide_cfg = dict(excl=True, m0=0.25, k0=1.8, pmin=0.01, pw=1.0)
    decide_path = os.path.join(WORK, "model_final_decide.json")
    if os.path.exists(decide_path):
        decide_cfg = json.load(open(decide_path))
    print(f"Decision config: {decide_cfg}", flush=True)

    dftab_path = os.path.join(WORK, "test_dftab.npz")
    z = np.load(dftab_path)
    rw_name, rw_addr = rarity_weight(z["name"]), rarity_weight(z["addr"])

    d1 = pd.read_parquet(os.path.join(WORK, "test_s1.parquet")).reset_index(drop=True)
    nq_total = len(d1)
    print(f"Test S1 entities: {nq_total:,}", flush=True)

    s1_ids_raw = pd.read_csv(
        os.path.join(DATA, "test", "test_source1.tsv"),
        sep="\t",
        dtype=str,
        keep_default_na=False,
        usecols=["entity_id"],
    ).entity_id.values
    ent_arr = np.array([int(x[3:]) for x in s1_ids_raw], np.int64)
    order = np.argsort(ent_arr)
    ent_sorted = ent_arr[order]

    # Fast targeted blocking: high recall, compact candidate set
    CHANS = (("all", CH_ALL, 12), ("name", CH_NAME, 4), ("addr", CH_ADDR, 4))
    KEEP_K = 4
    BUDGET = 3000
    MAXDF = 0.001
    QBATCH = 40000

    all_qi = []
    all_ds = []
    all_pp = []
    all_sr = []
    Q_cache = {}

    with mp.Pool(12, initializer=_init_worker, initargs=(dftab_path,)) as pool:
        for src in (2, 3):
            dd = pd.read_parquet(os.path.join(WORK, f"test_s{src}.parquet"))
            countries = sorted(dd.country.astype(str).unique())
            for ctry in countries:
                sub = dd[(dd.country.astype(str) == ctry).values].reset_index(drop=True)
                q = d1[(d1.country.astype(str) == ctry).values].reset_index(drop=True)
                if not len(q) or not len(sub):
                    continue

                part_file = os.path.join(WORK, f"test_pred_s{src}_{ctry}.npz")
                if os.path.exists(part_file):
                    print(f"\nLoading cached partition S{src} {ctry} from {part_file}...", flush=True)
                    pz = np.load(part_file)
                    c_qi, c_di, c_p = pz["qi"], pz["di"], pz["p"]
                    all_qi.append(np.searchsorted(ent_sorted, c_qi))
                    all_ds.append(c_di)
                    all_pp.append(c_p)
                    all_sr.append(np.full(len(c_p), src, np.int8))
                    print(f"   Loaded S{src} {ctry}: {len(c_p):,} pairs ({len(c_p)/len(q):.2f}/entity) ({time.time()-t0:.0f}s)", flush=True)
                    continue

                print(f"\nProcessing S{src} {ctry}: {len(q):,} S1 x {len(sub):,} S{src} ({time.time()-t0:.0f}s)", flush=True)
                docs = dict(nname=sub.nname.values, naddr=sub.naddr.values, dom=sub.dom.values)
                th, ti, tc = hash_terms_par(docs["nname"], docs["naddr"], docs["dom"], pool)
                idxs = {}
                for cname, ctypes, _ in CHANS:
                    idxs[cname] = Index(th, ti, tc, len(sub), types=ctypes, maxdf_frac=MAXDF)
                del th, ti, tc
                D = Side(sub.nname.values, sub.naddr.values, sub.dom.values, rw_name, rw_addr)
                if ctry not in Q_cache:
                    Q_cache[ctry] = Side(q.nname.values, q.naddr.values, q.dom.values, rw_name, rw_addr)
                Q = Q_cache[ctry]

                nq_part = len(q)
                part_qi, part_di, part_p = [], [], []
                n_batches = (nq_part + QBATCH - 1) // QBATCH

                for b_idx, b0 in enumerate(range(0, nq_part, QBATCH)):
                    b1 = min(nq_part, b0 + QBATCH)
                    nb = b1 - b0
                    qh, qi_h, qc = hash_terms_par(q.nname.values[b0:b1], q.naddr.values[b0:b1], q.dom.values[b0:b1], pool)
                    keys, chan_sc, chan_rk = [], [], []
                    for ci, (cname, ctypes, kk) in enumerate(CHANS):
                        Qq, Dd, Ss = topk_query(idxs[cname], qh, qc, qi_h, nb, kk, BUDGET, types=ctypes, mode="cos", beta=1.0)
                        if len(Qq) == 0:
                            continue
                        rk = _rank_within(Qq, Ss, nb)
                        keys.append(Qq.astype(np.int64) * D.n + Dd)
                        chan_sc.append((ci, Ss))
                        chan_rk.append(rk)
                    if not keys:
                        continue
                    allk = np.concatenate(keys)
                    uk, inv = np.unique(allk, return_inverse=True)
                    m = len(uk)
                    sc = np.zeros((m, 3), np.float32)
                    rkv = np.full((m, 3), 999.0, np.float32)
                    off = 0
                    for (ci, Ss), rk, kar in zip(chan_sc, chan_rk, keys):
                        sl = inv[off:off + len(kar)]
                        sc[sl, ci] = Ss
                        rkv[sl, ci] = rk
                        off += len(kar)
                    qq = (uk // D.n).astype(np.int32)
                    dd_idx = (uk % D.n).astype(np.int32)

                    ch = cheap_score(Q, D, qq + b0, dd_idx)
                    cs = ch["score"]

                    o = np.lexsort((-cs, qq))
                    qs = np.searchsorted(qq[o], np.arange(nb + 1))
                    sel = []
                    for j in range(nb):
                        a_idx, e_idx = qs[j], qs[j + 1]
                        if e_idx > a_idx:
                            sel.append(o[a_idx:min(e_idx, a_idx + KEEP_K)])
                    if not sel:
                        continue
                    sel = np.concatenate(sel)
                    sel_qi = qq[sel] + b0
                    sel_di = dd_idx[sel]

                    P_info = {
                        "cheap": cs[sel],
                        "sc_all": sc[sel, 0],
                        "sc_name": sc[sel, 1],
                        "sc_addr": sc[sel, 2],
                        "rk_all": rkv[sel, 0],
                        "rk_name": rkv[sel, 1],
                        "rk_addr": rkv[sel, 2],
                        "nchan": (sc[sel] > 0).sum(1).astype(np.float32),
                        "nm_ratio": ch["nm_ratio"][sel],
                        "nm_partial": ch["nm_partial"][sel],
                        "cj_jw": ch["cj_jw"][sel],
                        "ad_tsr": ch["ad_tsr"][sel],
                    }

                    F = compute_pair_features(Q, D, sel_qi, sel_di, P_info, src == 3, nq_part)
                    preds = booster.predict(F, num_threads=12).astype(np.float32)

                    part_qi.append(q.eid.values[sel_qi])
                    part_di.append(sub.eid.values[sel_di])
                    part_p.append(preds)

                    if (b_idx + 1) % 4 == 0 or (b_idx + 1) == n_batches:
                        print(f"      batch {b_idx + 1}/{n_batches} ({time.time()-t0:.0f}s)", flush=True)

                if part_qi:
                    c_qi = np.concatenate(part_qi)
                    c_di = np.concatenate(part_di)
                    c_p = np.concatenate(part_p)
                    print(f"   Done S{src} {ctry}: {len(c_p):,} pairs ({len(c_p)/len(q):.2f}/entity) ({time.time()-t0:.0f}s)", flush=True)
                    # Cache partition to disk
                    np.savez_compressed(part_file, qi=c_qi, di=c_di, p=c_p)
                    all_qi.append(np.searchsorted(ent_sorted, c_qi))
                    all_ds.append(c_di)
                    all_pp.append(c_p)
                    all_sr.append(np.full(len(c_p), src, np.int8))
                del idxs, D, sub, q
            del dd

    qi = np.concatenate(all_qi)
    ds = np.concatenate(all_ds)
    pp = np.concatenate(all_pp)
    sr = np.concatenate(all_sr)
    del all_qi, all_ds, all_pp, all_sr

    # Map back to original row order of test_source1.tsv
    qi = order[qi]
    print(f"\nTotal Candidate Pairs across all sources: {len(qi):,} ({len(qi)/nq_total:.2f}/entity) ({time.time()-t0:.0f}s)", flush=True)

    # Exclusivity resolution
    p = pp
    if decide_cfg.get("excl", True):
        dk = ds * 4 + sr
        ud, di = np.unique(dk, return_inverse=True)
        m = resolve_exclusive(di, p, len(ud))
        p = np.where(m, p, np.float32(0.0))
        print(f"Exclusivity filter resolved {int((~m).sum()):,} conflicting pairs ({time.time()-t0:.0f}s)", flush=True)

    # Optimal selection
    sel = select_sets(
        qi, p, nq_total,
        m0=decide_cfg["m0"],
        pmin=decide_cfg["pmin"],
        k0_boost=decide_cfg["k0"],
        pw=decide_cfg.get("pw", 1.0)
    )
    print(f"Selected {int(sel.sum()):,} matches ({sel.sum()/nq_total:.3f} per entity) ({time.time()-t0:.0f}s)", flush=True)

    sid = np.where(sr == 2, "S2-", "S3-")
    txt = np.char.add(sid, ds.astype(str))

    # Write output files
    def write_submission(path, mask, colname):
        lists = [[] for _ in range(nq_total)]
        qq = qi[mask]
        tt = txt[mask]
        for i in range(len(qq)):
            lists[qq[i]].append(tt[i])
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            f.write(f"source1_entity_id\t{colname}\n")
            for i in range(nq_total):
                f.write(s1_ids_raw[i])
                f.write("\t")
                if lists[i]:
                    f.write(",".join(dict.fromkeys(lists[i])))
                f.write("\n")
        nz = sum(1 for x in lists if x)
        print(f"Wrote {path}: {nz:,} non-empty rows, {sum(len(x) for x in lists):,} ids ({time.time()-t0:.0f}s)", flush=True)

    match_path = os.path.join(OUTPUT, "matching_results.tsv")
    cand_path = os.path.join(OUTPUT, "candidate_pairs.tsv")

    write_submission(match_path, sel, "matched_entity_ids")
    write_submission(cand_path, np.ones(len(qi), bool), "candidate_entity_ids")

    print("\nRunning submission validator...", flush=True)
    import subprocess
    cmd = [
        sys.executable,
        "utils/validate_submission.py",
        "--matching", match_path,
        "--candidate", cand_path,
        "--test-dir", os.path.join(DATA, "test"),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    print(res.stdout)
    if res.stderr:
        print(res.stderr)
    if res.returncode == 0:
        print("\nALL SUBMISSION CHECKS PASSED SUCCESSFULLY!")
    else:
        print(f"\nVALIDATION FAILED (exit {res.returncode})")


if __name__ == "__main__":
    main()
