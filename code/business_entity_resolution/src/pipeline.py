"""The per-partition pipeline: retrieve -> lexical prune -> features.

A partition is one (source, country) pair.  Work is split so that the wide
retrieval net is never materialised for the whole partition:

  1. build the three channel indexes over the partition's Source-2/3 records
  2. for each batch of Source-1 records: retrieve wide, merge the channels,
     score every candidate with the deterministic lexical score and keep the
     best ``KEEP`` per Source-1 record  -> this is the candidate set
  3. once the whole partition is pruned, add the competition features (which
     need global per-record statistics) and emit the feature matrix
"""
import os
import sys
import time

import numpy as np

from hw import THREADS, breathe

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from blocking import Index, hash_terms_par, topk_query
from features import (DERIV, FUZZ4, GRP_NAMES, RET_BASE, REV_NAMES,
                      WIDE_NAMES, Side, add_derived, cheap_prune_score,
                      full_prune_score, have_prune_models, pair_features,
                      prune_score, retr_components, two_stage)
from terms import CH_ADDR, CH_ALL, CH_NAME, NGRP

CFG = dict(
    k_all=int(os.environ.get("BER_KALL", "250")),
    k_name=int(os.environ.get("BER_KNAME", "20")),
    k_addr=int(os.environ.get("BER_KADDR", "20")),
    budget=int(os.environ.get("BER_BUDGET", "4000")),
    maxdf=float(os.environ.get("BER_MAXDF", "0.0002")),
    keep=int(os.environ.get("BER_KEEP", "15")),
    keep1=int(os.environ.get("BER_KEEP1", "120")),   # stage-1 cheap prune cut
    qbatch=int(os.environ.get("BER_QBATCH", "40000")),
    dbatch=int(os.environ.get("BER_DBATCH", "400000")),
    k_rev=int(os.environ.get("BER_KREV", "4")),
    # Reverse retrieval (Source-2/3 records querying a Source-1 index) was
    # implemented and measured: it adds only +0.003 pool recall for roughly
    # double the runtime, because the composite terms already cover the
    # truncated-record cases it was meant to catch.  Off by default.
    use_rev=os.environ.get("BER_REV", "0") == "1",
)
CHANS = (("all", CH_ALL, "k_all"), ("name", CH_NAME, "k_name"),
         ("addr", CH_ADDR, "k_addr"))


def _rank_within(q, s, nq):
    """0-based rank of each element by descending score inside its group."""
    if len(q) == 0:
        return np.zeros(0, np.float32)
    o = np.lexsort((-s, q))
    qs = np.searchsorted(q[o], np.arange(nq + 1))
    starts = np.repeat(qs[:-1], np.diff(qs))
    r = np.empty(len(q), np.float32)
    r[o] = np.arange(len(q), dtype=np.float32) - starts
    return r


def _groupmax(idx, val, n):
    """Maximum of ``val`` per group id, returned as an array of length n."""
    out = np.zeros(n, np.float32)
    if len(idx) == 0:
        return out
    o = np.argsort(idx, kind="stable")
    i2, v2 = idx[o], val[o]
    br = np.flatnonzero(np.r_[True, i2[1:] != i2[:-1]])
    out[i2[br]] = np.maximum.reduceat(v2, br)
    return out


# Components carried from the wide net through to the feature matrix.  DERIV is
# left out and recomputed (it is a pure function of the rest) and REV only joins
# when reverse retrieval is enabled.  Everything is stored as float16: LightGBM
# bins features into 255 buckets anyway, so the precision is irrelevant while
# the memory saving is not -- a US partition carries ~40M pairs.
CARRY = RET_BASE + GRP_NAMES + FUZZ4 + WIDE_NAMES
if CFG["use_rev"]:
    CARRY = CARRY + REV_NAMES


def reverse_pairs(q_nn, q_na, q_dom, d_nn, d_na, d_dom, pool, cfg, log=""):
    """Retrieve in the *reverse* direction: every Source-2/3 record queries an
    index built over the Source-1 records.

    Forward retrieval ranks documents by how well they explain a Source-1
    record, which fails when the generator truncated a variant down to a couple
    of common tokens -- the variant then looks like thousands of documents.
    Queried the other way round those same few tokens are almost entirely
    explained by the right parent, so the reverse direction recovers exactly the
    cases the forward direction loses.  Each record keeps its best few parents.

    Returns (s1_row, doc_row, score, rank) sorted by s1_row.
    """
    nq = len(q_nn)
    t0 = time.time()
    qh, qi, qc = hash_terms_par(q_nn, q_na, q_dom, pool)
    ridx = Index(qh, qi, qc, nq, types=CH_ALL, maxdf_frac=cfg["maxdf"])
    del qh, qi, qc
    print(f"    rev index {len(ridx.terms):,} terms "
          f"{time.time()-t0:.0f}s", flush=True)
    nd = len(d_nn)
    B = cfg["dbatch"]
    S1, DR, SC, RK = [], [], [], []
    for b0 in range(0, nd, B):
        b1 = min(nd, b0 + B)
        dh, dr, dc = hash_terms_par(d_nn[b0:b1], d_na[b0:b1], d_dom[b0:b1],
                                    pool)
        Dq, S1r, Ss = topk_query(ridx, dh, dc, dr, b1 - b0, cfg["k_rev"],
                                 cfg["budget"], types=CH_ALL, mode="cos",
                                 beta=1.0)
        if len(Dq):
            S1.append(S1r)
            DR.append(Dq.astype(np.int64) + b0)
            SC.append(Ss)
            RK.append(_rank_within(Dq, Ss, b1 - b0))
        del dh, dr, dc
        if log and (b0 // B) % 8 == 0:
            print(f"    rev {log} {b1:,}/{nd:,} pairs="
                  f"{sum(len(x) for x in S1):,} {time.time()-t0:.0f}s",
                  flush=True)
    del ridx
    if not S1:
        return (np.zeros(0, np.int32), np.zeros(0, np.int32),
                np.zeros(0, np.float32), np.zeros(0, np.float32),
                np.zeros(nq + 1, np.int64))
    s1 = np.concatenate(S1).astype(np.int32)
    dr = np.concatenate(DR).astype(np.int32)
    sc = np.concatenate(SC)
    rk = np.concatenate(RK)
    o = np.argsort(s1, kind="stable")
    s1, dr, sc, rk = s1[o], dr[o], sc[o], rk[o]
    off = np.searchsorted(s1, np.arange(nq + 1))
    print(f"    reverse pairs {len(s1):,} ({len(s1)/nq:.2f} per entity) "
          f"{time.time()-t0:.0f}s", flush=True)
    return s1, dr, sc, rk, off


def retrieve_prune(q_nn, q_na, q_dom, idxs, Q, D, pool, cfg, log="", rev=None,
                   src_is3=False, QE=None, DE=None):
    """Wide retrieval + deterministic lexical prune over every query row.

    Returns flat arrays for the surviving candidates, including every component
    the matcher will need, so nothing is recomputed downstream.
    """
    nq = len(q_nn)
    dmass = idxs["all"].dmass
    out = {k: [] for k in ["qi", "di", "prune", "wide_n"] + CARRY}
    B = cfg["qbatch"]
    t0 = time.time()
    for b0 in range(0, nq, B):
        tb = time.time()
        b1 = min(nq, b0 + B)
        qh, qi, qc = hash_terms_par(q_nn[b0:b1], q_na[b0:b1], q_dom[b0:b1], pool)
        nb = b1 - b0
        keys, per_chan = [], []
        gm_all = qmass = None
        for ci, (cname, ctypes, kk) in enumerate(CHANS):
            pg = cname == "all"
            r = topk_query(idxs[cname], qh, qc, qi, nb, cfg[kk], cfg["budget"],
                           types=ctypes, mode="cos", beta=1.0, per_group=pg)
            Qq, Dd, Ss = r[0], r[1], r[2]
            if pg:
                gm_all, qmass = r[3], r[4]
            if len(Qq) == 0:
                continue
            keys.append(Qq.astype(np.int64) * D.n + Dd)
            per_chan.append((ci, Ss, _rank_within(Qq, Ss, nb),
                             gm_all if pg else None))
        if rev is not None:
            r_s1, r_dr, r_sc, r_rk, r_off = rev
            a0, a1 = r_off[b0], r_off[b1]
            if a1 > a0:
                keys.append((r_s1[a0:a1].astype(np.int64) - b0) * D.n
                            + r_dr[a0:a1])
        if not keys:
            continue
        allk = np.concatenate(keys)
        uk, inv = np.unique(allk, return_inverse=True)
        m = len(uk)
        ch = np.zeros((m, 7), np.float32)
        ch[:, 3:6] = 999.0
        gm = np.zeros((m, NGRP), np.float32)
        off = 0
        for (ci, Ss, rk, g), kar in zip(per_chan, keys):
            sl = inv[off:off + len(kar)]   # a channel yields each pair once
            ch[sl, ci] = Ss
            ch[sl, 3 + ci] = rk
            if g is not None:
                gm[sl] = g
            off += len(kar)
        ch[:, 6] = (ch[:, 0:3] > 0).sum(1)
        rv = np.zeros((m, 3), np.float32)
        rv[:, 1] = 99.0
        if rev is not None and a1 > a0:
            kr = (r_s1[a0:a1].astype(np.int64) - b0) * D.n + r_dr[a0:a1]
            pos = np.searchsorted(uk, kr)
            ok = (pos < len(uk)) & (uk[np.minimum(pos, len(uk) - 1)] == kr)
            pos = pos[ok]
            rv[pos, 0] = r_sc[a0:a1][ok]
            rv[pos, 1] = r_rk[a0:a1][ok]
            rv[pos, 2] = 1.0
        qq = (uk // D.n).astype(np.int32)
        dd = (uk % D.n).astype(np.int32)
        if qmass is None:
            qmass = np.zeros((nb, NGRP), np.float32)
        C = retr_components(Q, D, qq + b0, dd, ch, gm, qmass, dmass,
                            D.n, qloc=qq, rev=rv, QE=QE, DE=DE,
                            qrow=qq, nq=nb)
        # relative strength inside the query's *wide* net
        wmax = _groupmax(qq, C["sc_all"], nb)
        C["sc_rel_w"] = C["sc_all"] / np.maximum(wmax[qq], np.float32(1e-6))

        def topk(score, K, rows=None):
            """Indices of the best K per query (within ``rows`` if given)."""
            idx = np.arange(len(score)) if rows is None else rows
            o = idx[np.lexsort((-score[idx], qq[idx]))]
            st = np.searchsorted(qq[o], np.arange(nb + 1))
            out = [o[st[j]:min(st[j + 1], st[j] + K)] for j in range(nb)
                   if st[j + 1] > st[j]]
            return np.concatenate(out) if out else np.empty(0, np.int64)

        if have_prune_models():
            ps = cheap_prune_score(C, Q, D, qq + b0, dd, src_is3)
            if two_stage():
                cut = topk(ps, cfg["keep1"])
                if len(cut) == 0:
                    continue
                Cc = {k: v[cut] for k, v in C.items()}
                Fb = pair_features(Q, D, qq[cut] + b0, dd[cut], Cc, None,
                                   src_is3)
                ps2 = np.full(len(qq), -1e9, np.float32)
                ps2[cut] = full_prune_score(Fb)
                del Fb
                sel = topk(ps2, cfg["keep"], rows=cut)
                ps = ps2
            else:
                sel = topk(ps, cfg["keep"])
        else:
            ps = prune_score(C)
            sel = topk(ps, cfg["keep"])
        if len(sel) == 0:
            continue
        wide = np.bincount(qq, minlength=nb).astype(np.float32)
        out["qi"].append(qq[sel] + b0)
        out["di"].append(dd[sel])
        out["prune"].append(ps[sel].astype(np.float16))
        out["wide_n"].append(wide[qq[sel]].astype(np.float16))
        for k in CARRY:
            out[k].append(C[k][sel].astype(np.float16))
        breathe(tb)
        if log and (b0 // B) % 2 == 0:
            print(f"    {log} {b1:,}/{nq:,} kept={sum(len(x) for x in out['qi']):,}"
                  f" {time.time()-t0:.0f}s", flush=True)
    if not out["qi"]:
        return None
    return {k: np.concatenate(v) for k, v in out.items()}


def competition(P, nq, ndoc):
    """Per-entity and per-record competition features."""
    qi, di = P["qi"], P["di"]
    ps = P["prune"].astype(np.float32)
    sa = P["sc_all"].astype(np.float32)
    pmax = _groupmax(qi, ps, nq)
    amax = _groupmax(qi, sa, nq)
    ncand = np.bincount(qi, minlength=nq).astype(np.float32)
    eps = np.float32(1e-6)
    comp = {}
    comp["sc_rel"] = sa / np.maximum(amax[qi], eps)
    comp["prune"] = ps
    comp["prune_rel"] = ps / np.maximum(pmax[qi], eps)
    comp["prune_rank"] = _rank_within(qi, ps, nq)
    comp["n_cand"] = ncand[qi]
    comp["prune_gap1"] = pmax[qi] - ps
    comp["prune_max"] = pmax[qi]
    comp["wide_n"] = P["wide_n"].astype(np.float32)
    dmax = _groupmax(di, ps, ndoc)
    dcnt = np.bincount(di, minlength=ndoc).astype(np.float32)
    comp["doc_best"] = dmax[di]
    comp["doc_gap"] = dmax[di] - ps
    comp["doc_ncand"] = dcnt[di]
    comp["doc_is_best"] = (dmax[di] <= ps + 1e-6).astype(np.float32)
    # the entity's strongest candidate, used for the cohesion features: true
    # matches of one business tend to resemble each other, sometimes sharing
    # address parts that the Source-1 record itself does not carry
    o = np.lexsort((-ps, qi))
    first = np.searchsorted(qi[o], np.arange(nq + 1))[:-1]
    bestrow = np.zeros(nq, np.int64)
    nzq = np.diff(np.searchsorted(qi[o], np.arange(nq + 1))) > 0
    bestrow[qi[o][first[nzq]]] = o[first[nzq]]
    comp["best_row"] = bestrow[qi]
    return comp


def build_indexes(docs, pool, cfg):
    th, ti, tc = hash_terms_par(docs["nname"], docs["naddr"], docs["dom"], pool)
    idxs = {}
    for cname, ctypes, _ in CHANS:
        idxs[cname] = Index(th, ti, tc, len(docs["nname"]), types=ctypes,
                            maxdf_frac=cfg["maxdf"])
    del th, ti, tc
    return idxs


def iter_features(P, Q, D, src_is3, nq, chunk=2_000_000, rows=None):
    """Yield (row_index, feature_matrix) chunks.

    ``rows`` optionally restricts output to a subset of pairs *after* the
    competition features were computed over all of them, so a subset keeps
    test-time feature semantics.
    """
    comp = competition(P, nq, D.n)
    idx = np.arange(len(P["qi"])) if rows is None else np.asarray(rows)
    for a in range(0, len(idx), chunk):
        sl = idx[a:a + chunk]
        C = {k: P[k][sl].astype(np.float32) for k in CARRY}
        if not CFG["use_rev"]:
            for k in REV_NAMES:
                C[k] = np.zeros(len(sl), np.float32)
            C["rev_rank"] = np.full(len(sl), 99.0, np.float32)
        add_derived(C, D, P["di"][sl])
        cm = {k: v[sl] for k, v in comp.items()}
        yield sl, pair_features(Q, D, P["qi"][sl], P["di"][sl], C, cm,
                                src_is3, di_all=P["di"])


def features_for(P, Q, D, src_is3, nq, chunk=2_000_000, rows=None):
    outs = [F for _, F in iter_features(P, Q, D, src_is3, nq, chunk, rows)]
    return np.concatenate(outs) if len(outs) > 1 else outs[0]
