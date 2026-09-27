"""Run the pipeline over a split and store candidates + features.

    python runpipe.py train  --nq 200000 --tag tr
    python runpipe.py test

For ``train`` the ground truth is joined in so the output can train the matcher
and measure blocking recall; for ``test`` only features are produced.
Output: work/<split>_<tag>_s{src}_{country}.npz with the feature matrix, the
entity ids and (for train) the labels.
"""
import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import hw  # noqa: E402  (must precede numpy/torch)

import argparse
import os
import sys
import time
import multiprocessing as mp

import numpy as np

from hw import THREADS
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from blocking import _init_worker
from features import Side, rarity_weight
from pipeline import (CFG, build_indexes, features_for,  # noqa: F401
                      iter_features, retrieve_prune, reverse_pairs)
from stack import stack_features

WORK = os.environ.get("BER_WORK", "../../../work")
DATA = os.environ.get("BER_DATA", "../../../dataset")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("split")
    ap.add_argument("--nq", type=int, default=0, help="subsample of Source 1")
    ap.add_argument("--tag", default="full")
    ap.add_argument("--only", default="")
    ap.add_argument("--dftab", default="")
    ap.add_argument("--model", default="", help="predict instead of storing F")
    ap.add_argument("--model2", default="", help="stage-2 stacked model")
    ap.add_argument("--prune1", default="", help="stage-1 cheap prune ranker")
    ap.add_argument("--prune2", default="", help="stage-2 full prune ranker")
    ap.add_argument("--emb", action="store_true",
                    help="use the dense encoder vectors from embed.py")
    ap.add_argument("--keepfrac", type=float, default=1.0,
                    help="store features for this fraction of entities")
    a = ap.parse_args()
    split = a.split
    t0 = time.time()
    dftab = a.dftab or f"{WORK}/{split}_dftab.npz"
    z = np.load(dftab)
    rw_name, rw_addr = rarity_weight(z["name"]), rarity_weight(z["addr"])

    if a.prune1 or a.prune2:
        from features import load_prune_models
        c, f2 = load_prune_models(a.prune1, a.prune2)
        print(f"prune models: cheap={'yes' if c else 'no'} "
              f"full={'yes' if f2 else 'no'}", flush=True)
    booster = booster2 = None
    if a.model:
        import lightgbm as lgb
        booster = lgb.Booster(model_file=a.model)
        print("loaded", a.model, booster.num_trees(), "trees", flush=True)
        if a.model2:
            booster2 = lgb.Booster(model_file=a.model2)
            print("loaded", a.model2, booster2.num_trees(), "trees", flush=True)

    E1 = None
    if a.emb:
        E1 = np.load(f"{WORK}/{split}_emb1.npy", mmap_mode="r")
        print(f"embeddings s1 {E1.shape}", flush=True)

    d1_all = pd.read_parquet(f"{WORK}/{split}_s1.parquet")
    rows1 = np.arange(len(d1_all))
    d1 = d1_all
    if a.nq and a.nq < len(d1_all):
        rng = np.random.default_rng(13)
        rows1 = np.sort(rng.choice(len(d1_all), a.nq, replace=False))
        d1 = d1_all.iloc[rows1]
    d1 = d1.reset_index(drop=True)
    # position of each kept Source-1 record in the parquet / embedding file
    d1["_row"] = rows1
    print(f"S1 rows {len(d1):,}", flush=True)

    truth = None
    if split == "train":
        gt = pd.read_csv(f"{DATA}/train/train_ground_truth.tsv", sep="\t",
                         dtype=str, keep_default_na=False)
        gt["e"] = gt.source1_entity_id.str.slice(3).astype(np.int64)
        keep = set(d1.eid.values.tolist())
        gt = gt[gt.e.isin(keep)]
        truth = dict(zip(gt.e.values, gt.matched_entity_ids.values))

    with mp.Pool(THREADS, initializer=_init_worker, initargs=(dftab,)) as pool:
        for src in (2, 3):
            dd = pd.read_parquet(f"{WORK}/{split}_s{src}.parquet")
            ES = np.load(f"{WORK}/{split}_emb{src}.npy",
                         mmap_mode="r") if a.emb else None
            for ctry in sorted(dd.country.astype(str).unique()):
                if a.only and f"{src}{ctry}" != a.only:
                    continue
                out = f"{WORK}/{split}_{a.tag}_s{src}_{ctry}.npz"
                if os.path.exists(out):
                    print("exists", out, flush=True)
                    continue
                cmask = (dd.country.astype(str) == ctry).values
                sub = dd[cmask].reset_index(drop=True)
                DE = np.ascontiguousarray(ES[cmask]) if ES is not None else None
                qmask = (d1.country.astype(str) == ctry).values
                q = d1[qmask].reset_index(drop=True)
                QE = (np.ascontiguousarray(E1[d1._row.values[qmask]])
                      if E1 is not None else None)
                # the reverse index must span EVERY Source-1 record of the
                # country, otherwise a record's "best parent" is chosen from a
                # sample and means nothing.  With no --nq these coincide.
                qall = d1_all[(d1_all.country.astype(str) == ctry).values
                              ].reset_index(drop=True)
                if not len(q) or not len(sub):
                    continue
                print(f"\n== s{src} {ctry}: {len(q):,} queries x {len(sub):,} docs "
                      f"{time.time()-t0:.0f}s", flush=True)
                docs = dict(nname=sub.nname.values, naddr=sub.naddr.values,
                            dom=sub.dom.values)
                rev = None
                if CFG["use_rev"]:
                    rev = reverse_pairs(qall.nname.values, qall.naddr.values,
                                        qall.dom.values, sub.nname.values,
                                        sub.naddr.values, sub.dom.values,
                                        pool, CFG, log=f"s{src}/{ctry}")
                    if len(qall) != len(q):
                        ae = qall.eid.values
                        ordr = np.argsort(ae)
                        loc = ordr[np.searchsorted(ae[ordr], q.eid.values)]
                        pos = np.full(len(qall), -1, np.int64)
                        pos[loc] = np.arange(len(q))
                        r_s1, r_dr, r_sc, r_rk, _ = rev
                        keepr = pos[r_s1] >= 0
                        r_s1 = pos[r_s1[keepr]].astype(np.int32)
                        r_dr, r_sc, r_rk = (r_dr[keepr], r_sc[keepr],
                                            r_rk[keepr])
                        o = np.argsort(r_s1, kind="stable")
                        r_s1, r_dr = r_s1[o], r_dr[o]
                        r_sc, r_rk = r_sc[o], r_rk[o]
                        rev = (r_s1, r_dr, r_sc, r_rk,
                               np.searchsorted(r_s1, np.arange(len(q) + 1)))
                        print(f"    reverse pairs kept for the sample: "
                              f"{len(r_s1):,} ({len(r_s1)/len(q):.2f}/entity)",
                              flush=True)
                    del qall
                idxs = build_indexes(docs, pool, CFG)
                print(f"   indexes built {time.time()-t0:.0f}s", flush=True)
                D = Side(sub.nname.values, sub.naddr.values, sub.dom.values,
                         rw_name, rw_addr)
                Q = Side(q.nname.values, q.naddr.values, q.dom.values,
                         rw_name, rw_addr)
                print(f"   sides built {time.time()-t0:.0f}s", flush=True)
                P = retrieve_prune(q.nname.values, q.naddr.values,
                                   q.dom.values, idxs, Q, D, pool, CFG,
                                   log=f"s{src}/{ctry}", rev=rev,
                                   src_is3=(src == 3), QE=QE, DE=DE)
                del rev
                del idxs
                if P is None:
                    continue
                print(f"   pruned pairs {len(P['qi']):,} "
                      f"({len(P['qi'])/len(q):.2f}/q) {time.time()-t0:.0f}s",
                      flush=True)
                npair = len(P["qi"])
                s1e = q.eid.values[P["qi"]]
                dse = sub.eid.values[P["di"]]
                lab = None
                if truth is not None:
                    pref = f"S{src}-"
                    tset = {}
                    for e in q.eid.values:
                        t = truth.get(e, "")
                        tset[e] = {int(x[3:]) for x in t.split(",")
                                   if x.startswith(pref)} if t else set()
                    lab = np.fromiter(
                        (1 if dse[i] in tset[s1e[i]] else 0
                         for i in range(npair)), np.int8, npair)
                    ntrue = sum(len(v) for v in tset.values())
                    print(f"   label recall {int(lab.sum())}/{ntrue} = "
                          f"{lab.sum()/max(1,ntrue):.4f}  pos rate "
                          f"{lab.mean():.4f}", flush=True)
                rows = None
                if a.keepfrac < 1.0:
                    rng2 = np.random.default_rng(29)
                    ek = rng2.random(len(q)) < a.keepfrac
                    rows = np.flatnonzero(ek[P["qi"]])
                res = dict(s1_all=q.eid.values.astype(np.int64))
                if truth is not None:
                    res["ntrue"] = np.int64(ntrue)
                if booster is not None:
                    pv = np.empty(npair, np.float32)
                    for sl, F in iter_features(P, Q, D, src == 3, len(q)):
                        pv[sl] = booster.predict(F, num_threads=THREADS)
                    print(f"   stage1 predicted {npair:,} "
                          f"{time.time()-t0:.0f}s", flush=True)
                    res["p1"] = pv.copy()
                    if booster2 is not None:
                        # the stage-2 competition block is computed over the
                        # whole partition, exactly as during training
                        St = stack_features(
                            P["qi"], dse, np.full(npair, src, np.int8),
                            np.zeros(npair, np.int8), pv, len(q))
                        p2 = np.empty(npair, np.float32)
                        for sl, F in iter_features(P, Q, D, src == 3, len(q)):
                            p2[sl] = booster2.predict(
                                np.hstack([F, St[sl]]), num_threads=THREADS)
                        pv = p2
                        print(f"   stage2 predicted {npair:,} "
                              f"{time.time()-t0:.0f}s", flush=True)
                    res["p"] = pv
                    sel = np.arange(npair)
                else:
                    F = features_for(P, Q, D, src == 3, len(q), rows=rows)
                    print(f"   features {F.shape} {time.time()-t0:.0f}s",
                          flush=True)
                    res["F"] = F
                    sel = np.arange(npair) if rows is None else rows
                res["s1"] = s1e[sel].astype(np.int64)
                res["d"] = dse[sel].astype(np.int64)
                res["qi"] = P["qi"][sel].astype(np.int32)
                res["prune"] = P["prune"][sel]
                if lab is not None:
                    res["y"] = lab[sel]
                    res["ntrue_kept"] = np.int64(
                        ntrue if rows is None else -1)
                np.savez(out, **res)
                print(f"   wrote {out} {time.time()-t0:.0f}s", flush=True)
                del D, Q, P, F, sub, q, QE, DE
            del dd


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
