"""How much recall is reachable at all, and how much is lost to ranking?

Retrieves *every* document that shares an in-budget term with the query
(K = infinity) and reports the ceiling plus recall at several K for the
candidate scoring functions.
"""
import os
import sys
import time
import multiprocessing as mp

import numpy as np

from hw import THREADS
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from blocking import Index, hash_terms_par, topk_query
from terms import CH_NAME, CH_ADDR, CH_ALL

WORK = os.environ.get("BER_WORK", "../../../work")
DATA = os.environ.get("BER_DATA", "../../../dataset")
NQ = int(os.environ.get("BER_NQ", "3000"))
SRC = int(os.environ.get("BER_SRC", "3"))
CTRY = os.environ.get("BER_CTRY", "India")
BUDGET = int(os.environ.get("BER_BUDGET", "20000"))
MAXDF = float(os.environ.get("BER_MAXDF", "0.02"))
KS = (10, 20, 40, 80, 160, 400, 10 ** 9)


def main():
    t0 = time.time()
    gt = pd.read_csv(f"{DATA}/train/train_ground_truth.tsv", sep="\t",
                     dtype=str, keep_default_na=False)
    rng = np.random.default_rng(11)
    gt = gt.iloc[np.sort(rng.choice(len(gt), NQ, replace=False))].reset_index(drop=True)
    s1q = gt.source1_entity_id.str.slice(3).astype(np.int64).values
    truth = [set(s.split(",")) if s else set() for s in gt.matched_entity_ids.values]
    d1 = pd.read_parquet(f"{WORK}/train_s1.parquet")
    d1 = d1[d1.eid.isin(set(s1q.tolist()))].reset_index(drop=True)
    pos = {e: i for i, e in enumerate(s1q)}
    d1["qi"] = [pos[e] for e in d1.eid.values]
    d1 = d1[(d1.country.astype(str) == CTRY).values].reset_index(drop=True)

    dd = pd.read_parquet(f"{WORK}/train_s{SRC}.parquet")
    sub = dd[(dd.country.astype(str) == CTRY).values].reset_index(drop=True)
    del dd
    pref = f"S{SRC}-"
    trows = []
    for k in range(len(d1)):
        trows.append({int(x[3:]) for x in truth[d1.qi.values[k]]
                      if x.startswith(pref)})
    tt = sum(len(x) for x in trows)
    eids = sub.eid.values
    e2r = {e: i for i, e in enumerate(eids)}
    trows = [{e2r[e] for e in s if e in e2r} for s in trows]
    print(f"queries={len(d1)} docs={len(sub):,} true={tt}", flush=True)

    dftab = os.environ.get("BER_DFTAB", f"{WORK}/train_dftab.npz")
    from blocking import _init_worker
    with mp.Pool(THREADS, initializer=_init_worker, initargs=(dftab,)) as pool:
        th, ti, tc = hash_terms_par(sub.nname.values, sub.naddr.values,
                                    sub.dom.values, pool)
        qh, qi, qc = hash_terms_par(d1.nname.values, d1.naddr.values,
                                    d1.dom.values, pool)
    print(f"terms hashed {time.time()-t0:.0f}s", flush=True)

    chans = os.environ.get("BER_CHANS", "name,addr,all").split(",")
    CH = {"name": CH_NAME, "addr": CH_ADDR, "all": CH_ALL}
    for cname in chans:
        ctypes = CH[cname]
        idx = Index(th, ti, tc, len(sub), types=ctypes, maxdf_frac=MAXDF)
        print(f"  [{cname}] terms={len(idx.terms):,} postings={len(idx.post):,}",
              flush=True)
        for label, kw in (("cos", dict(mode="cos", beta=1.0)),):
            tq = time.time()
            Q, D, S = topk_query(idx, qh, qc, qi, len(d1), 10 ** 9, BUDGET,
                                 types=ctypes, **kw)
            tq = time.time() - tq
            o = np.lexsort((-S, Q))
            Q, D = Q[o], D[o]
            qs = np.searchsorted(Q, np.arange(len(d1) + 1))
            hits = np.zeros(len(KS))
            ncand = len(Q)
            for k in range(len(d1)):
                if not trows[k]:
                    continue
                a, e = qs[k], qs[k + 1]
                for r, did in enumerate(D[a:e]):
                    if did in trows[k]:
                        for j, KK in enumerate(KS):
                            if r < KK:
                                hits[j] += 1
            print(f"  {cname:4s} {label:6s} {1000*tq/len(d1):5.2f}ms/q cand/q={ncand/len(d1):8.1f}  " +
                  "  ".join(f"@{KK if KK < 10**8 else 'inf'}={h/tt:.4f}"
                            for KK, h in zip(KS, hits)), flush=True)
        del idx


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
