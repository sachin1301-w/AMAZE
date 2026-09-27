"""Grid-search blocking scoring configs; one index build, many evaluations."""
import os
import sys
import time
import itertools
import multiprocessing as mp

import numpy as np

from hw import THREADS
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from blocking import Index, hash_terms_par, topk_query

WORK = os.environ.get("BER_WORK", "../../../work")
DATA = os.environ.get("BER_DATA", "../../../dataset")
NQ = int(os.environ.get("BER_NQ", "20000"))
KS = (10, 15, 20, 25, 40, 60, 100)
KMAX = 100


def main():
    t0 = time.time()
    gt = pd.read_csv(f"{DATA}/train/train_ground_truth.tsv", sep="\t",
                     dtype=str, keep_default_na=False)
    rng = np.random.default_rng(7)
    gt = gt.iloc[np.sort(rng.choice(len(gt), NQ, replace=False))].reset_index(drop=True)
    s1q = gt.source1_entity_id.str.slice(3).astype(np.int64).values
    truth = [set(s.split(",")) if s else set() for s in gt.matched_entity_ids.values]
    d1 = pd.read_parquet(f"{WORK}/train_s1.parquet")
    d1 = d1[d1.eid.isin(set(s1q.tolist()))].reset_index(drop=True)
    pos = {e: i for i, e in enumerate(s1q)}
    d1["qi"] = [pos[e] for e in d1.eid.values]

    configs = []
    for b in (0.0, 0.4, 0.75):
        for gw in (1.0, 0.5, 0.25):
            configs.append(dict(b=b, tw=(1.0, 1.0, 1.0, gw)))
    configs.append(dict(b=0.4, tw=(1.3, 1.0, 0.8, 0.35)))
    configs.append(dict(b=0.4, tw=(1.0, 1.2, 1.0, 0.35)))
    res = {i: np.zeros(len(KS)) for i in range(len(configs))}
    ntrue = sum(len(v) for v in truth)

    with mp.Pool(THREADS) as pool:
        for src in (2, 3):
            dd = pd.read_parquet(f"{WORK}/train_s{src}.parquet")
            for ctry in dd.country.astype(str).unique():
                sub = dd[(dd.country.astype(str) == ctry).values]
                q = d1[(d1.country.astype(str) == ctry).values]
                if not len(q):
                    continue
                th, ti, tc = hash_terms_par(sub.nname.values, sub.naddr.values,
                                            sub.dom.values, pool)
                idx = Index(th, ti, tc, len(sub))
                del th, ti, tc
                qh, qi, qc = hash_terms_par(q.nname.values, q.naddr.values,
                                            q.dom.values, pool)
                eids = sub.eid.values
                qmap = q.qi.values
                pref = f"S{src}-"
                # truth as doc-row ids per query row
                trows = [set() for _ in range(len(q))]
                e2r = None
                for ci, cfg in enumerate(configs):
                    Q, D, S = topk_query(idx, qh, qi, len(q), KMAX, 20000,
                                         tw=cfg["tw"], b=cfg["b"])
                    if e2r is None:
                        e2r = {}
                        for k in range(len(q)):
                            for x in truth[qmap[k]]:
                                if x.startswith(pref):
                                    trows[k].add(int(x[3:]))
                    # rank within each query
                    o = np.lexsort((-S, Q))
                    Q, D = Q[o], D[o]
                    qs = np.searchsorted(Q, np.arange(len(q) + 1))
                    hits = np.zeros(len(KS))
                    for k in range(len(q)):
                        if not trows[k]:
                            continue
                        a, e = qs[k], qs[k + 1]
                        dl = eids[D[a:e]]
                        for r, did in enumerate(dl):
                            if did in trows[k]:
                                for j, KK in enumerate(KS):
                                    if r < KK:
                                        hits[j] += 1
                    res[ci] += hits
                    print(f"  s{src} {ctry:6s} cfg{ci} "
                          f"r@25={hits[KS.index(25)]/max(1,sum(len(t) for t in trows)):.4f} "
                          f"{time.time()-t0:.0f}s", flush=True)
                del idx, sub
            del dd
    print("\n%-42s %s" % ("config", "  ".join("r@%d" % k for k in KS)))
    for ci, cfg in enumerate(configs):
        print("%-42s %s" % (str(cfg),
              "  ".join("%.4f" % (v / ntrue) for v in res[ci])))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
