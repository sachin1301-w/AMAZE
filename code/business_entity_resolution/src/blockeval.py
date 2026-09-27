"""Measure blocking recall / candidate cost at full corpus scale on train."""
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
NQ = int(os.environ.get("BER_NQ", "20000"))
BUDGET = int(os.environ.get("BER_BUDGET", "20000"))
KN = int(os.environ.get("BER_KN", "10"))
KA = int(os.environ.get("BER_KA", "10"))
KC = int(os.environ.get("BER_KC", "10"))
MAXDF = float(os.environ.get("BER_MAXDF", "0.02"))
BETA = float(os.environ.get("BER_BETA", "1.0"))
MODE = os.environ.get("BER_MODE", "fs")
FA = float(os.environ.get("BER_A", "2.4"))
FB = float(os.environ.get("BER_B", "1.2"))
IPOW = float(os.environ.get("BER_IPOW", "1.0"))
ONLY = os.environ.get("BER_ONLY", "")

CHANS = [("name", CH_NAME, KN), ("addr", CH_ADDR, KA), ("all", CH_ALL, KC)]


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

    ntrue = sum(len(v) for v in truth)
    tot_found, tot_cand = 0, 0
    per_chan = {c[0]: 0 for c in CHANS}
    dftab = os.environ.get("BER_DFTAB", f"{WORK}/train_dftab.npz")
    from blocking import _init_worker
    with mp.Pool(THREADS, initializer=_init_worker, initargs=(dftab,)) as pool:
        for src in (2, 3):
            dd = pd.read_parquet(f"{WORK}/train_s{src}.parquet")
            for ctry in sorted(dd.country.astype(str).unique()):
                if ONLY and f"{src}{ctry}" != ONLY:
                    continue
                sub = dd[(dd.country.astype(str) == ctry).values].reset_index(drop=True)
                q = d1[(d1.country.astype(str) == ctry).values].reset_index(drop=True)
                if not len(q):
                    continue
                th, ti, tc = hash_terms_par(sub.nname.values, sub.naddr.values,
                                            sub.dom.values, pool)
                qh, qi, qc = hash_terms_par(q.nname.values, q.naddr.values,
                                            q.dom.values, pool)
                eids = sub.eid.values
                qmap = q.qi.values
                pref = f"S{src}-"
                trows = [set() for _ in range(len(q))]
                for k in range(len(q)):
                    for x in truth[qmap[k]]:
                        if x.startswith(pref):
                            trows[k].add(int(x[3:]))
                tt = sum(len(x) for x in trows)
                cand = [set() for _ in range(len(q))]
                for cname, ctypes, KK in CHANS:
                    idx = Index(th, ti, tc, len(sub), types=ctypes,
                                maxdf_frac=MAXDF)
                    Q, D, S = topk_query(idx, qh, qc, qi, len(q), KK, BUDGET,
                                         types=ctypes, beta=BETA,
                                         mode=MODE, A=FA, B=FB, ipow=IPOW)
                    h = 0
                    for a, b in zip(Q, D):
                        cand[a].add(int(b))
                        if eids[b] in trows[a]:
                            h += 1
                    per_chan[cname] += h
                    print(f"  s{src} {ctry:7s} {cname:4s} recall={h/max(1,tt):.4f} "
                          f"{time.time()-t0:.0f}s", flush=True)
                    del idx
                found = sum(1 for k in range(len(q))
                            for b in cand[k] if eids[b] in trows[k])
                nc = sum(len(c) for c in cand)
                tot_found += found
                tot_cand += nc
                print(f"  s{src} {ctry:7s} UNION recall={found/max(1,tt):.4f} "
                      f"cand/q={nc/len(q):.2f}", flush=True)
                del sub, th, ti, tc
            del dd
    print(f"\nKN={KN} KA={KA} KC={KC} BUDGET={BUDGET} MAXDF={MAXDF}")
    for c in CHANS:
        print(f"  channel {c[0]:5s} recall={per_chan[c[0]]/ntrue:.5f}")
    print(f"  UNION recall={tot_found/ntrue:.5f}  cand/S1={tot_cand/len(d1):.2f}"
          f"  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
