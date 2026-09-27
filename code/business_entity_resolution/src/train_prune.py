"""Train the two learned prune rankers from a captured wide retrieval net.

    python runpipe.py train --nq 16000 --tag w4 ... BER_KEEP=900   # capture
    python train_prune.py --tag w4

Why two.  The candidate set that reaches the matcher is produced in two cuts:

*  **stage 1** ranks the whole retrieval net (~290 candidates per entity per
   source) using only components the retrieval pass already produced -- per
   channel scores and ranks, per-group shared evidence and its containment,
   four batched rapidfuzz similarities and the record shapes.  Everything is
   vectorised, so scoring the full net is affordable.
*  **stage 2** recomputes the full pair features for stage 1's top ``keep1``
   and re-ranks them.  Those features cost per-pair python work, so they can
   only be afforded on a shortlist.

Measured on held-out entities, as a fraction of the true pairs the retrieval
net contains:

    hand-tuned linear score, 30 per source   0.9924
    stage 1 alone,           20 per source   0.9936
    stage 1 -> stage 2,      15 per source   0.9985

which is better recall from half the candidates.
"""
import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import hw  # noqa: E402  (must precede numpy/torch)

import argparse
import glob
import os
import sys
import time

import numpy as np

from hw import THREADS

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lightgbm as lgb  # noqa: E402
from features import CHEAP_FEATS, FEAT_NAMES, PRUNE_FEATS  # noqa: E402

WORK = os.environ.get("BER_WORK", "../../../work")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="w4")
    ap.add_argument("--out", default="")
    ap.add_argument("--rounds", type=int, default=600)
    ap.add_argument("--leaves", type=int, default=127)
    ap.add_argument("--keep1", type=int, default=120)
    ap.add_argument("--keep2", type=int, default=15)
    a = ap.parse_args()
    t0 = time.time()
    IX = {n: i for i, n in enumerate(FEAT_NAMES)}
    F, Y, Q, ntrue, off = [], [], [], 0, 0
    for f in sorted(glob.glob(f"{WORK}/train_{a.tag}_s*.npz")):
        z = np.load(f)
        F.append(z["F"])
        Y.append(z["y"])
        Q.append(z["qi"].astype(np.int64) + off)
        off += len(z["s1_all"])
        ntrue += int(z["ntrue"])
        print(f"  {os.path.basename(f)}: {len(z['y']):,} pairs")
    F = np.concatenate(F)
    y = np.concatenate(Y)
    qi = np.concatenate(Q)
    nq = off
    print(f"net: {len(y):,} pairs over {nq:,} entity-slots, "
          f"recall {y.sum()/ntrue:.4f}, {len(y)/nq:.1f} per slot "
          f"({time.time()-t0:.0f}s)", flush=True)

    rng = np.random.default_rng(23)
    hold = rng.random(nq) < 0.25          # held-out entities, for reporting
    tr = ~hold[qi]
    params = dict(objective="binary", learning_rate=0.06, num_leaves=a.leaves,
                  min_data_in_leaf=80, feature_fraction=0.85,
                  bagging_fraction=0.8, bagging_freq=1, num_threads=THREADS,
                  verbose=-1, max_bin=127)

    def fit(cols, name):
        X = F[:, [IX[c] for c in cols]].astype(np.float32)
        m = lgb.train(params, lgb.Dataset(X[tr], label=y[tr]),
                      num_boost_round=a.rounds)
        s = m.predict(X, num_threads=THREADS).astype(np.float32)
        print(f"  {name}: {len(cols)} features, {time.time()-t0:.0f}s",
              flush=True)
        return m, s

    def rank(score, rows=None):
        idx = np.arange(len(score)) if rows is None else rows
        o = idx[np.lexsort((-score[idx], qi[idx]))]
        st = np.flatnonzero(np.r_[True, qi[o][1:] != qi[o][:-1]])
        pos = np.arange(len(o)) - np.repeat(st, np.diff(np.r_[st, len(o)]))
        r = np.full(len(score), 1 << 30, np.int64)
        r[o] = pos
        return r

    m1, s1 = fit(CHEAP_FEATS, "stage 1 (cheap)")
    m2, s2 = fit(PRUNE_FEATS, "stage 2 (full)")

    ev = hold[qi]
    tot = float(y[ev].sum())
    r1 = rank(s1)
    print(f"\nheld-out retention of the net's true pairs "
          f"(net recall {y.sum()/ntrue:.4f}):")
    for k in (15, 20, 30, 60, 120):
        print(f"  stage 1 alone, {k:>3} per source : "
              f"{float(y[ev & (r1 < k)].sum())/tot:.4f}")
    for k1 in (60, 120, 200):
        s2m = np.where(r1 < k1, s2, -1e9)
        r2 = rank(s2m)
        for k2 in (10, 15, 20):
            ret = float(y[ev & (r1 < k1) & (r2 < k2)].sum()) / tot
            print(f"  stage 1 top {k1:>3} -> stage 2 top {k2:>3} : {ret:.4f}"
                  f"   (blocking recall {ret*y.sum()/ntrue:.4f})")

    out = a.out or f"{WORK}/prune"
    m1.save_model(out + "_cheap.lgb")
    m2.save_model(out + "_full.lgb")
    print(f"\nsaved {out}_cheap.lgb and {out}_full.lgb")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
