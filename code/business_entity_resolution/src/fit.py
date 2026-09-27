"""Train the matcher and tune the decision layer.

    python fit.py --tag full

Two stacked LightGBM stages are fitted on an entity-disjoint split:

  stage 1  the pair features from ``features.py``
  stage 2  the same features plus *competition* features derived from the
           stage-1 probability -- how strongly this Source-2/3 record is
           claimed by some other entity, and where this candidate ranks inside
           its own entity.  The partition constraint (each record belongs to at
           most one entity) is what makes those features informative.

Finally the decision layer (exclusivity resolution + expected-F_0.5 set
selection) is grid-searched against macro F_0.5 on the held-out entities.
"""
import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import hw  # noqa: E402  (must precede numpy/torch)

import argparse
import glob
import json
import os
import sys
import time

import numpy as np

from hw import THREADS
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lightgbm as lgb  # noqa: E402
from decide import resolve_exclusive, score_flat, select_sets  # noqa: E402
from features import FEAT_NAMES  # noqa: E402
from stack import STACK_NAMES, stack_features  # noqa: E402

WORK = os.environ.get("BER_WORK", "../../../work")
DATA = os.environ.get("BER_DATA", "../../../dataset")


def load(tag):
    """Concatenate all partitions, mapping entities onto one global index."""
    files = sorted(glob.glob(f"{WORK}/train_{tag}_s*.npz"))
    assert files, "no partition files"
    Fs, ys, s1s, ds, srcs, ents, part = [], [], [], [], [], [], []
    for i, f in enumerate(files):
        z = np.load(f)
        Fs.append(z["F"])
        ys.append(z["y"])
        s1s.append(z["s1"])
        ds.append(z["d"])
        srcs.append(np.full(len(z["s1"]), 2 if "_s2_" in f else 3, np.int8))
        part.append(np.full(len(z["s1"]), i, np.int8))
        ents.append(z["s1_all"])
        print(f"  {os.path.basename(f)}: pairs={len(z['y']):,} "
              f"kept-recall={z['y'].sum()/max(1,int(z['ntrue'])):.4f}")
    F = np.concatenate(Fs); del Fs
    y = np.concatenate(ys)
    s1 = np.concatenate(s1s)
    d = np.concatenate(ds)
    src = np.concatenate(srcs)
    part = np.concatenate(part)
    # runpipe stores features for a *sample* of entities (--keepfrac), so the
    # evaluation universe is the entities that actually have candidates here.
    # Using every Source-1 id would score the un-sampled ones as total misses.
    ent = np.unique(s1)
    all_ent = np.unique(np.concatenate(ents))
    print(f"  kept {len(ent):,} of {len(all_ent):,} Source-1 entities "
          f"({len(ent)/len(all_ent):.3f})")
    qi = np.searchsorted(ent, s1)
    return F, y, qi, d, src, part, ent


def truth_counts(ent):
    gt = pd.read_csv(f"{DATA}/train/train_ground_truth.tsv", sep="\t",
                     dtype=str, keep_default_na=False)
    e = gt.source1_entity_id.str.slice(3).astype(np.int64).values
    m = np.isin(e, ent)
    e = e[m]
    lists = gt.matched_entity_ids.values[m]
    cnt = np.zeros(len(ent), np.int32)
    cnt[np.searchsorted(ent, e)] = [0 if s == "" else s.count(",") + 1
                                    for s in lists]
    return cnt


def metric_weights(nt, qi, mode="metric"):
    """Weight each pair by how much one mistake on it moves macro F_0.5.

    Since F_0.5 = 1.25*TP/(|S| + 0.25*|T|), one extra false positive or one
    missed match costs about 1/(|S| + 0.25*|T|) ~ 1/(1.25*|T|).  Entities with
    few true matches are therefore far more expensive to get wrong, and a false
    merge onto a singleton costs the full 1.0.  Training with these weights
    aligns the loss with the metric instead of treating all pairs alike.
    """
    t = nt[qi].astype(np.float32)
    w = 1.0 / (1.25 * np.maximum(t, 0.8))
    if mode == "sqrt":
        w = np.sqrt(w)
    return (w / w.mean()).astype(np.float32)


def train_stage(X, y, names, rounds, leaves, lr, seed, rng, w=None, es=None,
                use_es=True):
    """Fit one boosting stage.

    ``es`` is a boolean mask selecting an entity-disjoint early-stopping slice;
    without it early stopping would watch rows the trees already fitted and
    would never fire.
    """
    if es is None:
        es = rng.random(len(y)) < 0.02
    tr = ~es
    dtr = lgb.Dataset(X[tr], label=y[tr],
                      weight=None if w is None else w[tr],
                      feature_name=names, free_raw_data=True)
    dva = lgb.Dataset(X[es], label=y[es],
                      weight=None if w is None else w[es], reference=dtr)
    params = dict(objective="binary", metric=["binary_logloss", "auc"],
                  learning_rate=lr, num_leaves=leaves, min_data_in_leaf=100,
                  feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
                  lambda_l2=2.0, num_threads=THREADS, verbose=-1, max_bin=255,
                  seed=seed)
    cb = [lgb.log_evaluation(200)]
    if use_es:
        cb.append(lgb.early_stopping(80, verbose=True))
    return lgb.train(params, dtr, num_boost_round=rounds,
                     valid_sets=[dva], valid_names=["h"], callbacks=cb)


def tune_decision(qi, p, y, di, ndoc, nq, vpair, isval, vnt, verbose=True,
                  sub=None):
    """Grid-search the decision layer.

    Exclusivity is resolved over *all* pairs, not just the validation ones, so
    the competition a record faces matches inference time; only the scoring is
    restricted to held-out entities.
    """
    best = None
    for excl in (True, False):
        pp = p.copy()
        if excl:
            keep = resolve_exclusive(di, pp, ndoc)
            pp = np.where(keep, pp, np.float32(0.0))
        m = vpair if sub is None else (vpair & sub[qi])
        ev = isval if sub is None else (isval & sub)
        vq, vp, vy = qi[m], pp[m], y[m]
        for pw in (0.7, 1.0, 1.3, 1.7, 2.2, 3.0):
            for m0 in (0.0, 0.1, 0.2, 0.3, 0.4, 0.55, 0.7):
                for k0 in (0.5, 1.0, 2.0, 3.5, 5.0, 7.0, 10.0, 14.0):
                    for pmin in (0.0, 0.05, 0.12, 0.25, 0.4, 0.55):
                        sel = select_sets(vq, vp, nq, m0=m0, pmin=pmin,
                                          k0_boost=k0, pw=pw)
                        _, fa = score_flat(vq, sel, vy, nq, vnt)
                        f = float(fa[ev].mean())
                        if best is None or f > best[0]:
                            best = (f, dict(excl=excl, m0=m0, k0=k0,
                                            pmin=pmin, pw=pw))
        if verbose:
            print(f"  excl={excl}: best {best[0]:.5f} {best[1]}", flush=True)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="full")
    ap.add_argument("--rounds", type=int, default=900)
    ap.add_argument("--oofrounds", type=int, default=0,
                    help="rounds for the two out-of-fold stage-1 models "
                         "(they only produce the stacking input)")
    ap.add_argument("--leaves", type=int, default=160)
    ap.add_argument("--lr", type=float, default=0.06)
    ap.add_argument("--valfrac", type=float, default=0.3)
    ap.add_argument("--out", default="")
    ap.add_argument("--maxtrain", type=int, default=6_500_000,
                    help="cap on training rows, to bound the peak RSS of the "
                         "F[tr] copy")
    ap.add_argument("--nostack", action="store_true",
                    help="skip the stage-2 stacking (saves the out-of-fold "
                         "models and ~5 GB of peak RAM)")
    ap.add_argument("--es", type=int, default=1,
                    help="0 disables early stopping; logloss on a held-out "
                         "slice bottoms out long before macro F_0.5 does, so "
                         "more rounds keep helping")
    ap.add_argument("--weight", default="none",
                    choices=["none", "metric", "sqrt"],
                    help="metric-aligned pair weights; 'none' keeps the "
                         "probabilities calibrated, which the expected-F_0.5 "
                         "selection relies on")
    a = ap.parse_args()
    t0 = time.time()
    F, y, qi, d, src, part, ent = load(a.tag)
    nq = len(ent)
    nt = truth_counts(ent)
    print(f"pairs={len(y):,} entities={nq:,} true={nt.sum():,} "
          f"recall={y.sum()/nt.sum():.4f} pos={y.mean():.4f} "
          f"cand/entity={len(y)/nq:.2f} ({time.time()-t0:.0f}s)", flush=True)

    rng = np.random.default_rng(17)
    wt = None if a.weight == "none" else metric_weights(nt, qi, a.weight)
    u = rng.random(nq)
    isval = u < a.valfrac
    # a slice of *training* entities reserved for early stopping only
    ises = (u >= a.valfrac) & (u < a.valfrac + 0.03)
    vpair = isval[qi]
    tr = ~vpair
    if tr.sum() > a.maxtrain:
        keepr = rng.random(len(y)) < a.maxtrain / tr.sum()
        tr = tr & keepr
    print(f"train pairs={tr.sum():,} val pairs={vpair.sum():,}", flush=True)

    # ---- stage 1, with out-of-fold predictions for the stacking input ----
    fold = (rng.random(nq) < 0.5)
    p1 = np.zeros(len(y), np.float32)
    for f in ([] if a.nostack else (False, True)):
        sub = tr & (fold[qi] == f)
        oth = tr & (fold[qi] != f)
        mf = train_stage(F[sub], y[sub], FEAT_NAMES,
                         a.oofrounds or a.rounds, a.leaves, a.lr,
                         10 + int(f), rng, w=None if wt is None else wt[sub],
                         es=ises[qi[sub]], use_es=bool(a.es))
        p1[oth] = mf.predict(F[oth], num_threads=THREADS)
        print(f"  oof fold {int(f)}: {mf.num_trees()} trees "
              f"{time.time()-t0:.0f}s", flush=True)
        del mf
    m1 = train_stage(F[tr], y[tr], FEAT_NAMES, a.rounds, a.leaves, a.lr, 1,
                     rng, w=None if wt is None else wt[tr], es=ises[qi[tr]],
                     use_es=bool(a.es))
    print(f"stage1: {m1.num_trees()} trees {time.time()-t0:.0f}s", flush=True)
    if a.nostack:
        p1 = m1.predict(F, num_threads=THREADS).astype(np.float32)
    else:
        p1[vpair] = m1.predict(F[vpair], num_threads=THREADS).astype(np.float32)

    # ---- stage 2: competition features derived from the stage-1 score ----
    if a.nostack:
        m2, p2, names2 = m1, p1, FEAT_NAMES
        del F
    else:
        S = stack_features(qi, d, src, part, p1, nq)
        X2 = np.hstack([F, S])
        del F                   # the stage-1 matrix is no longer needed
        names2 = FEAT_NAMES + STACK_NAMES
        m2 = train_stage(X2[tr], y[tr], names2, a.rounds, a.leaves, a.lr, 2,
                         rng, w=None if wt is None else wt[tr],
                         es=ises[qi[tr]], use_es=bool(a.es))
        print(f"stage2: {m2.num_trees()} trees {time.time()-t0:.0f}s",
              flush=True)
        p2 = m2.predict(X2, num_threads=THREADS).astype(np.float32)
        del X2

    imp = sorted(zip(names2, m2.feature_importance("gain")), key=lambda z: -z[1])
    print("\ntop stage-2 features by gain:")
    for k, v in imp[:28]:
        print(f"   {k:14s} {v:12.0f}")

    out = a.out or f"{WORK}/model_{a.tag}"
    m1.save_model(out + "_s1.lgb")
    m2.save_model(out + "_s2.lgb")
    print("saved models (before the decision search, so a long search can be "
          "interrupted safely)", flush=True)

    dk = d.astype(np.int64) * 4 + src
    ud, di = np.unique(dk, return_inverse=True)
    vnt = np.where(isval, nt, 0)
    print(f"\nval entities={isval.sum():,} true={vnt.sum():,}", flush=True)
    # keep the validation arrays so the decision layer can be retuned in
    # seconds without refitting anything
    np.savez(f"{WORK}/valpred_{a.tag}.npz", qi=qi.astype(np.int32),
             p1=p1, p2=p2, y=y, di=di.astype(np.int32), ndoc=np.int64(len(ud)),
             nq=np.int64(nq), vpair=vpair, isval=isval, nt=nt, ent=ent)
    print("saved validation predictions", flush=True)

    # coarse search on a subsample of held-out entities, then verify in full
    # cap the coarse pass: the grid is ~4000 points and each one sorts every
    # validation pair, so it has to run on a slice
    sub = rng.random(nq) < min(1.0, 12000.0 / max(1, int(isval.sum())))
    for nm, pv in (("stage1", p1), ("stage2", p2)):
        b = tune_decision(qi, pv, y, di, len(ud), nq, vpair, isval, vnt,
                          verbose=False, sub=sub)
        print(f"  {nm} (subsample): F0.5={b[0]:.5f}  {b[1]}", flush=True)
    best = tune_decision(qi, p2, y, di, len(ud), nq, vpair, isval, vnt,
                         sub=sub)
    cfg = best[1]
    pp = p2.copy()
    if cfg["excl"]:
        keep = resolve_exclusive(di, pp, len(ud))
        pp = np.where(keep, pp, np.float32(0.0))
    sel = select_sets(qi[vpair], pp[vpair], nq, m0=cfg["m0"],
                      pmin=cfg["pmin"], k0_boost=cfg["k0"], pw=cfg["pw"])
    _, fa = score_flat(qi[vpair], sel, y[vpair], nq, vnt)
    full = float(fa[isval].mean())
    print(f"  chosen config on ALL held-out entities: F0.5={full:.5f}",
          flush=True)
    best = (full, cfg)
    for th in (0.4, 0.5, 0.6, 0.7):
        _, fa = score_flat(qi[vpair], p2[vpair] >= th, y[vpair], nq, vnt)
        print(f"  plain threshold {th}: F0.5={float(fa[isval].mean()):.5f}")
    print(f"\nBEST F0.5 = {best[0]:.5f}  {best[1]}")

    json.dump(best[1], open(out + "_decide.json", "w"))
    print("saved", out + "_s{1,2}.lgb and the decision config")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
