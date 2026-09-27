"""Decompose the matcher's loss against the oracle, from saved predictions.

    python diagnose.py --tag full
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from decide import resolve_exclusive, score_flat, select_sets

WORK = os.environ.get("BER_WORK", "../../../work")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="full")
    ap.add_argument("--decide", default="")
    a = ap.parse_args()
    z = np.load(f"{WORK}/valpred_{a.tag}.npz")
    qi, p, y = z["qi"], z["p2"], z["y"]
    di, ndoc, nq = z["di"], int(z["ndoc"]), int(z["nq"])
    vpair, isval, nt = z["vpair"], z["isval"], z["nt"]
    vnt = np.where(isval, nt, 0)
    cfg = dict(excl=True, m0=0.4, k0=6.0, pmin=0.0, pw=1.0)
    if a.decide and os.path.exists(a.decide):
        cfg.update(json.load(open(a.decide)))
    print("decision config:", cfg)

    vq, vy = qi[vpair], y[vpair]
    nval = int(isval.sum())
    print(f"val entities={nval:,} pairs={len(vq):,} true(in GT)={int(vnt.sum()):,}"
          f"  true(in candidates)={int(vy.sum()):,}")

    # ---- oracle -----------------------------------------------------------
    _, fo = score_flat(vq, vy > 0, vy, nq, vnt)
    print(f"\nORACLE  F0.5 = {fo[isval].mean():.5f}")

    pp = p.copy()
    if cfg["excl"]:
        keep = resolve_exclusive(di, pp, ndoc)
        pp = np.where(keep, pp, np.float32(0.0))
    sel = select_sets(vq, pp[vpair], nq, m0=cfg["m0"], pmin=cfg["pmin"],
                      k0_boost=cfg["k0"], pw=cfg["pw"])
    _, fm = score_flat(vq, sel, vy, nq, vnt)
    print(f"MODEL   F0.5 = {fm[isval].mean():.5f}"
          f"   (deficit {fo[isval].mean()-fm[isval].mean():.5f})")

    # ---- where the deficit sits -------------------------------------------
    ns = np.bincount(vq[sel], minlength=nq)
    tp = np.bincount(vq[sel & (vy > 0)], minlength=nq)
    fp = ns - tp
    inc = np.bincount(vq[vy > 0], minlength=nq)      # true & in candidates
    fn = inc - tp
    loss = (fo - fm)
    m = isval
    print(f"\nper-entity counts (held-out): predicted {ns[m].mean():.3f}, "
          f"TP {tp[m].mean():.3f}, FP {fp[m].mean():.3f}, "
          f"FN-in-candidates {fn[m].mean():.3f}, "
          f"FN-from-blocking {(vnt-inc)[m].mean():.3f}")
    print(f"precision {tp[m].sum()/max(1,ns[m].sum()):.5f}  "
          f"recall {tp[m].sum()/max(1,vnt[m].sum()):.5f}")

    clean = m & (fp == 0) & (fn == 0)
    print(f"\nentities with no matcher mistake: {clean[m].mean():.4f}")
    only_fp = m & (fp > 0) & (fn == 0)
    only_fn = m & (fp == 0) & (fn > 0)
    both = m & (fp > 0) & (fn > 0)
    for nm, k in (("FP only", only_fp), ("FN only", only_fn), ("both", both)):
        if k.sum():
            print(f"  {nm:8s}: {k[m].mean():.4f} of entities, "
                  f"mean loss {loss[k].mean():.4f}, "
                  f"share of total deficit "
                  f"{loss[k].sum()/max(1e-9,loss[m].sum()):.3f}")

    # singleton entities specifically
    sg = m & (vnt == 0)
    print(f"\nsingletons: {sg[m].mean():.4f} of entities, "
          f"F0.5 {fm[sg].mean():.5f}, "
          f"share of deficit {loss[sg].sum()/max(1e-9,loss[m].sum()):.3f}")
    for k in range(1, 7):
        kk = m & (vnt == k)
        if kk.sum() > 50:
            print(f"  |T|={k}: {kk[m].mean():.4f} of entities  "
                  f"oracle {fo[kk].mean():.4f}  model {fm[kk].mean():.4f}  "
                  f"share of deficit {loss[kk].sum()/max(1e-9,loss[m].sum()):.3f}")

    # ---- score separation -------------------------------------------------
    pt, pf = pp[vpair][vy > 0], pp[vpair][vy == 0]
    print(f"\np(true)  median {np.median(pt):.4f}  10th pct {np.quantile(pt,.1):.4f}"
          f"  frac<0.5 {(pt<0.5).mean():.4f}")
    print(f"p(false) median {np.median(pf):.4f}  99th pct {np.quantile(pf,.99):.4f}"
          f"  frac>0.5 {(pf>0.5).mean():.4f}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
