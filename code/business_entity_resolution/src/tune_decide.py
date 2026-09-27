"""Retune the decision layer from saved validation predictions.

`fit.py` dumps work/valpred_<tag>.npz; this searches the decision-layer
parameters against macro F_0.5 without refitting anything, which makes the
search cheap enough to explore properly.

    python tune_decide.py --tag full --sub 60000
"""
import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import hw  # noqa: E402  (must precede numpy/torch)

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from decide import resolve_exclusive, score_flat, select_sets

WORK = os.environ.get("BER_WORK", "../../../work")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="full")
    ap.add_argument("--sub", type=int, default=60000,
                    help="entities used for the coarse pass")
    ap.add_argument("--stage", default="p2", choices=["p1", "p2"])
    ap.add_argument("--save", action="store_true")
    a = ap.parse_args()
    t0 = time.time()
    z = np.load(f"{WORK}/valpred_{a.tag}.npz")
    qi, p, y = z["qi"], z[a.stage], z["y"]
    di, ndoc, nq = z["di"], int(z["ndoc"]), int(z["nq"])
    vpair, isval, nt = z["vpair"], z["isval"], z["nt"]
    vnt = np.where(isval, nt, 0)
    print(f"val entities={int(isval.sum()):,} pairs={int(vpair.sum()):,} "
          f"true={int(vnt.sum()):,}", flush=True)

    rng = np.random.default_rng(3)
    sub = rng.random(nq) < min(1.0, a.sub / max(1, int(isval.sum())))

    def run(cfg, mask_sub):
        pp = p.copy()
        if cfg["excl"]:
            keep = resolve_exclusive(di, pp, ndoc)
            pp = np.where(keep, pp, np.float32(0.0))
        m = vpair & (sub[qi] if mask_sub else np.ones(len(qi), bool))
        ev = isval & (sub if mask_sub else np.ones(nq, bool))
        sel = select_sets(qi[m], pp[m], nq, m0=cfg["m0"], pmin=cfg["pmin"],
                          k0_boost=cfg["k0"], pw=cfg["pw"])
        _, fa = score_flat(qi[m], sel, y[m], nq, vnt)
        return float(fa[ev].mean())

    results = []
    for excl in (True, False):
        # resolve exclusivity once per branch, then sweep the cheap parameters
        pp = p.copy()
        if excl:
            keep = resolve_exclusive(di, pp, ndoc)
            pp = np.where(keep, pp, np.float32(0.0))
        m = vpair & sub[qi]
        ev = isval & sub
        vq, vp, vy = qi[m], pp[m], y[m]
        for pw in (0.7, 1.0, 1.3, 1.7, 2.2, 3.0, 4.0):
            for m0 in (0.0, 0.1, 0.2, 0.3, 0.4, 0.55, 0.7, 0.9):
                for k0 in (0.5, 1.0, 2.0, 3.5, 5.0, 7.0, 10.0, 14.0, 20.0):
                    for pmin in (0.0, 0.05, 0.12, 0.25, 0.4, 0.55):
                        sel = select_sets(vq, vp, nq, m0=m0, pmin=pmin,
                                          k0_boost=k0, pw=pw)
                        _, fa = score_flat(vq, sel, vy, nq, vnt)
                        results.append((float(fa[ev].mean()),
                                        dict(excl=excl, m0=m0, k0=k0,
                                             pmin=pmin, pw=pw)))
        print(f"  excl={excl} done {time.time()-t0:.0f}s "
              f"best so far {max(r[0] for r in results):.5f}", flush=True)

    results.sort(key=lambda r: -r[0])
    print("\ntop 8 on the coarse subsample:")
    for f, c in results[:8]:
        print(f"   {f:.5f}  {c}")
    print("\nverifying the top 5 on all held-out entities:", flush=True)
    best = None
    for f, c in results[:5]:
        full = run(c, False)
        print(f"   {full:.5f}  (sub {f:.5f})  {c}", flush=True)
        if best is None or full > best[0]:
            best = (full, c)
    print(f"\nBEST F0.5 = {best[0]:.5f}  {best[1]}")
    if a.save:
        json.dump(best[1], open(f"{WORK}/model_{a.tag}_decide.json", "w"))
        print("saved decision config")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
