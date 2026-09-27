"""Tune the deterministic lexical prune score on a captured wide net.

Random search over a linear combination of interpretable similarity/rarity
components, maximising recall at the target candidate budget.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from features import FEAT_NAMES

WORK = os.environ.get("BER_WORK", "../../../work")
IX = {n: i for i, n in enumerate(FEAT_NAMES)}
# components offered to the search (all are per-pair or per-query-normalised,
# hence available before the matcher runs)
# only components that exist before the matcher runs
from features import DERIV  # noqa: E402
COMP = (["nm_ratio", "nm_partial", "cj_jw", "ad_tsr"]
        + ["sc_all", "sc_name", "sc_addr", "nchan", "sc_rel_w"]
        + [f"gm{g}" for g in range(4)] + [f"cd{g}" for g in range(4)]
        + [f"cq{g}" for g in range(4)] + DERIV)
KS = (10, 15, 20, 25, 30, 40, 60)
TK = int(os.environ.get("BER_TUNE_K", "30"))


def recall_at(score, qi, y, nq, ks=KS, ntrue=None):
    o = np.lexsort((-score, qi))
    q, yy = qi[o], y[o]
    qs = np.searchsorted(q, np.arange(nq + 1))
    pos = np.arange(len(q)) - np.repeat(qs[:-1], np.diff(qs))
    tot = ntrue if ntrue else int(y.sum())
    return [float(yy[(pos < k)].sum()) / tot for k in ks]


def main():
    files = sys.argv[1:]
    Fs, ys, qis, nqs, nts = [], [], [], [], []
    off = 0
    for f in files:
        z = np.load(f)
        Fs.append(z["F"])
        ys.append(z["y"])
        qis.append(z["qi"].astype(np.int64) + off)
        off += len(z["s1_all"])
        nts.append(int(z["ntrue"]))
    F = np.concatenate(Fs)
    y = np.concatenate(ys)
    qi = np.concatenate(qis)
    nq = off
    ntrue = sum(nts)
    # tune on a subsample of entities: the search needs a full sort per
    # candidate weight vector, and the ranking is stable well below full size
    frac = float(os.environ.get("BER_TUNE_FRAC", "1.0"))
    if frac < 1.0:
        rng0 = np.random.default_rng(5)
        keepq = rng0.random(nq) < frac
        m = keepq[qi]
        F, y = F[m], y[m]
        qi = np.searchsorted(np.flatnonzero(keepq), qi[m])
        nq = int(keepq.sum())
        ntrue = int(round(ntrue * frac))
    print(f"pairs={len(y):,} nq={nq:,} ntrue={ntrue:,} "
          f"netrecall={y.sum()/ntrue:.4f} cand/q={len(y)/nq:.1f}")

    # derive the components that are combinations of stored ones, so the
    # captured wide net can be reused without recomputing it
    def col(name):
        if name in IX:
            return F[:, IX[name]]
        if name == "sc_rel_w":
            sa = F[:, IX["sc_all"]]
            mx = np.zeros(nq, np.float32)
            o = np.argsort(qi, kind="stable")
            q2, v2 = qi[o], sa[o]
            br = np.flatnonzero(np.r_[True, q2[1:] != q2[:-1]])
            mx[q2[br]] = np.maximum.reduceat(v2, br)
            return sa / np.maximum(mx[qi], 1e-6)
        if name == "blank_d":
            return F[:, IX["d_addr_blank"]]
        if name == "mx_name":
            return np.maximum(col("nm_ratio"), col("cj_jw"))
        if name == "mn_na":
            return np.minimum(col("nm_ratio"), col("ad_tsr"))
        if name == "prod_na":
            return col("nm_ratio") * col("ad_tsr")
        if name == "cd_max":
            return np.stack([col(f"cd{g}") for g in range(4)], 1).max(1)
        if name == "cd_sum":
            return sum(col(f"cd{g}") for g in range(4))
        if name == "gm_tot":
            return sum(col(f"gm{g}") for g in range(4))
        if name == "nm_x_cd3":
            return col("nm_ratio") * col("cd3")
        if name == "sc_x_nm":
            return col("sc_all") * col("nm_ratio")
        raise KeyError(name)

    X = np.stack([col(c) for c in COMP], 1)
    # scale each component to a comparable range
    hi = np.quantile(X[::7], 0.99, axis=0)
    hi[hi <= 0] = 1.0
    X = X / hi

    base = {"nm_ratio": 1.36, "nm_partial": 0.73, "cj_jw": 0.29,
            "ad_tsr": 0.23, "sc_all": 0.70, "cd1": 1.06, "cd2": 0.94,
            "cq1": 0.26, "gm0": 0.10, "gm3": 0.10}
    w0 = np.array([base.get(c, 0.0) for c in COMP], np.float32)
    print("\nbaseline:", ["%.4f" % r for r in recall_at(X @ w0, qi, y, nq, ntrue=ntrue)])

    rng = np.random.default_rng(3)
    best = (recall_at(X @ w0, qi, y, nq, (TK,), ntrue)[0], w0)
    # start from a broad prior that trusts both fields plus retrieval evidence
    seeds = [w0]
    pr = {"nm_ratio": .4, "cj_jw": .2, "ad_tsr": .25, "sc_rel_w": .4,
          "sc_all": .4, "gm0": .3, "gm1": .3, "gm2": .2, "gm3": .4,
          "cd0": .2, "cd1": .2, "cd2": .2, "cd3": .2}
    seeds.append(np.array([pr.get(c, 0.0) for c in COMP], np.float32))
    for s in seeds:
        r = recall_at(X @ s, qi, y, nq, (TK,), ntrue)[0]
        if r > best[0]:
            best = (r, s)
    cur = best[1].copy()
    T = int(os.environ.get("BER_ITERS", "900"))
    for it in range(T):
        step = 0.35 * (1 - it / T) + 0.05
        w = cur + rng.normal(0, step, len(COMP)).astype(np.float32) * (
            rng.random(len(COMP)) < 0.35)
        w = np.maximum(w, 0)
        if w.sum() <= 0:
            continue
        r = recall_at(X @ w, qi, y, nq, (TK,), ntrue)[0]
        if r > best[0]:
            best = (r, w.copy())
            cur = w
        elif r > best[0] - 0.002 and rng.random() < 0.3:
            cur = w
        if (it + 1) % 100 == 0:
            print(f"  it{it+1} best r@{TK}={best[0]:.4f}", flush=True)
    w = best[1]
    print("\ntuned r@K:", ["%.4f" % r for r in recall_at(X @ w, qi, y, nq, ntrue=ntrue)])
    print("weights (on /q99-scaled components):")
    for c, v, h in sorted(zip(COMP, w, hi), key=lambda z: -z[1]):
        if v > 1e-3:
            print(f"   {c:14s} {v:7.4f}   (scale {h:.4g})")
    np.savez(f"{WORK}/prune_w.npz", comp=np.array(COMP), w=w, hi=hi)
    print("PRUNE_W = {")
    for c, v, h in sorted(zip(COMP, w, hi), key=lambda z: -z[1]):
        if v > 1e-3:
            print(f'    "{c}": {v/h:.5g},')
    print("}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
