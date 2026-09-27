"""Learned set-size selection.

The per-pair probabilities are well calibrated and rank almost perfectly, yet
picking the set size from them with an expected-F_0.5 rule leaves a lot on the
table.  The reason is that match indicators inside one entity are strongly
*correlated* -- the generator either produced variants for a business or it did
not -- so any rule that treats candidates as independent misjudges how many to
take, and badly misjudges the "no match at all" case.

So instead of assuming a joint distribution we learn the value directly: for
each entity and each cut-off k, a regressor predicts the F_0.5 that taking the
top k candidates would actually score.  At inference we evaluate every k and
take the arg max.  This optimises the competition metric end to end and needs no
independence assumption.
"""
import numpy as np

MAXK = 12          # cut-offs 0..MAXK are scored; beyond this F_0.5 is hopeless
NTOP = 10          # how many candidate probabilities enter the feature vector

FEATS = ([f"p{i}" for i in range(NTOP)]
         + [f"gap{i}" for i in range(NTOP - 1)]
         + ["psum", "pmax", "n_cand", "n_gt9", "n_gt7", "n_gt5", "n_gt3",
            "n_gt1", "ent_prod0", "logit_mean", "p_last", "src2_frac",
            "psum2", "psum_tail", "k", "k_minus_psum", "p_at_k",
            "p_after_k", "cum_k", "cum_frac"])


def entity_matrix(qi, p, nq, src=None):
    """Per-entity summary block plus the sorted probability list.

    Returns (E, P, bounds) where E is (nq, n_static), P is the padded sorted
    probability matrix (nq, MAXK+1) and bounds gives each entity's candidate
    count.
    """
    o = np.lexsort((-p, qi))
    q, pp = qi[o], p[o]
    qs = np.searchsorted(q, np.arange(nq + 1))
    cnt = np.diff(qs)
    starts = np.repeat(qs[:-1], cnt)
    r = np.arange(len(q)) - starts

    P = np.zeros((nq, max(NTOP, MAXK + 1)), np.float32)
    keep = r < P.shape[1]
    P[q[keep], r[keep]] = pp[keep]

    cs = np.cumsum(pp.astype(np.float64))
    base = np.where(qs[:-1] > 0, cs[np.maximum(qs[:-1] - 1, 0)], 0.0)
    psum = np.zeros(nq, np.float32)
    nz = cnt > 0
    psum[nz] = (cs[qs[1:][nz] - 1] - base[nz]).astype(np.float32)
    l1p = np.cumsum(np.log1p(-np.clip(pp, 0, 1 - 1e-6)).astype(np.float64))
    b2 = np.where(qs[:-1] > 0, l1p[np.maximum(qs[:-1] - 1, 0)], 0.0)
    prod0 = np.zeros(nq, np.float32)
    prod0[nz] = np.exp(l1p[qs[1:][nz] - 1] - b2[nz]).astype(np.float32)

    def cntgt(th):
        c = np.zeros(nq, np.float32)
        m = pp > th
        np.add.at(c, q[m], 1.0)
        return c

    lg = np.log(np.clip(pp, 1e-6, 1 - 1e-6) / (1 - np.clip(pp, 1e-6, 1 - 1e-6)))
    lgm = np.zeros(nq, np.float32)
    np.add.at(lgm, q, lg.astype(np.float32))
    ncand = cnt.astype(np.float32)
    lgm = lgm / np.maximum(ncand, 1)
    plast = np.zeros(nq, np.float32)
    plast[q[keep]] = pp[keep]           # overwritten in order -> last kept
    s2f = np.zeros(nq, np.float32)
    if src is not None:
        np.add.at(s2f, qi, (src == 2).astype(np.float32))
        s2f = s2f / np.maximum(ncand, 1)

    gaps = P[:, :NTOP - 1] - P[:, 1:NTOP]
    psum2 = P[:, :2].sum(1)
    tail = np.maximum(psum - P[:, :NTOP].sum(1), 0.0)
    E = np.column_stack([
        P[:, :NTOP], gaps, psum, P[:, 0], ncand,
        cntgt(0.9), cntgt(0.7), cntgt(0.5), cntgt(0.3), cntgt(0.1),
        prod0, lgm, plast, s2f, psum2, tail]).astype(np.float32)
    return E, P, cnt


def build_rows(E, P, cnt, ks=None):
    """Expand the per-entity block into one row per (entity, k)."""
    nq = E.shape[0]
    ks = np.arange(MAXK + 1) if ks is None else np.asarray(ks)
    nk = len(ks)
    rows = np.empty((nq * nk, E.shape[1] + 6), np.float32)
    cum = np.cumsum(P[:, :MAXK + 1], 1)
    for j, k in enumerate(ks):
        sl = slice(j * nq, (j + 1) * nq)
        rows[sl, :E.shape[1]] = E
        c = E.shape[1]
        rows[sl, c] = k
        ck = cum[:, k - 1] if k > 0 else np.zeros(nq, np.float32)
        rows[sl, c + 1] = k - ck
        rows[sl, c + 2] = P[:, k - 1] if k > 0 else 0.0
        rows[sl, c + 3] = P[:, k] if k < P.shape[1] else 0.0
        rows[sl, c + 4] = ck
        rows[sl, c + 5] = ck / np.maximum(E[:, NTOP * 2 - 1 + 0], 1e-6)
    return rows, ks


def targets(qi, p, y, nt, nq, ks=None):
    """Realised F_0.5 of taking the top k, for every entity and every k."""
    o = np.lexsort((-p, qi))
    q, yy = qi[o], y[o]
    qs = np.searchsorted(q, np.arange(nq + 1))
    cnt = np.diff(qs)
    starts = np.repeat(qs[:-1], cnt)
    r = np.arange(len(q)) - starts
    cy = np.cumsum(yy.astype(np.float64))
    base = np.where(qs[:-1] > 0, cy[np.maximum(qs[:-1] - 1, 0)], 0.0)
    TP = np.zeros((nq, MAXK + 1), np.float32)
    kk = r < MAXK
    np.maximum.at(TP, (q[kk], r[kk] + 1), 0.0)        # allocate
    tpflat = (cy - np.repeat(base, cnt)).astype(np.float32)
    TP[q[kk], r[kk] + 1] = tpflat[kk]
    # carry forward so TP[:,k] is the prefix count even past the last candidate
    for k in range(1, MAXK + 1):
        TP[:, k] = np.maximum(TP[:, k], TP[:, k - 1])
    ks = np.arange(MAXK + 1) if ks is None else np.asarray(ks)
    Y = np.empty((nq, len(ks)), np.float32)
    ntf = nt.astype(np.float32)
    for j, k in enumerate(ks):
        kk2 = np.minimum(k, cnt)                       # cannot take more
        with np.errstate(invalid="ignore", divide="ignore"):
            f = 1.25 * TP[:, k] / np.maximum(kk2 + 0.25 * ntf, 1e-9)
        f = np.where(ntf == 0, np.where(kk2 == 0, 1.0, 0.0), f)
        Y[:, j] = f
    return Y, cnt
