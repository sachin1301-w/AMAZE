"""Scoring metric and the decision layer that turns pair probabilities into
one predicted match set per Source-1 entity.

Two facts drive the design.

*  The metric simplifies.  With P = TP/|S| and R = TP/|T|,

       F_0.5 = 1.25 * TP / (|S| + 0.25 * |T|)

   so the best set is the one that buys true positives most cheaply: each extra
   prediction costs 1 in the denominator and returns p_i in the numerator.

*  In the training labels every Source-2/3 record matches *at most one*
   Source-1 entity (verified: 7.6M matched ids, none reused).  Candidate lists
   of different entities may therefore be pruned against each other -- a record
   is kept only for the entity that scores it highest.
"""
import numpy as np


def fbeta_half(pred, truth):
    """Macro F_0.5 over entities.  ``pred``/``truth`` map entity -> set of ids."""
    tot = 0.0
    for e, T in truth.items():
        S = pred.get(e, ())
        if not T:
            tot += 0.0 if S else 1.0
            continue
        if not S:
            continue
        tp = len(T & set(S))
        tot += 1.25 * tp / (len(S) + 0.25 * len(T))
    return tot / len(truth)


def score_flat(qi, sel, y, nq, ntrue_per_q):
    """Macro F_0.5 from flat pair arrays.

    ``ntrue_per_q`` is the true match count per entity taken from the ground
    truth, not from the candidates, so matches blocking never proposed are
    correctly counted as misses.
    """
    ns = np.bincount(qi[sel], minlength=nq).astype(np.float64)
    tp = np.bincount(qi[sel & (y > 0)], minlength=nq).astype(np.float64)
    nt = ntrue_per_q.astype(np.float64)
    f = np.where((nt == 0),
                 np.where(ns == 0, 1.0, 0.0),
                 1.25 * tp / np.maximum(ns + 0.25 * nt, 1e-9))
    return float(f.mean()), f


def group_bounds(qi, nq):
    return np.searchsorted(qi, np.arange(nq + 1))


def resolve_exclusive(di, p, ndoc, qi=None, margin=0.0):
    """Keep each Source-2/3 record only for the entity that scores it highest.

    Returns a boolean mask over pairs.
    """
    best = np.zeros(ndoc, np.float32)
    o = np.argsort(di, kind="stable")
    d2, p2 = di[o], p[o]
    br = np.flatnonzero(np.r_[True, d2[1:] != d2[:-1]])
    best[d2[br]] = np.maximum.reduceat(p2, br)
    return p >= best[di] - margin


def select_sets(qi, p, nq, m0=0.12, pmin=0.0, k0_boost=1.0, pw=1.0):
    """Pick, per entity, the prediction set maximising expected F_0.5.

    Candidates are taken in decreasing probability; for k >= 1 the expected
    score uses the mean-field substitution

        E[F] ~ 1.25 * sum_{i<=k} p_i / (k + 0.25 * (P_total + m0))

    and k = 0 is scored exactly as the probability that the entity really has
    no match, prod(1 - p_i).  ``m0`` accounts for true matches that blocking
    never proposed.
    """
    if len(qi) == 0:
        return np.zeros(0, bool)
    pe = np.clip(p, 1e-6, 1 - 1e-6) ** pw
    o = np.lexsort((-pe, qi))
    q, pp = qi[o], pe[o]
    qs = group_bounds(q, nq)
    cnt = np.diff(qs)
    starts = np.repeat(qs[:-1], cnt)
    r = np.arange(len(q)) - starts                      # 0-based rank
    cs = np.cumsum(pp)
    base = np.where(starts > 0, cs[np.maximum(starts - 1, 0)], 0.0)
    csum = cs - base                                    # sum of top-(r+1)
    ptot = np.repeat(np.where(cnt > 0, csum[qs[1:] - 1], 0.0), cnt)
    sc = 1.25 * csum / ((r + 1) + 0.25 * (ptot + m0))
    sc = np.where(pp >= pmin, sc, -1.0)
    # best k >= 1 per entity
    gm = np.full(nq, -1.0)
    nz = cnt > 0
    gm[nz] = np.maximum.reduceat(sc, qs[:-1][nz])
    # k = 0 : probability the entity is genuinely a singleton
    lg = np.cumsum(np.log1p(-pp))
    lgtot = np.where(nz, lg[qs[1:] - 1] - np.where(qs[:-1] > 0,
                     lg[np.maximum(qs[:-1] - 1, 0)], 0.0), 0.0)
    sc0 = np.exp(lgtot) * k0_boost
    take_any = gm > sc0
    keep = (sc >= np.repeat(gm, cnt) - 1e-12) & np.repeat(take_any, cnt)
    # a pair is kept when its rank is within the chosen k
    kchosen = np.zeros(nq, np.int64)
    kk = np.where(keep, r + 1, 0)
    gk = np.zeros(nq, np.int64)
    gk[nz] = np.maximum.reduceat(kk, qs[:-1][nz])
    kchosen = gk
    sel_sorted = (r < np.repeat(kchosen, cnt)) & (pp >= pmin)
    out = np.zeros(len(qi), bool)
    out[o] = sel_sorted
    return out
