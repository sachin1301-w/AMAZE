"""Competition features derived from the stage-1 match probability.

These are what let the matcher use the partition constraint: a Source-2/3
record belongs to at most one Source-1 entity, so a candidate that some *other*
entity claims much more strongly is probably a false merge.

Statistics are computed inside a partition (source x country), exactly as they
are at inference time, where each partition is scored on its own.
"""
import numpy as np

STACK_NAMES = ["p1", "p1_rel", "p1_rank", "p1_sum", "p1_max", "p1_gap",
               "dp1_best", "dp1_gap", "dp1_sum", "dp1_isbest", "dp1_rank",
               "p1_sum_above"]


def _groupmax(idx, val, n):
    out = np.zeros(n, np.float32)
    if len(idx) == 0:
        return out
    o = np.argsort(idx, kind="stable")
    i2, v2 = idx[o], val[o]
    br = np.flatnonzero(np.r_[True, i2[1:] != i2[:-1]])
    out[i2[br]] = np.maximum.reduceat(v2, br)
    return out


def _rank_within(key, val, n):
    """Rank by descending val inside each key group (0 = strongest)."""
    if len(key) == 0:
        return np.zeros(0, np.float32)
    o = np.lexsort((-val, key))
    ks = np.searchsorted(key[o], np.arange(n + 1))
    r = np.empty(len(key), np.float32)
    r[o] = np.arange(len(key), dtype=np.float32) - np.repeat(ks[:-1],
                                                            np.diff(ks))
    return r


def stack_features(qi, d, src, part, p1, nq):
    """Build the stage-2 feature block.

    ``part`` identifies the (source, country) partition so that per-entity
    statistics stay within the partition that produced them, matching how the
    features are computed one partition at a time during inference.
    """
    n = len(qi)
    S = np.zeros((n, len(STACK_NAMES)), np.float32)
    # entity key: entity within its partition
    npart = int(part.max()) + 1 if n else 1
    ekey = qi.astype(np.int64) * npart + part
    uek, ei = np.unique(ekey, return_inverse=True)
    ne = len(uek)
    dkey = d.astype(np.int64) * 4 + src
    udk, dind = np.unique(dkey, return_inverse=True)
    nd = len(udk)

    esum = np.bincount(ei, weights=p1.astype(np.float64),
                       minlength=ne).astype(np.float32)
    emax = _groupmax(ei, p1, ne)
    erank = _rank_within(ei, p1, ne)
    dsum = np.bincount(dind, weights=p1.astype(np.float64),
                       minlength=nd).astype(np.float32)
    dmax = _groupmax(dind, p1, nd)
    drank = _rank_within(dind, p1, nd)
    eps = np.float32(1e-6)

    # cumulative probability of the candidates ranked above this one
    o = np.lexsort((-p1, ei))
    cs = np.cumsum(p1[o].astype(np.float64))
    ks = np.searchsorted(ei[o], np.arange(ne + 1))
    base = np.where(ks[:-1] > 0, cs[np.maximum(ks[:-1] - 1, 0)], 0.0)
    above = np.empty(n, np.float32)
    above[o] = (cs - np.repeat(base, np.diff(ks)) - p1[o]).astype(np.float32)

    col = dict(p1=p1, p1_rel=p1 / np.maximum(emax[ei], eps), p1_rank=erank,
               p1_sum=esum[ei], p1_max=emax[ei], p1_gap=emax[ei] - p1,
               dp1_best=dmax[dind], dp1_gap=dmax[dind] - p1,
               dp1_sum=dsum[dind],
               dp1_isbest=(dmax[dind] <= p1 + 1e-6).astype(np.float32),
               dp1_rank=drank, p1_sum_above=above)
    for i, k in enumerate(STACK_NAMES):
        S[:, i] = col[k]
    return S
