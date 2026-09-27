"""Pairwise features.

The work is split by cost.

``retr_components``  cheap, fully vectorised evidence available for the whole
                    wide retrieval net: per-group shared IDF mass and its
                    containment in each side, plus four rapidfuzz similarities.
``prune_score``      a deterministic weighted sum of those components.  This is
                    the filter that produces candidate_pairs.tsv; no learned
                    model is involved.
``pair_features``    the full matrix for the LightGBM matcher, computed only for
                    surviving candidates, so the per-pair python work (token set
                    intersections) runs on ~16 pairs per entity, not ~250.
"""
import os
import sys

import numpy as np
from rapidfuzz import fuzz, process
from rapidfuzz.distance import JaroWinkler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from normalize import LEGAL  # noqa: E402
from terms import DF_MASK, NGRP, addr_parts, core_name  # noqa: E402

from hw import THREADS, breathe  # noqa: E402
WORKERS = int(os.environ.get("BER_WORKERS", str(THREADS)))
_LEGAL_ID = {v: i for i, v in enumerate(sorted(set(LEGAL.values())))}
assert len(_LEGAL_ID) <= 63, "legal-form bitmask needs <= 63 slots"
EPS = np.float32(1e-6)


def rarity_weight(tab):
    """uint8 log2(df) buckets -> a float32 rarity weight per hash slot."""
    lut = np.maximum(0.0, 23.0 - np.arange(256, dtype=np.float32))
    lut[0] = 23.0                       # unseen token: treat as very rare
    return lut[tab]


def _cp(a, b, scorer):
    return process.cpdist(a, b, scorer=scorer, workers=WORKERS,
                          dtype=np.float32)


class Side:
    """Per-record derived views for one side of the comparison.

    Only string views and scalar aggregates are cached; token sets are rebuilt
    per surviving pair, because caching sets for 5M records costs several GB.
    """

    def __init__(self, nname, naddr, dom, rw_name, rw_addr):
        n = len(nname)
        self.n = n
        self.nm = np.asarray(nname, dtype=object)
        self.ad = np.asarray(naddr, dtype=object)
        self.domf = np.asarray([1.0 if d else 0.0 for d in dom], np.float32)
        cn = np.empty(n, object); cj = np.empty(n, object)
        aw = np.empty(n, object); nu = np.empty(n, object)
        nw = np.zeros(n, np.float32); aww = np.zeros(n, np.float32)
        nlen = np.zeros(n, np.float32); alen = np.zeros(n, np.float32)
        dlen = np.zeros(n, np.float32)
        lgl = np.zeros(n, np.int64)
        for i in range(n):
            ct = core_name(nname[i])
            w, nums = addr_parts(naddr[i])
            if dom[i] and dom[i] not in ct:
                ct = ct + [dom[i]]
            sn, sa, sd = set(ct), set(w), set(nums)
            cn[i] = " ".join(sorted(sn))
            cj[i] = "".join(sorted(sn))
            aw[i] = " ".join(sorted(sa))
            nu[i] = " ".join(sorted(sd))
            nw[i] = sum(rw_name[hash(t) & DF_MASK] for t in sn)
            aww[i] = sum(rw_addr[hash(t) & DF_MASK] for t in sa)
            nlen[i] = len(sn); alen[i] = len(sa); dlen[i] = len(sd)
            m = 0
            for t in nname[i].split():
                j = _LEGAL_ID.get(LEGAL.get(t, ""), -1)
                if j >= 0:
                    m |= 1 << j
            lgl[i] = m
        self.cn, self.cj, self.aw, self.nu = cn, cj, aw, nu
        self.nw, self.aww = nw, aww
        self.nlen, self.alen, self.dlen = nlen, alen, dlen
        self.lgl = lgl
        self.rw_name, self.rw_addr = rw_name, rw_addr


# ------------------------------------------------------ stage B components ---
RET_BASE = ["sc_all", "sc_name", "sc_addr", "rk_all", "rk_name", "rk_addr",
            "nchan"]
GRP_NAMES = ([f"gm{g}" for g in range(NGRP)] + [f"cd{g}" for g in range(NGRP)]
             + [f"cq{g}" for g in range(NGRP)])
FUZZ4 = ["nm_ratio", "nm_partial", "cj_jw", "ad_tsr"]
# dense-encoder similarity.  A name transliterated into Devanagari or Kannada
# shares no character n-gram with its Latin form, and a scrambled token shares
# almost none, so this expresses agreement the lexical features cannot reach.
EMB_NAMES = ["emb_cos", "emb_rel", "emb_rank"]
# reverse-direction retrieval: how strongly this Source-2/3 record picks this
# entity as its parent when the record itself is the query.  rev_rank == 0 means
# the record's single best parent is this entity, which the partition constraint
# makes powerful evidence.
REV_NAMES = ["rev_sc", "rev_rank", "rev_hit"]
# computed over the wide net (before pruning), unlike the COMP_NAMES versions
WIDE_NAMES = ["sc_rel_w"]
# cheap non-linear combinations, so the (deliberately linear) prune score can
# express "agrees on at least one field" and "agrees on both"
DERIV = ["mx_name", "mn_na", "prod_na", "cd_max", "cd_sum", "gm_tot",
         "nm_x_cd3", "sc_x_nm", "blank_d"]


def emb_cosine(QE, DE, qi, di, chunk=4_000_000):
    """Cosine between the two records' encoder vectors (already L2-normalised)."""
    out = np.empty(len(qi), np.float32)
    if QE is None or DE is None:
        out[:] = 0.0
        return out
    for a in range(0, len(qi), chunk):
        b = min(len(qi), a + chunk)
        u = QE[qi[a:b]].astype(np.float32)
        v = DE[di[a:b]].astype(np.float32)
        out[a:b] = np.einsum("ij,ij->i", u, v)
    return out


def retr_components(Q, D, qi, di, ch, gm, qmass, dmass, n_doc,
                    qloc=None, rev=None, QE=None, DE=None, qrow=None, nq=0):
    """Vectorised evidence for every pair of the wide net.

    ``ch`` is the (npairs, 7) per-channel score/rank block, ``gm`` the
    (npairs, NGRP) shared evidence mass, ``qmass``/``dmass`` the per-record
    group masses used for containment.  ``qi`` indexes ``Q`` (whole split) and
    ``qloc`` indexes ``qmass`` (current batch); they differ after batch 0.

    The raw mass is divided by log(1 + n_doc)**2 so it reads as an effective
    count of shared rare terms.  Without that the feature would scale with the
    partition's corpus size and the model -- trained on US and India only --
    would not transfer to the smaller France partitions.
    """
    C = {k: ch[:, i] for i, k in enumerate(RET_BASE)}
    if qloc is None:
        qloc = qi
    unit = np.float32(max(1.0, np.log1p(n_doc) ** 2))
    for g in range(NGRP):
        C[f"gm{g}"] = gm[:, g] / unit
        C[f"cd{g}"] = gm[:, g] / np.maximum(dmass[g][di], EPS)
        C[f"cq{g}"] = gm[:, g] / np.maximum(qmass[qloc, g], EPS)
    if rev is None:
        rev = np.zeros((len(di), 3), np.float32)
        rev[:, 1] = 99.0
    for i, k in enumerate(REV_NAMES):
        C[k] = rev[:, i]
    C["nm_ratio"] = _cp(Q.nm[qi], D.nm[di], fuzz.ratio) / 100.0
    C["nm_partial"] = _cp(Q.cn[qi], D.cn[di], fuzz.partial_ratio) / 100.0
    C["cj_jw"] = _cp(Q.cj[qi], D.cj[di], JaroWinkler.normalized_similarity)
    a = _cp(Q.aw[qi], D.aw[di], fuzz.token_sort_ratio) / 100.0
    blank = (D.alen[di] + D.dlen[di]) == 0
    C["ad_tsr"] = np.where(blank, np.float32(0.45), a)
    C["blank_d"] = blank.astype(np.float32)
    ec = emb_cosine(QE, DE, qi, di)
    C["emb_cos"] = ec
    if qrow is not None and nq:
        mx = np.zeros(nq, np.float32)
        o = np.argsort(qrow, kind="stable")
        q2, v2 = qrow[o], ec[o]
        br = np.flatnonzero(np.r_[True, q2[1:] != q2[:-1]])
        mx[q2[br]] = np.maximum.reduceat(v2, br)
        C["emb_rel"] = ec / np.maximum(mx[qrow], EPS)
        rk = np.empty(len(ec), np.float32)
        o2 = np.lexsort((-ec, qrow))
        st = np.searchsorted(qrow[o2], np.arange(nq + 1))
        rk[o2] = (np.arange(len(ec), dtype=np.float32)
                  - np.repeat(st[:-1], np.diff(st)))
        C["emb_rank"] = rk
    else:
        C["emb_rel"] = np.ones_like(ec)
        C["emb_rank"] = np.zeros_like(ec)
    add_derived(C)
    return C


def add_derived(C, D=None, di=None):
    """Fill in the DERIV block from the base components.

    These are pure functions of components that are already present, so they are
    recomputed rather than carried through the pipeline -- at 30 candidates per
    entity per source, every carried array costs hundreds of MB.
    """
    if "blank_d" not in C:
        C["blank_d"] = ((D.alen[di] + D.dlen[di]) == 0).astype(np.float32)
    C["mx_name"] = np.maximum(C["nm_ratio"], C["cj_jw"])
    C["mn_na"] = np.minimum(C["nm_ratio"], C["ad_tsr"])
    C["prod_na"] = C["nm_ratio"] * C["ad_tsr"]
    cds = np.stack([C[f"cd{g}"] for g in range(NGRP)], 1)
    C["cd_max"] = cds.max(1)
    C["cd_sum"] = cds.sum(1)
    C["gm_tot"] = sum(C[f"gm{g}"] for g in range(NGRP))
    C["nm_x_cd3"] = C["nm_ratio"] * C["cd3"]
    C["sc_x_nm"] = C["sc_all"] * C["nm_ratio"]
    return C


# Weights for the deterministic prune, found by random search on the training
# split to maximise recall at the candidate budget (see tune_prune.py).  This is
# a fixed formula, not a fitted model: it is evaluated in numpy over the wide
# retrieval net and its output is what candidate_pairs.tsv contains.
PRUNE_W = {
    "prod_na": 3.7129,
    "cq3": 3.2438,
    "nm_x_cd3": 2.902,
    "gm2": 2.4789,
    "sc_addr": 2.2354,
    "sc_all": 1.9175,
    "nm_partial": 1.7494,
    "blank_d": 1.1072,
    "sc_x_nm": 1.0537,
    "sc_name": 1.011,
    "cd2": 0.95869,
    "nchan": 0.70209,
    "sc_rel_w": 0.56312,
    "cq2": 0.5153,
    "cd1": 0.51355,
    "gm1": 0.40581,
    "mx_name": 0.40091,
    "cq1": 0.30647,
    "ad_tsr": 0.30078,
    "cj_jw": 0.18247,
    "gm3": 0.058868,
    "cd_sum": 0.0030581,
}


# Components a prune model may legitimately see: everything known before the
# matcher runs.  Competition features that depend on the prune's own output are
# excluded, as are the cohesion features (they need the surviving set).
SHAPE_FEATS = ["q_nlen", "d_nlen", "q_alen", "d_alen", "q_dlen", "d_dlen",
               "src_is3"]
# Everything a stage-1 prune can see: available straight from the retrieval
# pass, fully vectorised, no per-pair python work.
CHEAP_FEATS = (RET_BASE + GRP_NAMES + FUZZ4 + WIDE_NAMES + REV_NAMES + DERIV
               + EMB_NAMES + SHAPE_FEATS)


def cheap_matrix(C, Q, D, qi, di, src_is3):
    n = len(qi)
    X = np.empty((n, len(CHEAP_FEATS)), np.float32)
    shp = dict(q_nlen=Q.nlen[qi], d_nlen=D.nlen[di], q_alen=Q.alen[qi],
               d_alen=D.alen[di], q_dlen=Q.dlen[qi], d_dlen=D.dlen[di],
               src_is3=np.full(n, np.float32(1.0 if src_is3 else 0.0)))
    for i, k in enumerate(CHEAP_FEATS):
        X[:, i] = shp[k] if k in shp else C[k]
    return X


_NOT_PRUNE = ("prune", "prune_rel", "prune_rank", "n_cand", "prune_gap1",
              "prune_max", "doc_best", "doc_gap", "doc_ncand", "doc_is_best",
              "sc_rel", "coh_nm", "coh_ad", "coh_is_best", "coh_extra_n",
              "coh_extra_a", "coh_extra_aw")
PRUNE_FEATS = []          # filled in below, once FEAT_NAMES exists

_PRUNE_CHEAP = None
_PRUNE_FULL = None


def load_prune_models(cheap_path=None, full_path=None):
    """Attach the learned prune rankers (see train_prune.py).

    The hand-tuned linear score kept ~99.2% of the retrieval net's true pairs
    at 30 candidates per source.  A cheap ranker over the retrieval components
    alone reaches 99.4%, and re-ranking its top 120 with the full pair features
    reaches 99.85% at only 15 -- half the candidates for better recall.
    """
    global _PRUNE_CHEAP, _PRUNE_FULL
    import lightgbm as lgb
    if cheap_path and os.path.exists(cheap_path):
        _PRUNE_CHEAP = lgb.Booster(model_file=cheap_path)
    if full_path and os.path.exists(full_path):
        _PRUNE_FULL = lgb.Booster(model_file=full_path)
    return _PRUNE_CHEAP, _PRUNE_FULL


def have_prune_models():
    """The cheap ranker alone is the operating point.

    Measured retention of the retrieval net's true pairs, held-out entities:

        candidates/source      15      20      30
        hand-tuned linear   0.9823  0.9847  0.9918
        cheap ranker        0.9920  0.9928  0.9942
        + full re-rank      0.9897  0.9918  0.9943

    Re-ranking the shortlist with the full pair features does not beat the
    cheap ranker, and it costs per-pair python work on a 120-deep shortlist, so
    the second stage is off unless BER_TWOSTAGE=1.
    """
    return _PRUNE_CHEAP is not None


def two_stage():
    return _PRUNE_FULL is not None and os.environ.get("BER_TWOSTAGE") == "1"


def cheap_prune_score(C, Q, D, qi, di, src_is3):
    if _PRUNE_CHEAP is None:
        return prune_score(C)
    return _PRUNE_CHEAP.predict(cheap_matrix(C, Q, D, qi, di, src_is3),
                                num_threads=THREADS).astype(np.float32)


def full_prune_score(Fbase):
    """Score the 73 pre-competition columns of an already-built matrix."""
    idx = [FEAT_NAMES.index(c) for c in PRUNE_FEATS]
    return _PRUNE_FULL.predict(Fbase[:, idx], num_threads=THREADS).astype(np.float32)


def _unused_load(path):
    """Attach a learned prune ranker (see train_prune.py).

    The hand-tuned linear score kept only ~97% of the retrieval net's true
    pairs at 30 candidates per source; a gradient-boosted ranker over the same
    components keeps ~100% at 20, which is worth more than any downstream
    change we measured.
    """
    raise NotImplementedError


def prune_score(C, w=None, D=None, di=None):
    """The original hand-tuned linear score; still the fallback when no learned
    prune model is present."""
    w = PRUNE_W if w is None else w
    s = np.zeros(len(C["sc_all"]), np.float32)
    for k, v in w.items():
        s += np.float32(v) * C[k]
    return s


# ---------------------------------------------------------- stage C matrix ---
SET_NAMES = ["nsh", "nsh_w", "nq_only_w", "nd_only_w", "n_jac", "n_cont_d",
             "n_cont_q", "ash", "ash_w", "aq_only_w", "ad_only_w", "a_jac",
             "a_cont_d", "a_cont_q", "dsh", "dsh_w", "d_jac", "d_cont_d",
             "first_num_eq"]
COMP_NAMES = ["sc_rel", "prune", "prune_rel", "prune_rank", "n_cand",
              "prune_gap1", "prune_max", "wide_n", "doc_best", "doc_gap",
              "doc_ncand", "doc_is_best"]
# cohesion: how much this candidate looks like the entity's strongest
# candidate, and whether it shares evidence with it that the Source-1 record
# itself lacks -- transitive evidence for records whose own name was replaced
COH_NAMES = ["coh_nm", "coh_ad", "coh_is_best", "coh_extra_n",
             "coh_extra_a", "coh_extra_aw"]
FEAT_NAMES = (
    ["nm_ratio", "nm_tsr", "nm_tset", "nm_partial", "cj_jw", "cj_ratio",
     "nm_full_tset", "ad_ratio", "ad_tsr", "ad_tset", "num_tsr"]
    + SET_NAMES
    + ["q_nlen", "d_nlen", "q_alen", "d_alen", "q_dlen", "d_dlen",
       "d_addr_blank", "legal_jac", "dom_flag"]
    + RET_BASE + GRP_NAMES + WIDE_NAMES + DERIV + REV_NAMES + EMB_NAMES
    + COMP_NAMES + COH_NAMES + ["src_is3"]
)
NF = len(FEAT_NAMES)
PRUNE_FEATS[:] = [c for c in FEAT_NAMES if c not in _NOT_PRUNE]


def pair_features(Q, D, qi, di, C, comp, src_is3, di_all=None):
    """Full feature matrix.  ``comp=None`` leaves the competition and cohesion
    columns zero, which is what the stage-2 prune scores (it must not see
    statistics derived from its own output)."""
    n = len(qi)
    F = np.empty((n, NF), np.float32)
    c = 0

    def put(v):
        nonlocal c
        F[:, c] = v
        c += 1

    qnm, dnm = Q.nm[qi], D.nm[di]
    qcn, dcn = Q.cn[qi], D.cn[di]
    put(C["nm_ratio"])
    put(_cp(qcn, dcn, fuzz.token_sort_ratio) / 100.0)
    put(_cp(qcn, dcn, fuzz.token_set_ratio) / 100.0)
    put(C["nm_partial"])
    put(C["cj_jw"])
    put(_cp(Q.cj[qi], D.cj[di], fuzz.ratio) / 100.0)
    put(_cp(qnm, dnm, fuzz.token_set_ratio) / 100.0)
    put(_cp(Q.ad[qi], D.ad[di], fuzz.ratio) / 100.0)
    put(C["ad_tsr"])
    put(_cp(Q.aw[qi], D.aw[di], fuzz.token_set_ratio) / 100.0)
    put(_cp(Q.nu[qi], D.nu[di], fuzz.token_sort_ratio) / 100.0)

    rn, ra = Q.rw_name, Q.rw_addr
    Qcn, Dcn, Qaw, Daw, Qnu, Dnu = Q.cn, D.cn, Q.aw, D.aw, Q.nu, D.nu
    nsh = np.empty(n, np.float32); nshw = np.empty(n, np.float32)
    ash = np.empty(n, np.float32); ashw = np.empty(n, np.float32)
    dsh = np.empty(n, np.float32); dshw = np.empty(n, np.float32)
    fne = np.zeros(n, np.float32)
    M = DF_MASK
    for k in range(n):
        a, b = qi[k], di[k]
        s = set(Qcn[a].split()) & set(Dcn[b].split())
        nsh[k] = len(s)
        nshw[k] = sum(rn[hash(t) & M] for t in s) if s else 0.0
        s = set(Qaw[a].split()) & set(Daw[b].split())
        ash[k] = len(s)
        ashw[k] = sum(ra[hash(t) & M] for t in s) if s else 0.0
        da, db = Qnu[a].split(), Dnu[b].split()
        s = set(da) & set(db)
        dsh[k] = len(s)
        dshw[k] = sum(ra[hash(t) & M] for t in s) if s else 0.0
        if da and db:
            fne[k] = 1.0 if max(da, key=len) == max(db, key=len) else 0.0

    qnw, dnw = Q.nw[qi], D.nw[di]
    qaww, daww = Q.aww[qi], D.aww[di]
    qnl, dnl = Q.nlen[qi], D.nlen[di]
    qal, dal = Q.alen[qi], D.alen[di]
    qdl, ddl = Q.dlen[qi], D.dlen[di]
    put(nsh); put(nshw); put(qnw - nshw); put(dnw - nshw)
    put(nsh / np.maximum(qnl + dnl - nsh, EPS))
    put(nsh / np.maximum(dnl, EPS)); put(nsh / np.maximum(qnl, EPS))
    put(ash); put(ashw); put(qaww - ashw); put(daww - ashw)
    put(ash / np.maximum(qal + dal - ash, EPS))
    put(ash / np.maximum(dal, EPS)); put(ash / np.maximum(qal, EPS))
    put(dsh); put(dshw)
    put(dsh / np.maximum(qdl + ddl - dsh, EPS))
    put(dsh / np.maximum(ddl, EPS)); put(fne)

    put(qnl); put(dnl); put(qal); put(dal); put(qdl); put(ddl)
    put(((dal + ddl) == 0).astype(np.float32))
    la, lb = Q.lgl[qi], D.lgl[di]
    ci = np.bitwise_count(np.bitwise_and(la, lb)).astype(np.float32)
    cu = np.bitwise_count(np.bitwise_or(la, lb)).astype(np.float32)
    put(np.where(cu > 0, ci / np.maximum(cu, EPS), np.float32(1.0)))
    put(D.domf[di])

    for k in RET_BASE + GRP_NAMES + WIDE_NAMES + DERIV + REV_NAMES + EMB_NAMES:
        put(C[k])
    if comp is None:
        for _ in COMP_NAMES + COH_NAMES:
            put(np.zeros(n, np.float32))
        put(np.full(n, np.float32(1.0 if src_is3 else 0.0)))
        assert c == NF, (c, NF)
        return F
    for k in COMP_NAMES:
        put(comp[k])

    # ---- cohesion with the entity's strongest candidate -------------------
    br = comp["best_row"].astype(np.int64)
    bi = di_all[br] if di_all is not None else di
    put(_cp(D.cn[di], D.cn[bi], fuzz.token_sort_ratio) / 100.0)
    put(_cp(D.aw[di], D.aw[bi], fuzz.token_sort_ratio) / 100.0)
    put((bi == di).astype(np.float32))
    exn = np.zeros(n, np.float32); exa = np.zeros(n, np.float32)
    exaw = np.zeros(n, np.float32)
    for k in range(n):
        b = bi[k]
        if b == di[k]:
            continue
        a = qi[k]
        sn = (set(Dcn[di[k]].split()) & set(Dcn[b].split())) - set(Qcn[a].split())
        exn[k] = len(sn)
        sa = (set(Daw[di[k]].split()) & set(Daw[b].split())) - set(Qaw[a].split())
        exa[k] = len(sa)
        if sa:
            exaw[k] = sum(ra[hash(t) & M] for t in sa)
    put(exn); put(exa); put(exaw)
    put(np.full(n, np.float32(1.0 if src_is3 else 0.0)))
    assert c == NF, (c, NF)
    return F
