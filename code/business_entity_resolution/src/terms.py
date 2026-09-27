"""Term extraction for the blocking index.

Each record becomes a bag of *tagged* terms.  Terms are hashed to uint64
(PYTHONHASHSEED is pinned by the driver so hashing is stable across processes
and runs), so the blocking index never materialises a vocabulary dict.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from normalize import (LEGAL, PREFIX_NOISE, STREET, UNIT_NOISE,  # noqa: E402
                       addr_tokens, canon_addr_words, strip_zeros)

_here = os.path.dirname(os.path.abspath(__file__))
_mp = os.environ.get("BER_MAPS", os.path.join(_here, "learned_maps.json"))
if os.path.exists(_mp):
    _M = json.load(open(_mp, encoding="utf-8"))
    GENERIC = set(_M.get("generic_name", []))
    ALIAS_ADDR = _M.get("alias_canon_addr", {})
    ALIAS_NAME = _M.get("alias_canon_name", {})
else:
    GENERIC, ALIAS_ADDR, ALIAS_NAME = set(), {}, {}

LEGSET = set(LEGAL) | set(LEGAL.values())
_SKIP_NAME = LEGSET | PREFIX_NOISE | {"and", "of", "the", "for", "de", "du",
                                      "des", "la", "le", "les", "et", "www"}

NUL = "\x00"


def core_name(nname):
    """Identifying name tokens: legal forms, honorifics and the filler words
    the generator injects are removed.

    If stripping the filler words would leave nothing, they are kept -- some
    businesses really are called "Service Center", and an empty core name would
    throw away the only name evidence there is.
    """
    out, fallback = [], []
    for t in nname.split():
        if len(t) < 2 or t in _SKIP_NAME:
            continue
        t = ALIAS_NAME.get(t, t)
        if t in GENERIC:
            fallback.append(t)
        else:
            out.append(t)
    return out if out else fallback


def legal_of(nname):
    return sorted({LEGAL[t] for t in nname.split() if t in LEGAL})


def addr_parts(naddr):
    w, nums = addr_tokens(naddr)
    w = canon_addr_words(w)
    w = [ALIAS_ADDR.get(t, t) for t in w]
    nums = [strip_zeros(x) for x in nums]
    return w, [x for x in nums if x != "0"]


def grams(tok, n=3):
    return [tok[i:i + n] for i in range(len(tok) - n + 1)]


# term type codes -- the tag character that precedes NUL in each term string
#   n name token    a address word    d address number
#   g name 3-gram   h address 3-gram  f whole-name key
# composite (conjunctive) terms, far rarer than their parts and therefore the
# main source of cheap, high-precision anchors:
#   p number x address word           q name token x name token
#   r name token x address word       s address word x address word
TAGS = "nadghfpqrs"
# evidence groups used for per-group retrieval scores
#   0 pure name (n,g,f)  1 address words (a,h)  2 numbers (d)  3 composites
GRP = (0, 1, 2, 0, 1, 0, 3, 3, 3, 3)
NGRP = 4

CH_NAME = (0, 3, 5, 7)
CH_ADDR = (1, 2, 4, 6, 9)
CH_ALL = tuple(range(10))

# how many of the rarest tokens of each kind take part in composites
CAP_NUM, CAP_AW, CAP_NT = 3, 5, 4

# Rarity lookup.  Exact document frequencies would need a 1.4M-entry dict in
# every worker process, so we keep a hashed table of log2(df) buckets instead:
# 4 MB, and an occasional collision only perturbs which tokens get picked for
# composites (consistently on both sides of a comparison).
DF_BITS = 22
DF_MASK = (1 << DF_BITS) - 1
DF_ADDR = None
DF_NAME = None


def set_df_tables(name_tab, addr_tab):
    global DF_NAME, DF_ADDR
    DF_NAME, DF_ADDR = name_tab, addr_tab


def _rarest(toks, tab, cap):
    u = set(toks)
    if len(u) <= cap:
        return sorted(u)
    if tab is None:
        return sorted(u)[:cap]
    return sorted(sorted(u, key=lambda t: (tab[hash(t) & DF_MASK], t))[:cap])


def record_terms(nname, naddr, dom, use_ngram=True, composite=True):
    """Tagged term list for one record (see TAGS for the tag alphabet)."""
    cn = core_name(nname)
    w, nums = addr_parts(naddr)
    T = []
    for t in set(cn):
        T.append("n" + NUL + t)
    for t in set(w):
        T.append("a" + NUL + t)
    for t in set(nums):
        T.append("d" + NUL + t)
    if dom:
        T.append("n" + NUL + dom)
        T.append("f" + NUL + dom)
    if cn:
        # whole-name keys catch short generic names that no single token can
        srt = "".join(sorted(cn))
        T.append("f" + NUL + srt)
        T.append("f" + NUL + srt.rstrip("s"))
        T.append("f" + NUL + "".join(t[:4] for t in cn))
    if use_ngram:
        g = set()
        for t in cn:
            if len(t) >= 4:
                g.update(grams(t))
        for x in g:
            T.append("g" + NUL + x)
        h = set()
        for t in w:
            if len(t) >= 6:
                h.update(grams(t))
        for x in h:
            T.append("h" + NUL + x)
    if composite:
        rn = _rarest(nums, DF_ADDR, CAP_NUM)
        rw = _rarest(w, DF_ADDR, CAP_AW)
        rt = _rarest(cn, DF_NAME, CAP_NT)
        for a in rn:
            for b in rw:
                T.append("p" + NUL + a + NUL + b)
        for i in range(len(rt)):
            for j in range(i + 1, len(rt)):
                T.append("q" + NUL + rt[i] + NUL + rt[j])
        for a in rt[:3]:
            for b in rw[:3]:
                T.append("r" + NUL + a + NUL + b)
        rw4 = rw[:4]
        for i in range(len(rw4)):
            for j in range(i + 1, len(rw4)):
                T.append("s" + NUL + rw4[i] + NUL + rw4[j])
    return T
