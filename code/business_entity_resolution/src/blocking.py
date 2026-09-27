"""Stage 2 - candidate generation.

For every (source, country) partition we build TF-IDF cosine indexes over
tagged terms, one per *channel*:

  * NAME channel - name tokens, name 3-grams, whole-name keys
  * ADDR channel - address words, address numbers, address 3-grams
  * ALL  channel - everything

Each Source-1 record retrieves top-K from each channel and the union is its
candidate set.  Three channels are needed because the generator regularly
destroys one field outright: records whose address was blanked can only be
found by name, and records whose name was replaced can only be found by
address -- a single combined score lets the surviving field be outvoted.

Everything is flat numpy sort/merge, so the full 10M-record corpus is indexed
in a couple of GB.
"""
import os
import sys
import time

import numpy as np

from hw import THREADS, breathe

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from terms import GRP, NGRP, record_terms, set_df_tables, TAGS  # noqa: E402

_TC = {c: i for i, c in enumerate(TAGS)}


def hash_terms(rows, use_ngram=True):
    """rows: iterable of (nname, naddr, dom) -> (hash, row_idx, type_code)."""
    H, R = [], []
    for i, (nn, na, dm) in enumerate(rows):
        t = record_terms(nn, na, dm, use_ngram)
        if t:
            H.append(t)
            R.append(np.full(len(t), i, dtype=np.int32))
    if not H:
        return (np.empty(0, np.uint64), np.empty(0, np.int32),
                np.empty(0, np.uint8))
    flat = [x for sub in H for x in sub]
    ha = np.fromiter((hash(x) & 0xFFFFFFFFFFFFFFFF for x in flat),
                     dtype=np.uint64, count=len(flat))
    tc = np.fromiter((_TC[x[0]] for x in flat), dtype=np.uint8, count=len(flat))
    return ha, np.concatenate(R), tc


def _init_worker(dftab_path):
    if dftab_path:
        import numpy as _np
        z = _np.load(dftab_path)
        set_df_tables(z["name"], z["addr"])


def _worker(args):
    nn, na, dm, ng = args
    return hash_terms(zip(nn, na, dm), ng)


SHARD = int(os.environ.get("BER_SHARD", "25000"))


def hash_terms_par(nname, naddr, dom, pool, use_ngram=True, shard=None):
    """Hash every record's terms in parallel.

    Shards are kept small on purpose: with composite terms a record yields ~60
    terms, so a 12-way split of a 2M-record partition would make each worker
    materialise tens of millions of strings at once.
    """
    n = len(nname)
    k = shard or SHARD
    parts = [(nname[i:i + k], naddr[i:i + k], dom[i:i + k], use_ngram)
             for i in range(0, n, k)]
    Hs, Rs, Ts, off = [], [], [], 0
    for i, (h, r, t) in enumerate(pool.imap(_worker, parts, chunksize=1)):
        Hs.append(h)
        Rs.append(r + off)
        Ts.append(t)
        off += len(parts[i][0])
    if not Hs:
        return (np.empty(0, np.uint64), np.empty(0, np.int32),
                np.empty(0, np.uint8))
    return np.concatenate(Hs), np.concatenate(Rs), np.concatenate(Ts)


class Index:
    """TF-IDF cosine index over one channel (a subset of term types)."""

    def __init__(self, term_h, doc_i, tcode, n_doc, types=None,
                 tw=None, maxdf_frac=1.0):
        if types is not None:
            m = np.isin(tcode, np.asarray(types, np.uint8))
            term_h, doc_i, tcode = term_h[m], doc_i[m], tcode[m]
        order = np.argsort(term_h, kind="stable")
        th = term_h[order]
        post = doc_i[order]
        tc = tcode[order]
        uniq, start = np.unique(th, return_index=True)
        start = np.append(start, len(th)).astype(np.int64)
        df = np.diff(start).astype(np.int32)
        ttype = tc[start[:-1]]
        idf = np.log(1.0 + n_doc / df.astype(np.float32))
        if tw is not None:
            idf = idf * np.asarray(tw, np.float32)[ttype]
        # prune hopelessly common terms: they cost postings and carry no signal
        cap = max(64, int(maxdf_frac * n_doc))
        good = df <= cap
        if not good.all():
            post = post[np.repeat(good, df)]
            uniq, df, idf, ttype = uniq[good], df[good], idf[good], ttype[good]
            start = np.append(0, np.cumsum(df.astype(np.int64)))
        self.terms, self.post, self.start = uniq, post, start
        self.df, self.idf, self.ttype = df, idf, ttype
        self.n_doc = n_doc
        w2 = np.repeat((idf * idf).astype(np.float64), df)
        nrm = np.bincount(post, weights=w2, minlength=n_doc)
        self.dnorm = np.sqrt(np.maximum(nrm, 1e-9)).astype(np.float32)
        # number of channel terms per document (for the Fellegi-Sunter penalty)
        self.dl = np.bincount(post, minlength=n_doc).astype(np.float32)
        # per-document evidence mass per group, for containment features
        self.tgrp = np.asarray(GRP, np.uint8)[ttype]
        pg = np.repeat(self.tgrp, df)
        self.dmass = np.empty((NGRP, n_doc), np.float32)
        for g in range(NGRP):
            self.dmass[g] = np.bincount(post, weights=np.where(pg == g, w2, 0.0),
                                        minlength=n_doc)

    def lookup(self, h):
        if len(self.terms) == 0:
            return np.full(len(h), -1, np.int64)
        p = np.searchsorted(self.terms, h)
        p[p >= len(self.terms)] = 0
        ok = self.terms[p] == h
        p = p.astype(np.int64)
        p[~ok] = -1
        return p


def topk_query(idx, qh, qtype, qrow, nq, K, budget, types=None,
               batch=4096, log=None, min_score=0.0, beta=1.0,
               mode="fs", A=2.4, B=1.2, ipow=1.0, per_group=False):
    """Top-K cosine neighbours per query row.

    Each query spends its posting ``budget`` on its rarest terms first, so the
    cost per query is bounded no matter how common its vocabulary is.
    """
    if types is not None:
        m = np.isin(qtype, np.asarray(types, np.uint8))
        qh, qrow = qh[m], qrow[m]
    tid = idx.lookup(qh)
    keep = tid >= 0
    tid, qrow = tid[keep], qrow[keep]
    empty = (np.empty(0, np.int32), np.empty(0, np.int32), np.empty(0, np.float32))
    if len(tid) == 0:
        return (empty + ((np.empty((0, NGRP), np.float32),
                          np.zeros((nq, NGRP), np.float32)) if per_group else ()))
    dfv = idx.df[tid].astype(np.int64)
    order = np.lexsort((dfv, qrow))          # rarest term of each query first
    tid, qrow, dfv = tid[order], qrow[order], dfv[order]

    qstart = np.searchsorted(qrow, np.arange(nq + 1))
    cnt = np.diff(qstart)
    cs = np.cumsum(dfv)
    prev = np.where(qstart[:-1] > 0, cs[np.maximum(qstart[:-1] - 1, 0)], 0)
    cost = cs - np.repeat(prev, cnt)
    sel = cost <= budget
    first = np.zeros(len(dfv), bool)
    first[qstart[:-1][cnt > 0]] = True        # never drop a query's rarest term
    sel |= first
    tid, qrow = tid[sel], qrow[sel]

    qstart = np.searchsorted(qrow, np.arange(nq + 1))
    st = idx.start[tid]
    ln = idx.df[tid].astype(np.int64)
    if mode == "fs":
        # Fellegi-Sunter: log-likelihood ratio.  A shared term is worth its
        # idf; every term of either record left unexplained costs a constant.
        w = idx.idf[tid] ** ipow + A
    else:
        w = idx.idf[tid] ** 2
    qn = np.sqrt(np.maximum(np.bincount(qrow, weights=w.astype(np.float64),
                                        minlength=nq), 1e-9)).astype(np.float32)
    qmass = None
    if per_group:
        qg = idx.tgrp[tid]
        qmass = np.empty((nq, NGRP), np.float32)
        for g in range(NGRP):
            qmass[:, g] = np.bincount(qrow, weights=np.where(qg == g, w, 0.0)
                                      .astype(np.float64), minlength=nq)

    Q, D, S, G = [], [], [], []
    t0 = time.time()
    for b0 in range(0, nq, batch):
        tb = time.time()
        b1 = min(nq, b0 + batch)
        lo, hi = qstart[b0], qstart[b1]
        if hi <= lo:
            continue
        L = ln[lo:hi]
        tot = int(L.sum())
        if tot == 0:
            continue
        ends = np.cumsum(L)
        gidx = np.repeat(st[lo:hi] - (ends - L), L) + np.arange(tot, dtype=np.int64)
        docs = idx.post[gidx]
        key = np.repeat(qrow[lo:hi], L).astype(np.int64) * idx.n_doc + docs
        wts = np.repeat(w[lo:hi], L)
        o = np.argsort(key, kind="stable")
        key, wts, docs = key[o], wts[o], docs[o]
        newg = np.empty(len(key), bool)
        newg[0] = True
        np.not_equal(key[1:], key[:-1], out=newg[1:])
        gstart = np.flatnonzero(newg)
        sc = np.add.reduceat(wts, gstart)
        dd = docs[gstart]
        if per_group:
            pg = idx.tgrp[np.repeat(tid[lo:hi], L)][o]
            gm = np.empty((len(gstart), NGRP), np.float32)
            for g in range(NGRP):
                gm[:, g] = np.add.reduceat(np.where(pg == g, wts, 0.0), gstart)
        qq = (key[gstart] // idx.n_doc).astype(np.int64)
        if mode == "fs":
            sc = sc - B * idx.dl[dd]
        elif beta == 1.0:
            sc = sc / (idx.dnorm[dd] * qn[qq])
        elif beta > 0:
            sc = sc / (idx.dnorm[dd] ** beta)

        gq = np.searchsorted(qq, np.arange(b0, b1 + 1))
        for j in range(b1 - b0):
            a, e = gq[j], gq[j + 1]
            if e <= a:
                continue
            s = sc[a:e]
            p = np.argpartition(-s, K)[:K] if e - a > K else np.arange(e - a)
            if min_score > 0:
                p = p[s[p] >= min_score]
            if len(p):
                Q.append(np.full(len(p), b0 + j, np.int32))
                D.append(dd[a:e][p])
                S.append(s[p])
                if per_group:
                    G.append(gm[a:e][p])
        breathe(tb)
        if log and (b0 // batch) % 40 == 0:
            print(f"    q {b0:,}/{nq:,} {time.time()-t0:.0f}s", flush=True)
    if not Q:
        return (empty + ((np.empty((0, NGRP), np.float32), qmass)
                         if per_group else ()))
    res = (np.concatenate(Q), np.concatenate(D).astype(np.int32),
           np.concatenate(S).astype(np.float32))
    if per_group:
        res = res + (np.concatenate(G), qmass)
    return res
