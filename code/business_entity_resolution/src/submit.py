"""Turn per-partition predictions into the two submission files.

    python submit.py --tag pred --outdir ../../../output

Writes matching_results.tsv (the selected matches) and candidate_pairs.tsv
(every pair the matcher scored, i.e. the output of blocking + the deterministic
lexical filter).  Every Source-1 entity of the test set gets exactly one row in
both files, empty when nothing was selected.
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
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from decide import resolve_exclusive, select_sets

WORK = os.environ.get("BER_WORK", "../../../work")
DATA = os.environ.get("BER_DATA", "../../../dataset")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="pred")
    ap.add_argument("--split", default="test")
    ap.add_argument("--outdir", default="../../../output")
    ap.add_argument("--decide", default="")
    a = ap.parse_args()
    t0 = time.time()
    os.makedirs(a.outdir, exist_ok=True)

    ids = pd.read_csv(f"{DATA}/{a.split}/{a.split}_source1.tsv", sep="\t",
                      dtype=str, keep_default_na=False,
                      usecols=["entity_id"]).entity_id.values
    ent = np.array([int(x[3:]) for x in ids], np.int64)
    order = np.argsort(ent)
    ent_sorted = ent[order]
    nq = len(ent)
    print(f"{nq:,} Source-1 entities", flush=True)

    files = sorted(glob.glob(f"{WORK}/{a.split}_{a.tag}_s*.npz"))
    assert files, f"no prediction files for tag {a.tag}"
    QI, DS, PP, SR = [], [], [], []
    for f in files:
        z = np.load(f)
        src = 2 if "_s2_" in f else 3
        QI.append(order[np.searchsorted(ent_sorted, z["s1"])])
        DS.append(z["d"])
        PP.append(z["p"].astype(np.float32))
        SR.append(np.full(len(z["d"]), src, np.int8))
        print(f"  {os.path.basename(f)} {len(z['d']):,} pairs", flush=True)
    qi = np.concatenate(QI)
    ds = np.concatenate(DS)
    pp = np.concatenate(PP)
    sr = np.concatenate(SR)
    del QI, DS, PP, SR
    print(f"total pairs {len(qi):,} ({len(qi)/nq:.2f} per entity) "
          f"{time.time()-t0:.0f}s", flush=True)

    cfg = dict(excl=True, m0=0.15, pw=1.0, k0=2.5, pmin=0.0)
    if a.decide:
        cfg.update(json.load(open(a.decide)))
    print("decision config:", cfg, flush=True)

    p = pp
    if cfg.get("excl", True):
        dk = ds.astype(np.int64) * 4 + sr
        ud, di = np.unique(dk, return_inverse=True)
        keep = resolve_exclusive(di, p, len(ud))
        p = np.where(keep, p, np.float32(0.0))
        print(f"exclusivity dropped {int((~keep).sum()):,} pairs "
              f"{time.time()-t0:.0f}s", flush=True)
        del dk, ud, di

    sel = select_sets(qi, p, nq, m0=cfg["m0"], pmin=cfg["pmin"],
                      k0_boost=cfg["k0"], pw=cfg.get("pw", 1.0))
    print(f"selected {int(sel.sum()):,} pairs ({sel.sum()/nq:.3f} per entity) "
          f"{time.time()-t0:.0f}s", flush=True)

    def write(path, mask, colname):
        """Stream one row per Source-1 entity, in the test file's own order.

        Ids are formatted per entity instead of materialising ~35M python
        strings at once, which would cost several GB.
        """
        q = qi if mask is None else qi[mask]
        dv = ds if mask is None else ds[mask]
        sv = sr if mask is None else sr[mask]
        o = np.argsort(q, kind="stable")
        q, dv, sv = q[o], dv[o], sv[o]
        bounds = np.searchsorted(q, np.arange(nq + 1))
        nz = nids = 0
        nl = "\n"
        with open(path, "w", encoding="utf-8", newline=nl) as f:
            f.write("source1_entity_id\t" + colname + nl)
            for i in range(nq):
                a0, e0 = bounds[i], bounds[i + 1]
                f.write(ids[i])
                f.write("\t")
                if e0 > a0:
                    parts = ["S2-" + str(v) if t == 2 else "S3-" + str(v)
                             for v, t in zip(dv[a0:e0], sv[a0:e0])]
                    parts = list(dict.fromkeys(parts))
                    f.write(",".join(parts))
                    nz += 1
                    nids += len(parts)
                f.write(nl)
        print(f"wrote {path}: {nz:,} non-empty rows, {nids:,} ids, "
              f"{nids/nq:.3f} per entity  {time.time()-t0:.0f}s", flush=True)

    write(os.path.join(a.outdir, "matching_results.tsv"), sel,
          "matched_entity_ids")
    write(os.path.join(a.outdir, "candidate_pairs.tsv"), None,
          "candidate_entity_ids")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
