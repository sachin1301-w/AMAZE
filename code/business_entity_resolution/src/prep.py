"""Stage 1 - normalise every source file into a compact parquet table.

Usage:  python prep.py <split>            # split = train | test
"""
import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import hw  # noqa: E402  (must precede numpy/torch)

import os
import sys
import time

import numpy as np

from hw import THREADS
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from normalize import base_norm, domain_stem   # noqa: E402

DATA = os.environ.get("BER_DATA", "dataset")
WORK = os.environ.get("BER_WORK", "work")
CHUNK = 250_000


def _norm_block(args):
    names, addrs = args
    nn = [base_norm(x) for x in names]
    na = [base_norm(x) for x in addrs]
    dm = [domain_stem(x) if "." in x else "" for x in names]
    return nn, na, dm


def prep_file(split, src, pool):
    path = f"{DATA}/{split}/{split}_source{src}.tsv"
    out = f"{WORK}/{split}_s{src}.parquet"
    if os.path.exists(out):
        print("exists", out)
        return
    t0 = time.time()
    frames = []
    for ch in pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False,
                          chunksize=CHUNK):
        names = ch.business_name.values
        addrs = ch.business_address.values
        k = max(1, len(names) // 12 + 1)
        parts = [(names[i:i + k], addrs[i:i + k]) for i in range(0, len(names), k)]
        res = pool.map(_norm_block, parts)
        nn = [x for r in res for x in r[0]]
        na = [x for r in res for x in r[1]]
        dm = [x for r in res for x in r[2]]
        frames.append(pd.DataFrame({
            "eid": ch.entity_id.str.slice(3).astype(np.int64),
            "country": ch.country.values,
            "nname": nn, "naddr": na, "dom": dm,
        }))
        print(f"  {split}_s{src}: {sum(len(f) for f in frames):,} rows "
              f"{time.time()-t0:.0f}s", flush=True)
    df = pd.concat(frames, ignore_index=True)
    df["country"] = df["country"].astype("category")
    df.to_parquet(out, index=False, compression="zstd")
    print(f"wrote {out} {len(df):,} rows in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    import multiprocessing as mp
    os.makedirs(WORK, exist_ok=True)
    split = sys.argv[1]
    with mp.Pool(THREADS) as pool:
        for src in (1, 2, 3):
            prep_file(split, src, pool)
