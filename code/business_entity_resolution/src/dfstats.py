"""Corpus token document frequencies for a split.

Counts how many records contain each name token and each address token, over
all three sources of the split.  Used only to decide *rarity* -- which tokens
are worth forming composite blocking keys from.

    python dfstats.py train
"""
import os
import pickle
import sys
import time
from collections import Counter

import pandas as pd

WORK = os.environ.get("BER_WORK", "../../../work")


def main():
    split = sys.argv[1]
    t0 = time.time()
    dfn, dfa = Counter(), Counter()
    for src in (1, 2, 3):
        df = pd.read_parquet(f"{WORK}/{split}_s{src}.parquet",
                             columns=["nname", "naddr"])
        for col, C in (("nname", dfn), ("naddr", dfa)):
            v = df[col].values
            for i in range(0, len(v), 200_000):
                for s in v[i:i + 200_000]:
                    C.update(set(s.split()))
        print(f"  s{src}: name vocab {len(dfn):,} addr vocab {len(dfa):,} "
              f"{time.time()-t0:.0f}s", flush=True)
        del df
    with open(f"{WORK}/{split}_df.pkl", "wb") as f:
        pickle.dump((dfn, dfa), f)
    print(f"wrote {WORK}/{split}_df.pkl")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
