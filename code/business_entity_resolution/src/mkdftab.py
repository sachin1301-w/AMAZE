"""Build the hashed log2(df) rarity tables used to pick composite terms."""
import os, pickle, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from terms import DF_BITS, DF_MASK
WORK = os.environ.get("BER_WORK", "../../../work")
split = sys.argv[1]
dfn, dfa = pickle.load(open(f"{WORK}/{split}_df.pkl", "rb"))
out = {}
for nm, C in (("name", dfn), ("addr", dfa)):
    t = np.zeros(1 << DF_BITS, np.uint8)
    for tok, c in C.items():
        h = hash(tok) & DF_MASK
        v = min(255, int(c).bit_length())
        if v > t[h]:
            t[h] = v
    out[nm] = t
    print(nm, "filled", int((t > 0).sum()), "/", len(C))
np.savez(f"{WORK}/{split}_dftab.npz", **out)
print("wrote", f"{WORK}/{split}_dftab.npz")
