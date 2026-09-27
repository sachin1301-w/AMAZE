"""Learn alias maps and injected-token vocabularies from the training labels.

Only the provided training data is used.  Two things are learned:

*  ``GENERIC`` - name tokens the generator injects into variants (they appear
   in a variant but not in the paired Source-1 name far more often than the
   reverse).  These must not be treated as identifying evidence.
*  ``ALIAS``   - token pairs that stand for the same thing (state name vs.
   postal abbreviation vs. transliterated native-script form, street-type
   spellings, transliteration artefacts).  Learned from pairs where exactly one
   token is unexplained on each side.
"""
import json
import os
import sys
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from normalize import LEGAL, PREFIX_NOISE, UNIT_NOISE, STREET  # noqa: E402

WORK = os.environ.get("BER_WORK", "../../../work")
DATA = os.environ.get("BER_DATA", "../../../dataset")
N_ENT = int(os.environ.get("BER_LEARN_ENT", "500000"))


def main():
    rng = np.random.default_rng(0)
    gt = pd.read_csv(f"{DATA}/train/train_ground_truth.tsv", sep="\t",
                     dtype=str, keep_default_na=False)
    gt = gt[gt.matched_entity_ids != ""]
    gt = gt.sample(min(N_ENT, len(gt)), random_state=0)
    s1ids = gt.source1_entity_id.str.slice(3).astype(np.int64).values
    mlists = [s.split(",") for s in gt.matched_entity_ids.values]

    need2, need3 = set(), set()
    for lst in mlists:
        for m in lst:
            (need2 if m[1] == "2" else need3).add(int(m[3:]))

    def load(src, keep):
        df = pd.read_parquet(f"{WORK}/train_s{src}.parquet")
        df = df[df.eid.isin(keep)]
        return dict(zip(df.eid.values, zip(df.nname.values, df.naddr.values)))

    r1 = load(1, set(s1ids.tolist()))
    r2 = load(2, need2)
    r3 = load(3, need3)
    print("loaded", len(r1), len(r2), len(r3), flush=True)

    inj_n, drop_n = Counter(), Counter()
    inj_a, drop_a = Counter(), Counter()
    alias_n, alias_a = Counter(), Counter()
    df_n, df_a = Counter(), Counter()

    for i, sid in enumerate(s1ids):
        if sid not in r1:
            continue
        n1, a1 = r1[sid]
        t1n, t1a = set(n1.split()), set(a1.split())
        df_n.update(t1n)
        df_a.update(t1a)
        for m in mlists[i]:
            rec = (r2 if m[1] == "2" else r3).get(int(m[3:]))
            if rec is None:
                continue
            n2, a2 = rec
            t2n, t2a = set(n2.split()), set(a2.split())
            for (A, B, inj, drop, alias) in (
                    (t1n, t2n, inj_n, drop_n, alias_n),
                    (t1a, t2a, inj_a, drop_a, alias_a)):
                extra = B - A
                miss = A - B
                inj.update(extra)
                drop.update(miss)
                if len(extra) == 1 and len(miss) == 1:
                    x = next(iter(extra))
                    y = next(iter(miss))
                    if x != y:
                        alias[(y, x) if y < x else (x, y)] += 1
        if i % 100000 == 0:
            print(" ", i, flush=True)

    out = {}
    # --- generic / injected name tokens -----------------------------------
    known = set(LEGAL) | set(LEGAL.values()) | PREFIX_NOISE
    gen = []
    for t, c in inj_n.most_common(6000):
        if t in known or len(t) < 2:
            continue
        d = drop_n.get(t, 0) + 1
        # Keep only tokens the generator *injects* far more often than they
        # occur as genuine Source-1 name tokens.  The df_n test matters: words
        # like "lakshmi" are injected often but are also real business names,
        # and dropping them would destroy the identifying part of the name.
        if c >= 200 and c / d >= 3.0 and df_n.get(t, 0) * 2.0 < c:
            gen.append((t, c, df_n.get(t, 0)))
    out["generic_name"] = [g[0] for g in gen]
    out["generic_counts"] = {g[0]: [g[1], g[2]] for g in gen}
    print("\ntop injected NAME tokens:", inj_n.most_common(60))
    print("\nlearned generic_name (%d):" % len(gen), gen[:80])

    # --- aliases ----------------------------------------------------------
    def keep_alias(counter, minc, tag):
        tot = defaultdict(int)
        for (a, b), c in counter.items():
            tot[a] += c
            tot[b] += c
        res = {}
        for (a, b), c in counter.most_common(20000):
            if c < minc:
                break
            res["%s|%s" % (a, b)] = c
        print("\ntop %s aliases:" % tag,
              [(k, v) for k, v in list(res.items())[:70]])
        return res

    out["alias_addr"] = keep_alias(alias_a, 60, "ADDR")
    out["alias_name"] = keep_alias(alias_n, 60, "NAME")
    print("\ntop injected ADDR tokens:", inj_a.most_common(50))
    print("\ntop dropped ADDR tokens:", drop_a.most_common(50))

    with open(f"{WORK}/learned_maps.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("\nwrote learned_maps.json")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
