# Business Entity Resolution — pipeline

Resolves records from Source 2 and Source 3 against the deduplicated Source 1
reference, for the full challenge data (2.2M train / 1.73M test Source-1
entities against ~10M Source-2/3 records per split).

Everything runs on CPU with only the provided data. No external database, API,
gazetteer or geocoder is consulted; every vocabulary the pipeline uses is either
a hand-written rule table or is learned from `train_ground_truth.tsv`.

## Layout

```
src/
  translit.py    Brahmic -> Latin transliteration (one Unicode offset table for
                 Devanagari, Bengali, Gurmukhi, Gujarati, Oriya, Tamil, Telugu,
                 Kannada, Malayalam)
  normalize.py   accent/punctuation/case folding, legal-form, street-type and
                 unit-marker vocabularies (US, India and France forms)
  learn_maps.py  learns, from the training labels, (a) the filler tokens the
                 generator injects into names and (b) alias pairs such as
                 maharashtra/mh, texas/tx, road/rd
  dfstats.py     token document frequencies per split
  mkdftab.py     packs them into a 4 MB hashed log2(df) rarity table
  terms.py       turns a record into tagged terms, including the composite
                 (conjunctive) terms that make blocking work
  blocking.py    flat-numpy inverted index + top-K retrieval per channel
  features.py    pair features; the deterministic prune score
  ksel.py        learned set-size selection (measured: no better than the
                 tuned expected-F_0.5 rule -- kept for the record)
  diagnose.py    decomposes the matcher's loss against the oracle
  tune_decide.py retunes the decision layer from saved validation predictions
  pipeline.py    per-partition retrieve -> prune -> features
  stack.py       stage-2 competition features from the stage-1 probability
  runpipe.py     driver over a split (labels for train, predictions for test)
  fit.py         trains the two matcher stages, tunes the decision layer
  decide.py      the F_0.5 metric, exclusivity resolution, set selection
  submit.py      writes matching_results.tsv and candidate_pairs.tsv
  run_all.py     end-to-end driver
  embed.py       optional dense encoder pass (off by default; see the
                 methodology document for why it is not in the shipped model)
  hw.py          CPU budget: BER_THREADS / BER_THROTTLE keep a thin laptop
                 from cooking during the multi-hour passes
  train_prune.py trains the learned prune ranker
  thermal_test.py  load test for choosing the CPU budget
  tune_prune.py  offline weight search for the prune score
  blockeval.py, ceil.py, diag.py   blocking diagnostics
```

## Reproduce

```bash
python -m pip install -r requirements.txt

# expects  <root>/dataset/{train,test}/*.tsv
cd src
python run_all.py
```

`run_all.py` writes intermediates to `<root>/work/` and the two submission
files to `<root>/output/`. Stages are idempotent and skipped when their
artefacts exist; `--from <stage>` resumes at `prep`, `dfstats`, `dftab`,
`maps`, `train_pairs`, `fit`, `test_pairs` or `submit`.

**`PYTHONHASHSEED=0` is required** and `run_all.py` sets it for every stage.
The blocking index hashes term strings with Python's `hash()` instead of
building a 20M-entry vocabulary dict, so the seed must be pinned or the index
and the queries would disagree.

Validate before uploading:

```bash
python utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test
```

## Cost

Measured on 12 logical cores / 16 GB RAM, which is what the memory budgets in
the code assume:

| stage | time |
|---|---|
| normalise all six source files | ~3 min |
| token df + rarity tables | ~3 min |
| learn alias/filler maps | ~6 min |
| blocking + features + labels, full train | ~1 h 20 min |
| train the matcher (2200 trees, 191 leaves) | ~30 min |
| blocking + features + inference, full test | ~2 h 20 min |

Peak RSS stays under ~7 GB. The corpus is processed one (source, country)
partition at a time; the wide candidate net exists only for one batch of
Source-1 records at a time; term hashing is sharded so no worker materialises
tens of millions of strings; and the components carried from the net to the
feature matrix are stored as float16, since LightGBM bins every feature into 255
buckets anyway and a US partition carries ~40M pairs.

Set `BER_QBATCH=20000` on a 16 GB machine -- the default 40000 roughly doubles
the transient net.

## Knobs

Set as environment variables (defaults in `pipeline.CFG`):

| var | default | meaning |
|---|---|---|
| `BER_KALL` | 250 | top-K from the combined channel |
| `BER_KNAME` / `BER_KADDR` | 20 | top-K from the name-only / address-only channels |
| `BER_KEEP` | 30 | candidates kept per entity **per source** after the prune |
| `BER_BUDGET` | 4000 | posting-list budget per query |
| `BER_REV` | 0 | reverse-direction retrieval (measured: +0.003 pool recall for ~2x runtime) |
| `BER_KREV` | 4 | parents retrieved per Source-2/3 record when `BER_REV=1` |
| `BER_MAXDF` | 0.0002 | index prunes terms above this document-frequency share |
| `BER_QBATCH` | 40000 | Source-1 records retrieved per batch |
| `BER_SHARD` | 25000 | records per term-hashing shard (memory bound) |
| `BER_THREADS` | 6 | threads for every stage; 12 runs flat out and hot |
| `BER_THROTTLE` | 0.25 | pause after each batch, as a fraction of its runtime |
| `BER_QBATCH` | 40000 | lower to 20000 on a 16 GB machine |

Running cool (`BER_THREADS=6 BER_THROTTLE=0.25`, the defaults) costs about
1.6x wall time versus `BER_THREADS=12 BER_THROTTLE=0`, and keeps a thin laptop
well away from thermal throttling.
