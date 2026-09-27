# AMAZE — Business Entity Resolution

Matching business records across three noisy, independent sources at full
scale: **1.73M Source-1 test entities against ~10M Source-2/3 records**, on one
12-core laptop with 16 GB of RAM.

**Result: macro F₀.₅ = 0.9715** on a held-out split of 44,340 training
entities, with precision 0.9912 and blocking recall 0.9770, at **60 candidates
per entity** — a reduction ratio of 6.0 × 10⁻⁶ of the full cross product.

## What the data looks like

Three measurements shaped the whole design:

- **The matching is a partition.** Across 7,638,365 matched ids in the training
  labels, no Source-2/3 record is ever claimed by two entities. So candidate
  lists can be pruned against one another.
- **Country never crosses.** Zero mismatches in 733,484 sampled true pairs, so
  blocking runs per country. The test set adds France, which appears in no
  training data; nothing in the pipeline is hard-coded to a country list.
- **The metric collapses.** `F₀.₅ = 1.25·TP / (|S| + 0.25·|T|)`. Each extra
  prediction costs exactly 1 in the denominator and returns its probability of
  being right. Precision dominates, and singletons (5.6% of entities) are worth
  a full 1.0 when correctly left empty.

The generator corrupts names by dropping and inventing legal suffixes,
injecting diacritics, transliterating into nine Indic scripts, scrambling
characters (`Wooten` → `Wheooitne`), replacing names outright, and rewriting
them as domains. Addresses get reordered, abbreviated, zero-padded, truncated,
or blanked entirely.

## How it works

1. **Normalise** — accent folding plus a rule-based Brahmic→Latin
   transliterator covering Devanagari, Bengali, Gurmukhi, Gujarati, Oriya,
   Tamil, Telugu, Kannada and Malayalam from a single Unicode offset table,
   with word-final schwa deletion so `लिमिटेड` → `limited`.
2. **Learn the vocabularies from the labels** — aligning each entity with its
   true variants recovers the US state abbreviation map, Indian state codes,
   street-type spellings and the 228 filler words the generator injects, with
   no hard-coded gazetteer.
3. **Block with composite terms** — the key idea. Indexing *token pairs*
   (number × address word, name × name, …) rather than single tokens. `403` and
   `pune` are each useless; `(403, pune)` is rare. This moved recall@20 on the
   hardest partition from **0.744 to 0.939**.
4. **Retrieve over three channels** — name-only, address-only and combined, so
   a blanked address or a replaced name cannot be outvoted by the surviving
   field. Flat-numpy inverted indexes handle the 10M-record corpus in a couple
   of GB; a per-query posting budget bounds cost at 0.43 ms/query.
5. **Prune deterministically** to the candidate set that is submitted.
6. **Match** with LightGBM over 90 features — string similarity, rarity-weighted
   set overlap, retrieval evidence, and competition features that exploit the
   partition constraint.
7. **Decide** per entity by maximising expected F₀.₅, choosing a different set
   size for each entity rather than applying one global threshold.

## Reproducing

See [`code/business_entity_resolution/README.md`](code/business_entity_resolution/README.md).

```bash
python -m pip install -r code/business_entity_resolution/requirements.txt
cd code/business_entity_resolution/src
python run_all.py
```

`PYTHONHASHSEED=0` is required and set by the driver — the blocking index hashes
term strings instead of building a 20M-entry vocabulary.

Data, intermediates and outputs are gitignored: the challenge data is 2.4 GB and
not ours to redistribute, intermediates run to 38 GB, and `candidate_pairs.tsv`
alone is 1.3 GB.

## Methodology

[`Documentation_template.md`](Documentation_template.md) has the full write-up,
including the measured dead ends — reverse-direction retrieval, a learned
set-size model, the exact Poisson-binomial decision rule, isotonic
recalibration, cohesion features and stage-2 stacking all returned ≤0.002, and
the evidence for why.
