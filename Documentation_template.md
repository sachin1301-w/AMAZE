# Business Entity Resolution — Methodology

> Numbers quoted below are measured on the training data at full corpus scale
> (every Source-1 query is matched against all ~10M Source-2/3 records of its
> split), with an entity-disjoint validation split. See "Results" for the table.

## 1. What the data actually is

Before designing anything we measured the structure of the labels, and three
facts determined the whole design.

**1. The matching is a partition.** The training ground truth contains 7,638,365
matched ids across 2,206,821 Source-1 entities, and **no Source-2/3 id is ever
reused** — the maximum reuse count is exactly 1. So each Source-2/3 record
belongs to at most one Source-1 entity. Candidate lists of different entities
can therefore be pruned *against each other*, and "another entity claims this
record more strongly" is direct evidence against a pair.

**2. `country` is a hard constraint.** Across 733,484 sampled true pairs there
were **zero** country mismatches. All blocking is done inside a country, which
both cuts the search space and keeps index sizes manageable. Country is treated
as an open set of string labels — partitions are discovered from the data, so
the unseen France label in the test set needs no special casing.

**3. The metric collapses to something simple.** With `P = TP/|S|` and
`R = TP/|T|`,

```
F_0.5 = 1.25·P·R / (0.25·P + R) = 1.25·TP / (|S| + 0.25·|T|)
```

Every extra prediction costs exactly 1 in the denominator and returns its
probability of being correct in the numerator. Two consequences drove our
effort allocation:

* **Precision is what matters.** With perfect precision, 95% recall still
  scores ≈0.99 macro. With perfect recall, 90% precision scores only ≈0.918.
* **Singletons are cheap to win and expensive to lose** (5.6% of entities): a
  correct empty prediction is worth a full 1.0, a single false merge on one is
  worth 0.

We therefore spent our blocking budget on getting recall "good enough"
(~0.96–0.97) at a *small* candidate count, and spent the modelling budget
almost entirely on precision.

### Noise model recovered from the data

Reading paired records showed a generator with a well-defined repertoire, which
we mirrored in the normaliser:

*Names* — legal-suffix add/drop/respell (`Ltd`↔`Limited`, `Corp`↔`Corporation`,
`L.L.C.`); injected filler words (`Center`, `Services`, `Partners`, `Company`);
diacritic injection (`Gold`→`Góld`); full transliteration into Devanagari,
Bengali, Gujarati, Oriya, Tamil, Telugu, Kannada or Malayalam; website form
(`victorylaboratories.com`); bracket/punctuation wrapping (`[Church]`,
`-- Holloway Peak Inc`); word-order transposition; token duplication
(`Kochar Kochar`); honorific and alias prefixes (`Mr`, `Smt`, `F/K/A`, `DBA`);
character-level typos and scrambles (`Wooten`→`Wheooitne`); and, in a minority
of cases, **outright replacement of the name with an unrelated string**
(`Irijaxnex`), which is unrecoverable from the name field alone.

*Addresses* — component reordering; street-type abbreviation (`Drive`↔`Dr`);
state/region spelled out, abbreviated or transliterated
(`Maharashtra`↔`MH`↔`महाराष्ट्र`; `Texas`↔`TX`); house-number zero padding
(`272`→`00272`) and prefixing (`Door No`, `H.NO 0684`, `#`, `PO Box`); dropped
components; `NULL` placeholders; digit-level corruption (`2535`→`253`); and
**a completely blanked address** in ~3% of Source-2/3 records.

The two failure modes that matter are the last one in each list: a record whose
name was replaced can only be found by address, and a record whose address was
blanked can only be found by name. Any single similarity score lets the
surviving field be outvoted — which is why candidate generation uses separate
channels (§2.3).

## 2. Candidate generation

Blocking is the part of this problem that has to scale, and it is where we
invested most of the engineering. The design target was **a small candidate set
per Source-1 entity at high recall**, computed for 1.73M entities against ~10M
records on one 12-core machine.

### 2.1 Normalisation

`normalize.py` / `translit.py`. NFKD decomposition with combining marks
dropped (kills injected diacritics); rule-based Brahmic→Latin transliteration;
case folding; `&`→`and`; punctuation stripped; whitespace collapsed.

The transliterator is worth a note. The nine Brahmic Unicode blocks share a
parallel code-point layout, so a *single* offset→Latin table plus inherent-vowel
and virama handling covers all of them in ~60 lines, with word-final schwa
deletion so that `लिमिटेड`→`limited` rather than `limiteda`. It is derived
purely from the Unicode block layout — no external transliteration package or
data.

Tokens are then classified with hand-written vocabularies for legal forms
(including French `SARL`/`SAS`/`SCI`/`EURL`/…), street types (including French
`Rue`/`Bd`/`Allée`/`Impasse`/…) and unit/filler markers (`Door No`, `Flat`,
`PO Box`, `Near`, `NULL`, and French articles).

### 2.2 Vocabularies learned from the labels

`learn_maps.py` aligns each Source-1 name/address with its true variants and
counts, per token, how often it appears in a variant but not the reference and
vice versa. Two things come out:

* **Injected filler tokens** — tokens the generator adds to names. Kept only
  when injected ≥200 times, ≥3× more often than they are dropped, **and** at
  least twice as often as they occur as a genuine Source-1 token. That last
  test matters: `lakshmi` is injected often but is also a real business name,
  and dropping it would delete the identifying part of the name. 228 tokens
  survive, including the transliteration round-trip artefacts (`praivet`,
  `limitet`, `prodakts`, `knstrakshan`). `core_name()` removes them — unless
  that would leave the name empty, in which case they are kept, because some
  businesses really are called "Service Center".
* **Alias pairs** — from pairs where exactly one token is unexplained on each
  side. This recovered, with no hard-coding, the full US state
  name↔abbreviation map, Indian state codes (`maharashtra|mh`, `ka|karnataka`,
  `telangana|tg`), street-type spellings (`rd|road`, `dr|drive`), and
  transliteration artefacts (`maharashtr|maharashtra`, `kerala|keralam`).

### 2.3 Composite terms — the key idea

Each record becomes a bag of *tagged* terms:

| tag | content |
|---|---|
| `n` | identifying name token |
| `a` | address word (street types canonicalised, filler dropped) |
| `d` | address number, zero-stripped |
| `g` / `h` | character 3-gram of a name token / address word (typo robustness) |
| `f` | whole-name keys: sorted-token concatenation, its `s`-stripped form, and first-4-chars-per-token |
| `p` | **number × address word** |
| `q` | **name token × name token** |
| `r` | **name token × address word** |
| `s` | **address word × address word** |

The last four are *conjunctive* terms, and they are what makes this work. A
truncated Indian variant such as `H NO 403, PUNE, MH` shares only `403` and
`pune` with its reference — both far too common to retrieve on. The pair
`(403, pune)` is rare. Composites are formed over the rarest few tokens of each
kind (3 numbers, 5 address words, 4 name tokens), where rarity comes from a
4 MB hashed `log2(df)` table so that worker processes do not each need a 1.4M
entry dictionary.

Adding composites moved recall@20 on the hardest partition (Source 3, India)
from **0.744 to 0.939** and raised the reachable ceiling from 0.959 to 0.994.
This single change mattered more than every scoring tweak we tried.

### 2.4 Index and retrieval

`blocking.py` builds, per (source, country) partition, a TF-IDF cosine index as
flat numpy arrays: term strings are hashed to uint64 (`PYTHONHASHSEED` pinned)
and sort/merged, so no vocabulary dict is ever materialised. Postings above a
document-frequency share of 0.0002 are pruned — they cost budget and carry no
signal.

Three **channels** are indexed over subsets of the term types:

* **name** (`n`, `g`, `f`, `q`) — finds records whose address was blanked
* **addr** (`a`, `d`, `h`, `p`, `s`) — finds records whose name was replaced
* **all** (everything) — the general case

Each Source-1 record takes top-K from each channel and the union is its wide
net. Retrieval spends a per-query **posting budget** (3000) on its rarest terms
first, so cost per query is bounded regardless of how common its vocabulary is;
tuning this down from 20000 gave a **15× speed-up with no recall loss**
(0.43 ms/query). Alongside the cosine score the retrieval returns the shared
IDF **mass per evidence group** (name / address words / numbers / composites) —
computed in the same sort-merge pass, so it is free — plus each side's group
mass, giving containment ratios. Masses are divided by `log(1+n_doc)²` so they
read as an effective count of shared rare terms and stay comparable across
partitions of very different sizes, which is what lets a model trained on US and
India transfer to France.

Scoring alternatives we measured and rejected: unnormalised IDF sum, BM25 length
normalisation (`b` swept), asymmetric cosine (`β` swept) and a Fellegi-Sunter
log-likelihood-ratio form. Plain cosine over the composite-enriched term space
beat all of them.

### 2.5 Deterministic prune — the candidate set we submit

The wide net is ~126 candidates per entity per source at **0.981** recall. A
single deterministic linear score then cuts it to **8 per source**:
a weighted sum of four rapidfuzz similarities (name `ratio`, name
`partial_ratio`, Jaro-Winkler on the sorted-token concatenation, address
`token_sort_ratio`), the channel cosine, and the per-group evidence containment
ratios. Weights were chosen by a random search on the training split to
maximise recall at the budget; no learned model is involved, and the scoring is
a fixed formula evaluated in numpy.

`candidate_pairs.tsv` is the output of this stage — exactly the set the matcher
runs inference over. Nothing is filtered after it.

## 3. Matching model

`features.py` computes **78 features** per candidate pair, in four families.

**Name similarity** — `ratio`, `token_sort_ratio`, `token_set_ratio`,
`partial_ratio` on the core-token string; `ratio` and Jaro-Winkler on the
order-invariant sorted-token concatenation; `token_set_ratio` on the raw
normalised name.

**Address similarity** — `ratio` on the whole address, `token_sort_ratio` and
`token_set_ratio` on canonicalised address words, `token_sort_ratio` on the
number multiset.

**Rarity-weighted set overlap** — for name tokens, address words and numbers
separately: count and rarity-weighted mass of the intersection, the mass left
unexplained on each side, Jaccard, and containment in each direction; plus
whether the longest number agrees, a blank-address flag, legal-form-set Jaccard
and a website-form flag. Rarity comes from the same hashed table, so sharing
`zephay` counts for far more than sharing `smith`.

**Retrieval and competition** — per-channel cosine and rank, per-group evidence
mass and containment, and then the features that make the partition constraint
usable: the candidate's prune score relative to the entity's best, its rank, the
gap to the best, the wide-net size (how crowded the entity's neighbourhood is),
and on the record side the best score any entity gives this record, the gap to
it, and whether this entity is that best.

### 3.1 Two stacked stages

Stage 1 is a LightGBM binary classifier over those features. Its probability is
then used to rebuild the competition features properly — `stack.py` adds 12 of
them (probability relative to the entity's best, rank, the entity's total
probability mass, the mass ranked above this candidate, and the record-side
equivalents: best claim by any entity, gap, rank, sum). Stage 2 is a second
LightGBM over features + that block. Statistics are computed within a partition
in both training and inference, so the two agree exactly.

This is where the partition constraint pays off: the strongest false merges are
records that genuinely resemble several entities, and stage 2 can see that
someone else has a better claim.

LightGBM is MIT-licensed; the pipeline uses no pretrained or large language
model, so the ≤8B-parameter and licence constraints are satisfied trivially.

## 4. Decision layer

`decide.py`. Given stage-2 probabilities:

1. **Exclusivity resolution** — each Source-2/3 record is kept only for the
   entity that scores it highest; competing pairs are zeroed. This is licensed
   directly by the "no id is ever reused" property of the labels.
2. **Expected-F_0.5 set selection** — using `F_0.5 = 1.25·TP/(|S| + 0.25·|T|)`,
   candidates are sorted by probability and, for each prefix length `k ≥ 1`,
   the expected score is evaluated as
   `1.25·Σ_{i≤k} p_i / (k + 0.25·(Σ_all p_i + m₀))`, where `m₀` accounts for
   true matches blocking never proposed. `k = 0` is scored exactly, as the
   probability that the entity is genuinely a singleton, `Π(1 - p_i)`. The
   best `k` wins. This chooses a different set size per entity instead of
   applying one global threshold, which is what the macro metric rewards, and
   it handles the 5.6% singletons explicitly rather than as a side effect.

`m₀`, a probability floor, a weight on the singleton branch, and whether to
apply exclusivity at all are tuned on held-out entities against the true macro
F_0.5.

## 5. Results

All figures are measured at **full corpus scale**: every Source-1 query is
retrieved against all ~10M Source-2/3 records of its split, partitioned by
country. Matching is scored on an **entity-disjoint** held-out split of the
training entities (44,340 entities, 153,433 true matches) with the same macro
F_0.5 the leaderboard uses -- singletons included, and matches that blocking
never proposed correctly counted as misses.

### Candidate generation

| stage | candidates per entity | recall |
|---|---|---|
| wide retrieval net (3 channels, K = 250/20/20 per source) | ~580 | 0.9852 |
| after the deterministic prune (30 per source) | **60** | **0.9770** |

Per partition, after the prune:

| partition | Source-1 queries | Source-2/3 records | recall |
|---|---|---|---|
| Source 2, India | 883,188 | 2,017,799 | 0.9603 |
| Source 2, US | 1,323,633 | 3,016,817 | 0.9925 |
| Source 3, India | 883,188 | 2,115,547 | 0.9509 |
| Source 3, US | 1,323,633 | 3,170,056 | 0.9904 |

On the test set this is **59.93 candidates per Source-1 entity**: a reduction
ratio of **6.0e-06** of the full cross product (60 of 9,969,589 possible
partners). India is consistently the harder half -- its addresses are longer and
more heavily truncated in the variants, and a larger share of its names are
transliterated into a native script.

**The composite-term result.** On the hardest partition (Source 3, India):

| term space | recall@20 | reachable ceiling |
|---|---|---|
| single tokens + 3-grams | 0.744 | 0.959 |
| **+ composite (conjunctive) terms** | **0.939** | **0.994** |

This single change mattered more than every scoring refinement combined.

**Net width.** Widening the combined channel is nearly free (retrieval cost is
dominated by posting traversal, not by K), and it buys real recall:

| K (combined channel) | candidates/query | pool recall (S3 India) |
|---|---|---|
| 60 | 65 | 0.9536 |
| 120 | 128 | 0.9644 |
| 250 | 251 | 0.9709 |
| 500 | 477 | 0.9747 |

**Prune retention.** With the tuned weights the prune keeps almost all of what
the net found:

| kept per source | 10 | 15 | 20 | 25 | 30 | 40 | 60 |
|---|---|---|---|---|---|---|---|
| recall | 0.9711 | 0.9758 | 0.9784 | 0.9821 | 0.9848 | 0.9873 | 0.9897 |

### Matching

| configuration | macro F_0.5 |
|---|---|
| oracle: perfect matcher on the same candidate set | 0.99178 |
| best single probability threshold | 0.9691 |
| **matcher + decision layer (submitted)** | **0.97147** |

with precision **0.99121** and recall **0.94106**; 86.6% of held-out entities
get a completely correct prediction.

Where the remaining 0.0203 sits:

| | share of entities | share of the deficit |
|---|---|---|
| false positives only | 2.45% | 36.3% |
| missed matches only | 10.69% | 59.2% |
| both | 0.31% | 4.5% |

### What we tried and rejected

Every one of these was implemented and measured, not assumed. They are the
reason we are confident about where the ceiling is.

| idea | result |
|---|---|
| **Reverse-direction retrieval** (each Source-2/3 record queries a Source-1 index) | +0.003 pool recall for ~2x the runtime. The composite terms already cover the truncated-record case it was designed to catch. Code retained, off by default. |
| **Learned set-size model** (regress the realised F_0.5 of every cut-off k on entity-level features, take the arg max) | 0.96887 vs 0.96903 for the tuned rule -- no gain. |
| **Exact expected-F_0.5 rule** (Poisson-binomial over the candidate probabilities) | 0.96711, *worse* than the mean-field rule, because match indicators inside an entity are strongly correlated rather than independent. |
| **Isotonic recalibration** of the pair probabilities | The fitted map is almost the identity; +0.0001. The probabilities were already calibrated. |
| **Cohesion features** (similarity to the entity's strongest candidate, and evidence shared with it that Source 1 lacks) | Implemented; earned no meaningful importance. |
| **Stage-2 stacking** on competition features from the stage-1 score | +0.0001 on the wide candidate set. Retained but not used in the shipped model, which is single stage. |
| **Metric-aligned pair weights** (weight by 1/(1.25 max(&#124;T&#124;,0.8))) | Distorts the probability calibration the selection rule depends on; left off by default. |
| **Early stopping on log-loss** | Log-loss bottoms out around 250 trees while macro F_0.5 keeps improving to 2200. Early stopping costs 0.001; disabled. |

### Dense embeddings: implemented, not shipped

The clearest remaining weakness is that every similarity in the shipped
pipeline is lexical.  That fails exactly where this data is hardest: a name
transliterated into Devanagari or Kannada shares no character n-gram with its
Latin form, and a scrambled token shares almost none.  India is correspondingly
the weak half of blocking (0.95 recall against 0.99 for the US).

We implemented the fix -- `embed.py` encodes every record with
`intfloat/multilingual-e5-small` (MIT licence, 118M parameters, well inside the
challenge's MIT/Apache and 8B limits; a general-purpose text encoder, not a
business directory) and `emb_cos` / `emb_rel` / `emb_rank` are wired through
both the prune ranker and the matcher.  Spot-checked on real pairs from the
training labels:

| case the lexical features cannot express | encoder cosine |
|---|---|
| Kannada vs Latin, true match | 0.909 |
| Devanagari vs Latin, true match | 0.939 |
| scrambled `Wheooitne` vs `Wooten`, true match | 0.956 |
| name replaced outright, true match | 0.964 |
| blank address, true match | 0.878 |

Encoding all 24.2M records takes about 80 minutes on a laptop RTX 3050, and
four of the six vector files were produced.  The remaining retrain and
inference passes did not fit the time budget, so **the submitted model does not
use embeddings**; the code path is present and defaults to off.  On the
evidence above we would expect it to help most on the Indian partitions, but we
are not reporting a number we did not measure end to end.

### Where the ceiling is

The loss decomposes cleanly, and the decomposition is what tells us to stop:

| | macro F_0.5 |
|---|---|
| perfect matcher on the candidate set (oracle) | 0.99178 |
| best prefix of the matcher's own ranking (oracle cut-off) | 0.98879 |
| **shipped: matcher + decision layer** | **0.97167** |

Ranking errors cost 0.00298 -- for **96.66%** of held-out entities the top-|T|
candidates by model score are *exactly* the true matches. Everything else,
0.01713, is not knowing where to cut.

That 0.01713 is irreducible, and we established it six independent ways. All six
land within 0.0005 of each other:

| method for choosing the set size | macro F_0.5 |
|---|---|
| tuned expected-F_0.5 rule (shipped) | 0.97256 |
| learned regressor on the probability vector | 0.96887 |
| learned regressor + mutual similarity between candidates | 0.97230 |
| exact Poisson-binomial expected F_0.5 | 0.96711 |
| isotonic recalibration then the tuned rule | +0.0001 |
| candidate-pool sweep, 10 to 60 per source | monotone, saturated |

The cause is visible in the data. Conditional true-match rates on held-out
pairs, against what the model predicts:

| candidate type | true match rate | model's mean probability |
|---|---|---|
| blank address, **exact** name match | 0.174 | 0.154 |
| present address, **exact** name match | 0.269 | 0.270 |
| blank address, record claimed by one entity only | 0.989 | 0.977 |
| blank address, record claimed by several entities | 0.114 | 0.102 |

An exact name match is only 17-27% likely to be a true match, because the
generator manufactures decoys: distinct businesses with identical names at
different addresses, and addresses differing in a single digit (`7108` vs
`7129 Muirfield Drive`, same name and street, labelled *not* a match). The model
reproduces every one of those rates to within 0.02, including on the slices
where its errors live -- it is reporting real uncertainty, not making a mistake,
and no decision rule can beat correctly reported uncertainty.

The same wall explains why extra blocking recall does not pay: raising it from
0.9644 to 0.9770 moved the final score by only **0.0018**, because the matches
recovered are exactly the ones whose evidence is too weak to confirm. And
training on 22% more rows with 45% more trees *cost* 0.0003.

Two final observations from error inspection, which is why we are confident the
remaining errors are not a missing feature. Where the numeric evidence
disagrees -- strong name and street agreement but no shared house number -- the
model separates the slice almost perfectly: of 14,493 such pairs it puts 10,481
below p = 0.2 (2.9% true) and 3,109 above p = 0.95 (98.3% true). And the false
merges that survive differ from a true match by one digit in one number, which
is also a documented *noise* pattern, so the two are not separable in principle.

### Predictions on the test set

1,732,544 entities; 5,648,873 predicted matches (3.260 per entity); 106,683
entities predicted as singletons (6.2%, against 5.6% in the training labels).
France, absent from all training data, receives 29.8 candidates per entity per
source and an expected 1.5 Source-2 matches per entity against 1.67 in the
training labels -- the country-agnostic, corpus-scale-normalised features
transfer to the unseen label without the model collapsing.

## 6. Engineering notes

* **Scale.** Peak RSS stays under ~6 GB on a 16 GB machine. The corpus is
  processed one (source, country) partition at a time; the wide net exists only
  for one batch of 40k Source-1 records at a time; term hashing is sharded at
  25k records so that a worker never materialises tens of millions of strings.
* **Reproducibility.** `PYTHONHASHSEED=0` is mandatory and set by `run_all.py`,
  because the blocking index hashes term strings rather than building a 20M
  entry vocabulary.
* **Cost control in retrieval.** Candidates are scored with vectorised evidence
  taken from the retrieval pass itself, and the expensive per-pair Python work
  (token-set intersections) runs only on the ~16 surviving candidates per
  entity rather than the ~250 in the wide net.
* **Fair play.** No external data source of any kind. The only "knowledge"
  beyond the provided files is rule tables written into the source
  (Unicode block layout, legal-form and street-type spellings) and the alias
  and filler vocabularies learned from `train_ground_truth.tsv`.
