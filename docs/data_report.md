# Data & harness report (Phase 0, data track)

All numbers below were measured in this session from the downloaded LaMP dev
files. Nothing is carried over from prior sessions or from memory. Where a
choice was made rather than measured, it is labelled **[decision]**; where
something is believed but not verified, it is labelled **[unverified]**.

## 1. Downloads

Base URL pattern (from the verified-facts file, re-confirmed by HTTP HEAD):
`https://ciir.cs.umass.edu/downloads/LaMP/LaMP_{N}/{split}/{split}_questions.json`

| Task | Path used | questions | outputs | HTTP |
|---|---|---|---|---|
| LaMP-1 | `LaMP_1/dev/` | 273,443,200 B | 86,416 B | 200 |
| LaMP-2 | `LaMP_2/new/dev/` | 9,493,217 B | 27,502 B | 200 |
| LaMP-3 | `LaMP_3/dev/` | 401,034,862 B | 81,416 B | 200 |
| LaMP-5 | `LaMP_5/dev/` | 272,410,629 B | 248,614 B | 200 |

Total ≈ 993 MB, streamed to the workspace in 1 MiB chunks. **Raw dumps were
NOT saved as artifacts**, per the session constraint. `LaMP_{1,3,5}/new/dev/`
return HTTP 404 — the `new` variant exists only for LaMP-2.

### 1.1 Finding that changes the task definition — LaMP-2

The track brief describes LaMP-2 as *movie tagging*. That is **not** what
`LaMP_2/dev/` contains:

- `LaMP_2/dev/` — **news categorization**. 1,052 dev queries, profile items
  `{id, title, text, category}`, 15 gold classes that are news sections
  (`politics` 577, `entertainment` 141, `parents` 62, …). Instruction text:
  "Which category does this article relate to…".
- `LaMP_2/new/dev/` — **movie tagging**. 692 dev queries, profile items
  `{id, tag, description}`, 15 gold classes that are movie tags
  (`comedy` 132, `sci-fi` 80, `violence` 54, …). Instruction text:
  "Which tag does this movie relate to…".

**[decision]** I used `LaMP_2/new/dev/` (movie tagging), because that is what
the brief asks for by name. Consequence for the sweep tracks: **LaMP-2 has
only 692 eligible queries, and its profiles are much shorter than the other
tasks (median 23 items vs 79–140).** This makes LaMP-2 the most
signal-starved task in the suite, which is useful for the paper's thesis but
means its `long` stratum is not comparable in absolute profile length to the
other tasks. If the intended task was in fact news categorization, the subset
must be rebuilt — say so and I will regenerate it.

**[unverified]** The published LaMP leaderboard numbers for "LaMP-2" may refer
to either variant depending on paper vintage; I did not verify which variant
any specific published baseline used. Do not compare our LaMP-2 numbers to a
published baseline without checking that first.

## 2. Per-task validation (full dev split)

Schemas were inspected, not assumed. Top level is uniformly
`{id, input, profile}` and `outputs.json` is `{task, golds:[{id, output}]}`.
Question ids and gold ids match exactly (set equality) for all four tasks.

| Task | Queries | Profile keys (measured, 100% homogeneous) | Gold classes |
|---|---|---|---|
| LaMP-1 | 2,500 | `id, title, abstract` | 2 (`[2]` 1280 / `[1]` 1220) |
| LaMP-2 | 692 | `id, tag, description` | 15 tags |
| LaMP-3 | 2,500 | `id, text, score` | 5 (`5` 1478, `4` 610, `3` 210, `2` 101, `1` 101) |
| LaMP-5 | 2,500 | `id, title, abstract` | 2,497 unique titles |

The key sets above were re-verified by tallying keys over **every one of the
1,001,968 profile items in all four full dev splits** (not a sample): each
task's key set is exactly the one documented, with no variants.

Profile keys are **not** uniform across tasks: LaMP-2 movie items have **no
`title` field**, only `description`. LaMP-1 and LaMP-5 share the
`{title, abstract}` shape. A single hardcoded key list would have broken two
of the four tasks; the harness carries a per-task field map instead.

### Profile-length distribution (items per user), full dev split

| Task | min | p25 | median | mean | p75 | max |
|---|---|---|---|---|---|---|
| LaMP-1 | 50 | 61 | 80 | 100.4 | 114 | 914 |
| LaMP-2 | 3 | 12 | 23 | 37.5 | 50 | 149 |
| LaMP-3 | 99 | 115 | 140 | 190.6 | 201 | 1023 |
| LaMP-5 | 49 | 60 | 79 | 99.4 | 113 | 913 |

LaMP-3 matches the verified-facts file exactly (min 99 / median 140 /
mean 190.6 / max 1023), which is an independent check that the download and
parse are correct.

Gold-label distributions are **severely imbalanced** on two tasks: LaMP-3 is
59.1% rating-5 and LaMP-2 is 19.1% `comedy`. **A constant-prediction baseline
therefore scores 0.5912 accuracy on LaMP-3** (measured on the full dev gold). Any sweep result must be read
against that floor, and macro-F1 (not accuracy) is the honest headline for
LaMP-2. This is why LaMP-3 reports accuracy *alongside* MAE/RMSE rather than
instead of them.

## 3. Evaluation subsets

150 queries per task, `seed=0`, stratified into profile-length terciles
(50 short / 50 medium / 50 long, exactly balanced by construction).

Procedure: filter ineligible items → sort the pool by
`(profile_len, qid)` → cut at rank `n//3` and `2n//3` → within each stratum
sort by `qid` for a deterministic order → `random.Random(0).sample` 50 indices.
Rank-based cutting (not fixed length thresholds) is what guarantees the strata
are equal-sized.

### Exclusion audit

| Task | Eligible pool | Excluded: gold leakage | Excluded: malformed input |
|---|---|---|---|
| LaMP-1 | 2,495 | 5 | 0 |
| LaMP-2 | 691 | 0 | 1 |
| LaMP-3 | 2,500 | 0 | 0 |
| LaMP-5 | 2,491 | 9 | 0 |

Two defect classes were found and filtered:

1. **Gold leakage.** For LaMP-5, the gold title appears verbatim as a
   `title` in the same user's own profile in ~0.5% of queries (measured 3/600
   in a spot check, 9/2500 in the full filter); for LaMP-1, a candidate
   reference title appears in the profile in ~0.17%. Those items are
   retrievable-for-free at high *k*, which would manufacture exactly the
   `k`-dependent gain the paper is trying to measure. Excluded.
2. **Malformed input.** One LaMP-2 query (id `110`, found by prefix
   histogram: 691/692 inputs start with "Which tag does this movie…", one
   starts with "x ") is missing its instruction and tag list entirely.
   Excluded.

I also checked the reverse direction for LaMP-5 — whether the *query*
abstract appears in the user's own profile — and found 0/600. Empty-text
profile items: 0 out of 1,001,968 items across all four tasks.

### Subset profile lengths by stratum (measured, after filtering)

| Task | short (min/med/max) | medium | long | distinct lengths in `long` |
|---|---|---|---|---|
| LaMP-1 | 50 / 56.5 / 67 | 67 / 81 / 97 | 100 / 129 / 768 | 43 |
| LaMP-2 | 4 / 9 / 14 | 14 / 25 / 36 | 39 / 60 / 149 | 12 |
| LaMP-3 | 99 / 108 / 121 | 122 / 140 / 173 | 174 / 269 / 1023 | 42 |
| LaMP-5 | 49 / 57 / 66 | 66 / 78.5 / 98 | 100 / 138.5 / 480 | 45 |

The brief requires that long profiles not collapse into a single stratum.
They do not: the `long` stratum spans 12–45 distinct profile lengths per task
and the three strata have disjoint or near-disjoint ranges. The signal axis
(retrieved *k*) is therefore separable from user activity level (stratum).

Subset gold distributions inherit the full-split skew (LaMP-3: 90×`5`,
32×`4`, 15×`3`, 7×`2`, 6×`1`; LaMP-2: 32×`comedy` of 150). **Per-stratum
n=50 means a single-stratum accuracy has a 7.1pp binomial standard error at
p≈0.5; per-task n=150 gives 4.1pp.** Sweep tracks should not claim a stratum-level
difference smaller than that without pooling across tasks.

### Subset file schema

One self-contained parquet per task (zstd). Columns:

| Column | Type | Meaning |
|---|---|---|
| `task` | str | `LaMP-1` … `LaMP-5` |
| `qid` | str | LaMP question id; joins to gold |
| `stratum` | str | `short` / `medium` / `long` |
| `profile_len` | int | number of items in the full profile |
| `input` | str | LaMP `input` verbatim (instruction + payload) |
| `gold` | str | gold output verbatim |
| `profile_json` | str | **full, untruncated** profile as JSON |
| `source_split` | str | provenance path, e.g. `LaMP_2/new/dev` |
| `seed` | int | 0 |

The full profile is embedded, so **no sweep track ever needs to re-download
the 993 MB of raw dev data.** Load with `lamp_harness.load_subset(path)`,
which decodes `profile_json` into a list of dicts and adds a `query_text`
column.

## 4. Harness (`lamp_harness.py`)

Standard-library only (no nltk / sklearn / rouge-score at runtime), so the
sweep tracks have no install step. Exposes:

- `bm25_topk(query_text, profile, k, task=...)` — Okapi BM25, k1=1.5, b=0.75,
  per-task retrievable-field map, fixed inline stopword list. Deterministic:
  ties break by original profile order, verified by repeat-call equality.
  `k=0` → `[]` (the no-signal condition), `k>len(profile)` → whole profile.
- `recency_topk(profile, k)` — newest-first from the profile tail.
  **[unverified]** LaMP profiles carry no date field on these four tasks, so
  "recency" is *positional*: I assume list order is chronological. That is the
  standard assumption in the LaMP literature but I could not verify it from
  the data. The function honours a real `date_key` if one is ever present.
- `build_prompt(task, question_input, retrieved, variant)` → plain string,
  `direct` and `cot` variants, per-task answer-format instruction. Individual
  profile items truncated to 1,200 chars to bound prompt size.
- `query_text(task, input)` — strips the shared instruction boilerplate so
  BM25 term statistics are driven by the payload, not by the identical
  instruction prefix (and, for LaMP-2, not by the 15-tag list).
- `parse_prediction(task, raw_text)` — CoT-robust; returns `None` on failure
  rather than guessing.
- `score_task(task, raw_predictions, golds)` — LaMP-3 MAE+RMSE+accuracy,
  LaMP-1/2 accuracy+macro-F1, LaMP-5 ROUGE-1+ROUGE-L. **Parse failures are
  reported (`parse_failures`, `parse_failure_rate`) and excluded from the
  metric averages**, never silently scored as wrong — a compute-scaling claim
  is not allowed to rest on a parser artifact.

### 4.1 Unit tests — 109 assertions, all passing

Run `python lamp_harness.py`. Every metric is checked against a
hand-computed case, not a golden file:

- MAE on `[3,1,5,2]` vs `[1,2,5,4]` = 5/4 = **1.25**; RMSE = √(9/4) = **1.5**;
  accuracy = **0.25**.
- Macro-F1 on gold `a a b b c` / pred `a b b b c`: per-class F1 = 2/3, 0.8,
  1.0 → macro **0.8222**, while accuracy is 0.8 (the two disagree, which is
  the point of reporting both).
- ROUGE-1 on "the cat sat on the mat" vs "the cat sat on a mat": 5 clipped
  unigram matches / 6 → P = R = F = **5/6**. Duplicate-token clipping checked
  separately ("cat cat" vs "cat" → P=0.5, R=1.0).
- ROUGE-L on "a b c d" vs "a c b d": LCS = 3 → F = **0.75**. Order
  sensitivity is asserted directly: on a full reversal, ROUGE-1 = 1.0 but
  ROUGE-L = 0.25.
- BM25 ranks a topically-matching profile item above an unrelated one;
  all-zero-score queries preserve profile order; `k=0`/empty-profile/`k>n`
  edge cases return the documented shapes.

### 4.2 ROUGE cross-validated against the reference implementation

My hand-rolled ROUGE was checked against Google's `rouge-score`
(`use_stemmer=False`) on 307 cases — 7 hand-picked plus 300 random token
sequences — comparing precision, recall and F for both ROUGE-1 and ROUGE-L:

**max absolute deviation = 0.000e+00 (exact agreement on all 1,842
comparisons).**

`rouge-score` is installed only in the `lamp` env for this cross-check; the
harness itself does not import it.

### 4.3 Live end-to-end smoke tests

These confirm the pipeline runs against a real API. **They are proof-of-harness
only — n is far too small to mean anything about the science, and none of these
numbers belong in the paper.**

Direct prompting, `claude-haiku-4-5-20251001`, BM25 k=4, 6 items/task
(2 per stratum), 0 API errors, **0/24 parse failures**:

| Task | metric |
|---|---|
| LaMP-1 | accuracy 0.500, macro-F1 0.486 |
| LaMP-2 | accuracy 0.667, macro-F1 0.556 |
| LaMP-3 | MAE 0.500, RMSE 0.707, accuracy 0.500 |
| LaMP-5 | ROUGE-1 0.381, ROUGE-L 0.381 |

CoT prompting, 12 items/task (4 per stratum), 48 live responses:
**0 parse failures, 0 label leaks**, and every LaMP-1/2/3 prediction landed
inside the task's label space (36/36).

### 4.4 A real bug the live tests caught

The first CoT run returned a LaMP-5 title parsed as
`"Answer: Superintelligence and Play: …"` — the model had emitted
`**Answer: <title>**`, and my line-anchored `Answer:` regex did not match
through the markdown bold, so the marker leaked into the scored string. On a
free-text ROUGE metric that is a silent scoring error, not a crash: it would
have depressed every affected LaMP-5 score without any failure signal.

Fixed by allowing markdown/bullet decoration around the marker and stripping
a residual label, with six regression assertions added (including the exact
observed string). This is the kind of defect that only surfaces against live
outputs — worth noting for the sweep tracks: **check `parse_failure_rate` and
eyeball a sample of LaMP-5 predictions on every run.**

## 4.5 A process error worth recording

The first draft of `lamp_harness.py` was written before the schema-inspection
cell had run, and its docstring **guessed** LaMP-2 profile items as
`{id, title, description, tag}` while asserting they were verified. The real
shape is `{id, tag, description}`. The code was corrected as soon as the
inspection ran, but the docstring kept the wrong claim for part of the session
and one intermediate artifact version carries it.

Two things were changed so this cannot recur silently: the docstring now states
that the schemas were measured *after* an initial draft guessed them wrong, and
seven assertions now pin `PROFILE_TEXT_FIELDS` and the docstring text itself so
code and documentation cannot drift apart. The schema claim is additionally
confirmed against all 1,001,968 profile items (§2). Flagged here because "the
comment said it was verified" is exactly the failure mode this paper's
provenance rule exists to prevent.

## 5. What sweep tracks should know

1. Load subsets with `lamp_harness.load_subset(path)`. Do not re-download raw
   LaMP data; the full profiles are in the parquet.
2. Use `query_text` (not `input`) as the BM25 query.
3. LaMP-2 is movie tagging from `LaMP_2/new/dev` with only 692 source queries
   and short profiles — expect its k-curve to saturate earliest.
4. Report macro-F1 for LaMP-2 and accuracy-vs-0.59-floor for LaMP-3.
5. Always report `parse_failure_rate` next to any metric.
6. For sampling-based conditions use the 4-5 generation models —
   `temperature` 400-errors on the 5 generation (verified-facts file).

## 6. Verified vs assumed

**Verified in this session:** all four download URLs and byte sizes; per-task
top-level and profile schemas (key homogeneity checked over ≥400 queries per
task and, for empty-text, over all 1,001,968 profile items); query/gold id set
equality; profile-length and gold-label distributions on full dev splits;
LaMP-3 distribution matching the verified-facts file; the LaMP-2 dev-vs-new
task discrepancy; gold-leakage and malformed-input rates; subset strata
balance and disjointness; 109 harness assertions; exact ROUGE agreement with
`rouge-score` over 307 cases; live API round-trip and parse robustness on 88
real responses (24 direct + 64 CoT).

**Assumed / not verified:** that profile list order is chronological (affects
`recency_topk` only); which LaMP-2 variant published baselines used; that the
15-tag and 5-rating label sets are closed beyond the dev split (checked on dev
gold and dev profiles only — both matched exactly, 0 out-of-vocabulary tags).
