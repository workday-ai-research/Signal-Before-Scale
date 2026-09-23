# Scaling analysis: signal volume x inference compute on LaMP

Track: scaling analysis (Phase 2). Every number below was computed in this session from `sweep_all_raw.parquet` using metric primitives imported from `lamp_harness.py`. Nothing is recalled or extrapolated. Bootstraps are percentile bootstraps over items (qids), 2000 resamples, seed 12345. Contrasts between two cells on the same task are **paired**: the resample draws qids once and both cells are re-scored on the same draw.

## 0. Corrections to the task brief, established from the data

Four statements in the brief do not match the file. They are corrected here and the corrected values are used throughout.

| Brief said | Data says |
|---|---|
| 197 api_errors are Sonnet refusals concentrated on LaMP-2/LaMP-5 | 197 non-null `api_error` rows split into **29 refusals** (all `claude-sonnet-4-5`, `variant=direct`; 8 LaMP-1 / 16 LaMP-2 / 4 LaMP-3 / 1 LaMP-5) and **168 usage-limit aborts** (`Claude usage limit reached`), which are an infrastructure loss, not model behaviour |
| Grid B: LaMP-3 k=32 has 39 items, LaMP-2 k=4 has 78 items with 112 rows | **All six grid-B cells have exactly 39 items.** Samples per item are uniform within a cell: 4 for LaMP-3 (all k) and LaMP-2 k=4; 3 for LaMP-2 k=0 and k=32 |
| Grid A is a complete 7 k x 3 model x 4 task grid | **LaMP-5 / haiku / k=4 is empty** -- all 99 calls hit the usage limit. Grid A is 83 of 84 cells |
| n ~ 99 per cell | 99 items per task (33 per profile-length stratum). Sonnet cells lose 1-6 items to refusals; LaMP-2 haiku k=2 has n=90 |

Two further data facts that matter for interpretation:

- The 168 usage-limit aborts also destroyed **grid C for LaMP-3 at k=32** (0 usable rows) and reduced LaMP-3 grid C at k=0/k=4 and LaMP-2 grid C at k=4 to n=60.
- `input_tokens` is not comparable across models in this file: Sonnet runs hit the prompt cache, so its prompt cost appears under `cache_read_input_tokens`. The comparable quantity is `input_tokens + cache_read_input_tokens`, which agrees across models to within 2 tokens.

## 1. Deviation from `score_task`, and the check that it is safe

`score_task` computes LaMP-2 macro-F1 over `sorted(set(gold) | set(pred))`. That denominator changes with the predicted label set, which makes macro-F1 non-comparable across cells and unstable under bootstrap resampling. All macro-F1 here is computed with `lamp_harness.macro_f1(..., labels=LAMP2_TAGS)` -- the closed 15-tag set, which the data confirms is closed (14 tags appear in gold, 15 in predictions, 0 out-of-vocabulary). On the point estimate the two agree (LaMP-2 haiku k=8: 0.3938 both ways).

Every other metric is `score_task`'s own output. Item-level decompositions used for the bootstraps were verified against `score_task` across all 107 cells: **maximum absolute deviation 3.3e-16** across 107 cells and 392 metric comparisons for n, parse_failure_rate, MAE, RMSE, accuracy, ROUGE-1 and ROUGE-L. The vectorised macro-F1 used inside bootstraps was verified against `lamp_harness.macro_f1` on 5 random resamples of every LaMP-1/LaMP-2 cell: maximum deviation 2.2e-16.

## 2. Grid A: the primary surface

Direct prompting, BM25 retrieval, one greedy sample. Headline metric per task: LaMP-1 accuracy, LaMP-2 macro-F1, LaMP-3 MAE, LaMP-5 ROUGE-1 F. Full table with all metrics, n, parse-failure rates and 95% bootstrap CIs: `cell_metrics.csv`.

**LaMP-1** (accuracy, higher = better)

| k | haiku | sonnet | opus |
|---|---|---|---|
| 0 | 0.4444 [0.3434, 0.5455] (n=99) | 0.4124 [0.3093, 0.5155] (n=97) | 0.4242 [0.3232, 0.5152] (n=99) |
| 1 | 0.4848 [0.3838, 0.5859] (n=99) | 0.4490 [0.3469, 0.5408] (n=98) | 0.4646 [0.3636, 0.5657] (n=99) |
| 2 | 0.5152 [0.4141, 0.6162] (n=99) | 0.5204 [0.4184, 0.6224] (n=98) | 0.5354 [0.4343, 0.6263] (n=99) |
| 4 | 0.5051 [0.4040, 0.5962] (n=99) | 0.5918 [0.4997, 0.6837] (n=98) | 0.5960 [0.4949, 0.6869] (n=99) |
| 8 | 0.5657 [0.4646, 0.6667] (n=99) | 0.6429 [0.5510, 0.7449] (n=98) | 0.5859 [0.4848, 0.6768] (n=99) |
| 16 | 0.6224 [0.5204, 0.7172] (n=99) | 0.6122 [0.5204, 0.7041] (n=98) | 0.6162 [0.5152, 0.7172] (n=99) |
| 32 | 0.5960 [0.4949, 0.6970] (n=99) | 0.6531 [0.5612, 0.7449] (n=98) | 0.5657 [0.4646, 0.6566] (n=99) |

**LaMP-2** (macro_f1, higher = better)

| k | haiku | sonnet | opus |
|---|---|---|---|
| 0 | 0.4200 [0.2833, 0.4992] (n=99) | 0.3048 [0.2175, 0.3664] (n=94) | 0.3208 [0.2351, 0.3879] (n=99) |
| 1 | 0.3572 [0.2435, 0.4300] (n=99) | 0.3714 [0.2694, 0.4405] (n=95) | 0.3456 [0.2517, 0.4119] (n=99) |
| 2 | 0.3814 [0.2654, 0.4613] (n=90) | 0.4107 [0.3014, 0.4862] (n=99) | 0.3966 [0.2958, 0.4744] (n=99) |
| 4 | 0.3792 [0.2770, 0.4465] (n=99) | 0.4358 [0.3233, 0.5084] (n=93) | 0.4388 [0.3254, 0.5123] (n=99) |
| 8 | 0.3938 [0.2841, 0.4721] (n=99) | 0.4665 [0.3483, 0.5359] (n=99) | 0.4724 [0.3564, 0.5458] (n=99) |
| 16 | 0.4223 [0.3094, 0.5006] (n=99) | 0.4756 [0.3600, 0.5450] (n=99) | 0.4389 [0.3309, 0.5085] (n=99) |
| 32 | 0.4037 [0.2870, 0.4778] (n=99) | 0.4797 [0.3645, 0.5544] (n=98) | 0.4613 [0.3441, 0.5392] (n=99) |

**LaMP-3** (mae, lower = better)

| k | haiku | sonnet | opus |
|---|---|---|---|
| 0 | 0.4545 [0.3333, 0.5758] (n=99) | 0.2653 [0.1735, 0.3574] (n=98) | 0.2828 [0.1919, 0.3838] (n=99) |
| 1 | 0.2727 [0.1818, 0.3737] (n=99) | 0.2121 [0.1313, 0.2932] (n=99) | 0.2323 [0.1414, 0.3333] (n=99) |
| 2 | 0.2626 [0.1717, 0.3737] (n=99) | 0.1856 [0.1031, 0.2784] (n=97) | 0.1919 [0.1111, 0.2929] (n=99) |
| 4 | 0.2020 [0.1111, 0.3030] (n=99) | 0.1414 [0.0707, 0.2222] (n=99) | 0.1414 [0.0707, 0.2222] (n=99) |
| 8 | 0.1414 [0.0707, 0.2222] (n=99) | 0.1111 [0.0505, 0.1818] (n=99) | 0.1313 [0.0606, 0.2121] (n=99) |
| 16 | 0.1919 [0.1111, 0.2929] (n=99) | 0.1212 [0.0606, 0.1919] (n=99) | 0.1515 [0.0806, 0.2323] (n=99) |
| 32 | 0.1414 [0.0707, 0.2222] (n=99) | 0.1224 [0.0612, 0.1939] (n=98) | 0.1515 [0.0808, 0.2323] (n=99) |

**LaMP-5** (rouge_1, higher = better)

| k | haiku | sonnet | opus |
|---|---|---|---|
| 0 | 0.3919 [0.3538, 0.4323] (n=99) | 0.4268 [0.3801, 0.4709] (n=99) | 0.4449 [0.3966, 0.4936] (n=99) |
| 1 | 0.4164 [0.3713, 0.4607] (n=99) | 0.4329 [0.3880, 0.4783] (n=99) | 0.4441 [0.3954, 0.4935] (n=99) |
| 2 | 0.4256 [0.3815, 0.4681] (n=99) | 0.4496 [0.4030, 0.4961] (n=99) | 0.4592 [0.4121, 0.5073] (n=99) |
| 4 | -- | 0.4774 [0.4281, 0.5243] (n=99) | 0.4581 [0.4115, 0.5051] (n=99) |
| 8 | 0.4275 [0.3811, 0.4717] (n=99) | 0.4449 [0.3946, 0.4925] (n=99) | 0.4485 [0.3976, 0.4963] (n=99) |
| 16 | 0.4226 [0.3785, 0.4656] (n=99) | 0.4546 [0.4062, 0.5037] (n=99) | 0.4321 [0.3840, 0.4821] (n=99) |
| 32 | 0.4175 [0.3687, 0.4646] (n=99) | 0.4721 [0.4231, 0.5199] (n=98) | 0.4396 [0.3933, 0.4870] (n=99) |

Parse failures across all 106 cells: **14 unparseable responses of 11,197 usable calls (0.13%)**, 13 on LaMP-1 (6 of them in the recency k=32 cell, rate 6.1%) and 1 on LaMP-5. Parse failures are excluded from metric averages, per the harness contract, and are reported per cell in `cell_metrics.csv`.

## 3. Saturation in the signal axis

### 3.1 Non-parametric knee (the reported estimate)

The knee is the smallest k at which the running maximum of the quality curve reaches 90% of the total k=0 -> k=32 gain. The running maximum is used so a single non-monotone dip cannot push the knee rightward. Curves are pooled over the three model tiers (mean quality), and the bootstrap resamples the items shared by all 21 cells of that task.

| Task | n | total gain, k=0 -> 32 | knee (90%) | fraction of gain by k=4 | by k=8 |
|---|---|---|---|---|---|
| LaMP-1 | 97 | 0.1821 [0.1100, 0.2543] | 8 [4, 16] | 0.774 [0.475, 1.098] | 0.962 [0.759, 1.238] |
| LaMP-2 | 77 | 0.1128 [0.0335, 0.1730] | 8 [1, 32] | 0.609 [-0.127, 1.464] | 1.003 [0.693, 1.881] |
| LaMP-3 | 95 | 0.2105 [0.1404, 0.2842] | 8 [4, 8] | 0.883 [0.660, 1.115] | 1.067 [0.910, 1.255] |
| LaMP-5 | 98 | 0.0207 [-0.0096, 0.0528] | 2 [1, 32] | -- | 0.833 [-1.618, 4.063] |

Readout. On LaMP-1, LaMP-2 and LaMP-3 the total gain from retrieving user history is resolvable and the knee sits at **k=8**; the LaMP-3 knee CI is tight ([4, 8]) while LaMP-1 ([4, 16]) and LaMP-2 ([1, 32]) are not. On **LaMP-5 the total gain is not resolvable** (0.0207 [-0.0096, 0.0528] ROUGE-1), so its knee is meaningless and is reported as such. Per-model knees are in `fits.json` under `nonparametric_knee.per_model`; at n~99 most per-model knee CIs span [2, 32] and should not be quoted individually.

### 3.2 Parametric fits, and what they do and do not identify

Six forms were fitted to the 21 cell errors per task (error = MAE on LaMP-3, 1 - metric elsewhere) and compared by **paired repeated 5-fold cross-validation over items** (10 repeats = 50 folds; the same folds for every form, so the comparison is paired). `sat_per_model` is `err(k) = e_inf(m) + a(m)(k+1)^-b(m)`; `sat_shared` removes the compute effect entirely; `flat_per_model` removes the signal effect; `sat_shared_floor` gives all three tiers one common e_inf.

| Task | drop the signal term | drop the compute term | one shared floor |
|---|---|---|---|
| LaMP-1 | +3.69e-3 (t=+6.3) | -0.58e-3 (t=-5.1) | -0.05e-3 (t=-1.2) |
| LaMP-2 | -0.35e-3 (t=-1.4) | -0.56e-3 (t=-4.2) | -0.08e-3 (t=-4.5) |
| LaMP-3 | +5.01e-3 (t=+11.3) | +0.92e-3 (t=+3.8) | -0.15e-3 (t=-4.4) |
| LaMP-5 | -0.02e-3 (t=-0.6) | +0.03e-3 (t=+0.7) | +0.03e-3 (t=+1.9) |

Values are the change in held-out MSE relative to the full `sat_per_model` fit; **positive means the restricted form predicts worse, i.e. the dropped term was needed**. t is the paired t across the 50 folds.

- **The signal term is needed on LaMP-1 (t=+6.3) and LaMP-3 (t=+11.3)**; on LaMP-2 (t=-1.4) and LaMP-5 (t=-0.6) held-out prediction does not need it.
- **The compute term is needed only on LaMP-3 (t=+3.8).** On LaMP-1 (t=-5.1), LaMP-2 (t=-4.2) and LaMP-5 (t=+0.7) dropping the compute effect leaves prediction the same or better.
- **Forcing one irreducible-error floor shared by all three tiers improves held-out prediction** on LaMP-3 (t=-4.4) and LaMP-2 (t=-4.5), and is neutral on LaMP-1 (t=-1.2) and LaMP-5 (t=+1.9). Compute changes how fast the floor is approached, not where it is.

LaMP-3 shared-floor fit (n=95 items, in-sample RMSE 0.0194 MAE):

- irreducible error `e_inf` = **0.1206 [0.0323, 0.1808] MAE**
- haiku: a = 0.348 [0.258, 0.470], b = 0.900 [0.406, 1.627]
- sonnet: a = 0.162 [0.095, 0.265], b = 0.979 [0.326, 2.256]
- opus: a = 0.181 [0.117, 0.277], b = 0.905 [0.283, 1.814]

The three exponents b are statistically indistinguishable (all CIs overlap, all include ~0.9), and only the amplitude a separates the tiers -- haiku 0.348 versus sonnet 0.162 and opus 0.181. Read plainly: **the tiers differ in how far they start from the floor, not in the rate at which retrieved history closes the gap, and not in where the floor is.**

**What the fits do not support.** The free-floor `sat_per_model` fit is under-identified on LaMP-1, LaMP-2 and LaMP-5: e_inf CIs span [0.00, 0.47] (LaMP-1 haiku) to [0.00, 0.72] (LaMP-2 haiku) and b CIs reach the [0.01, 5.0] bounds. **We cannot claim a clean power law.** The exponential form `e_inf + a exp(-k/tau)` is indistinguishable from the power form on LaMP-3 (t=+0.06) and LaMP-1 (t=-2.0, favouring exponential by 1e-4). Consistent with the broken-neural-scaling-law literature, our data resolve *that* the curve saturates and *where*, not the algebraic form. Quote the knee, not the exponent, except for the LaMP-3 shared-floor e_inf which is resolvable.

## 4. The cross-term: is compute a substitute for signal?

### 4.1 Cheap model with signal versus expensive model without

Paired contrast, haiku at its best k against opus at k=0, on the qids present in both cells.

| Task | metric | haiku best k | advantage of haiku+signal | n | p (boot) |
|---|---|---|---|---|---|
| LaMP-1 | accuracy | 16 | 0.1982 [0.1010, 0.2993] | 99 | 0.0005 |
| LaMP-2 | macro_f1 | 16 | 0.1015 [0.0212, 0.1708] | 99 | 0.0060 |
| LaMP-3 | mae | 8 | 0.1414 [0.0606, 0.2222] | 99 | 0.0005 |
| LaMP-5 | rouge_1 | 8 | -0.0174 [-0.0646, 0.0281] | 99 | 0.4650 |

**On 3 of 4 tasks the small model with retrieved history significantly beats the largest model with none. On LaMP-5 it does not** (-0.017 ROUGE-1, CI includes 0), and on LaMP-5 the k-axis carries no resolvable signal at all, so there is nothing for the small model to exploit. This asymmetry is the honest scope limit on the claim: it holds where retrieved history moves the metric, and free-text title generation is a task where it does not.

The sharpest single number, LaMP-3: haiku at k=8 reaches MAE 0.1414 [0.0707, 0.2222] against opus at k=0 at 0.2828 [0.1919, 0.3838]; the paired difference is **-0.1414 MAE [-0.2222, -0.0606], n=99, p=0.0005** -- the cheap model with eight of the user's own reviews halves the error of the largest model given none.

### 4.2 Convergence of the model tiers

Spread = best minus worst model tier at fixed k, in quality units, paired bootstrap:

| Task | spread at k=0 | spread at k=32 |
|---|---|---|
| LaMP-1 | 0.0206 [0.0103, 0.1134] | 0.0825 [0.0206, 0.1753] |
| LaMP-2 | 0.0942 [0.0161, 0.1763] | 0.0825 [0.0322, 0.1667] |
| LaMP-3 | 0.2000 [0.1158, 0.3053] | 0.0211 [0.0105, 0.0947] |
| LaMP-5 | 0.0502 [0.0196, 0.0878] | 0.0533 [0.0214, 0.0922] |

**Convergence to a common floor is a LaMP-3 result**: the tier spread falls from 0.2000 [0.1158, 0.3053] MAE at k=0 to 0.0211 [0.0105, 0.0947] at k=32, a factor of 9.5. On LaMP-1 the spread is small at every k (never above 0.093 accuracy), so there is nothing to converge. **On LaMP-2 and LaMP-5 the spread does not shrink** (0.094 -> 0.082 macro-F1; 0.050 -> 0.053 ROUGE-1). The paper may not claim that the axes converge in general.

### 4.3 The interaction, tested

Difference-in-differences on the shared qids of four cells: `[M(large,k) - M(large,0)] - [M(small,k) - M(small,0)]`. Full table in `contrasts.csv` (`family=interaction_DiD`); 17 of 46 have a CI excluding zero.

| Task | pair | k | DiD on the headline metric | reading |
|---|---|---|---|---|
| LaMP-3 | opus-haiku | 32 | 0.1818 [0.0606, 0.3232] | compute advantage SHRINKS with signal |
| LaMP-3 | sonnet-haiku | 32 | 0.1856 [0.0825, 0.2990] | compute advantage SHRINKS with signal |
| LaMP-2 | opus-haiku | 32 | 0.1568 [0.0224, 0.2490] | compute advantage GROWS with signal |
| LaMP-2 | sonnet-haiku | 32 | 0.1668 [0.0486, 0.2522] | compute advantage GROWS with signal |
| LaMP-1 | opus-haiku | 32 | -0.0101 [-0.1515, 0.1111] | not resolvable |
| LaMP-5 | opus-haiku | 16 | -0.0435 [-0.0871, -0.0018] | compute advantage SHRINKS with signal |

Sign convention. The metric-space DiD must be converted to quality units before it can be read as substitution: `Delta(compute advantage) = s * DiD` with s = -1 for MAE and +1 otherwise. Doing so:

- **LaMP-3: substitution.** The opus-over-haiku advantage falls by 0.182 MAE [0.061, 0.323] going from k=0 to k=32 (p=0.003); same for sonnet-over-haiku (0.186 [0.083, 0.299], p=0.0005). Resolvable at every k in {4, 8, 16, 32}.
- **LaMP-2: complementarity, the opposite of the thesis.** The opus-over-haiku advantage *grows* by 0.157 macro-F1 [0.022, 0.249] (p=0.016), resolvable at every k. This is not a macro-F1 artefact: on accuracy, opus-minus-haiku goes from -0.010 at k=0 to +0.121 at k=32. The larger models extract more from 32 retrieved movie tags than haiku does.
- **LaMP-1: no resolvable interaction** (DiD -0.010 [-0.152, +0.111] at k=32); the compute axis has no measurable effect at any k, so there is no interaction to find.
- **LaMP-5: weak substitution, one resolvable cell** (opus-haiku at k=16, -0.044 [-0.087, -0.002]); all effects under 0.05 ROUGE-1.

**The interaction is therefore task-dependent and the paper must say so.** The claim 'compute is a poor substitute for signal' is supported on LaMP-3, contradicted on LaMP-2, and untestable on LaMP-1/LaMP-5 for want of a compute effect.

### 4.4 Exchange rate

The available compute ladder has two model-scale steps (haiku -> sonnet -> opus). The exchange rate is the signal-axis span at fixed model divided by the per-step compute gain at k=0.

| Task | signal span, haiku k=0->32 | compute span at k=0 (2 steps) | compute span at k=32 | model-scale steps to match the signal span |
|---|---|---|---|---|
| LaMP-1 | 0.1515 | -0.0202 | -0.0303 | unreachable -- compute span at k=0 is <= 0 |
| LaMP-2 | -0.0163 | -0.0992 | 0.0576 | unreachable -- compute span at k=0 is <= 0 |
| LaMP-3 | 0.3131 | 0.1717 | -0.0101 | 3.65 |
| LaMP-5 | 0.0256 | 0.0530 | 0.0221 | 0.96 |

**LaMP-3 is the quantitative statement.** Matching the k=0 -> 32 signal gain (0.3131 MAE) would take **3.65 model-scale steps**, on a ladder that has 2. The compute axis cannot reach what the signal axis reaches, and the shortfall is not marginal -- it is 1.8x the entire ladder. On LaMP-1 and LaMP-2 the compute span at k=0 is <= 0, so **no** number of model-scale steps matches the signal gain. On LaMP-5 the exchange rate is ~1.0, i.e. the axes are interchangeable there, but both spans are under 0.06 ROUGE-1 and neither is resolvable.

### 4.5 Resolvable-effect rate, at equal n

A blunt summary that needs no functional form. Of the paired contrasts computed at n~99 with identical bootstrap settings:

| Task | signal contrasts (k vs k=0) resolved | compute contrasts (tier vs tier at fixed k) resolved |
|---|---|---|
| LaMP-1 | 13 / 18 | 0 / 21 |
| LaMP-2 | 8 / 18 | 3 / 21 |
| LaMP-3 | 15 / 18 | 4 / 21 |
| LaMP-5 | 3 / 17 | 3 / 19 |
| **all** | **39 / 71** | **10 / 82** |

At the same sample size and the same test, the signal axis produces a resolvable effect **39 times in 71 attempts (55%)** and the compute axis **10 times in 82 (12%)**.


## 5. The decision rule (contribution C2)

### 5.1 Form

The observable at inference time, before any model call, is `effective_k = min(k, profile_len)` -- how many of the user's own items actually reach the context. The expected quality gain from upgrading the model tier is fitted directly as a decaying function of it:

```
g(k) = c * (k + 1)^(-d)            [expected reduction in per-item task loss from haiku -> opus]
upgrade  iff  effective_k <= k*(tau),   k*(tau) = (c/tau)^(1/d) - 1
```

`tau` is the smallest quality gain that justifies the upgrade -- the operator's exchange rate between quality and cost. We do not have per-token prices in this session, so `tau` is left as the single free parameter rather than converted to money.

Fitted on all items:

| Task | c | d | k*(tau=0.02) | k*(tau=0.05) | observed per-k gains |
|---|---|---|---|---|---|
| LaMP-1 | 0.0119 | 0.1294 | -1.0 | -1.0 | -0.010, -0.010, 0.010, 0.093, 0.021, -0.006, -0.031 |
| LaMP-2 | 0.0600 | 0.0100 | 520183664532889814277077232517739607023096430592.0 | 83590291.1 | 0.000, 0.026, 0.052, 0.078, 0.078, 0.065, 0.117 |
| LaMP-3 | 0.1668 | 0.8942 | 9.7 | 2.8 | 0.179, 0.042, 0.084, 0.063, 0.011, 0.042, -0.011 |
| LaMP-5 | 0.0444 | 0.4407 | 5.1 | -1.0 | 0.050, 0.022, 0.030, 0.016, 0.006, 0.022 |

### 5.2 Held-out validation

20 random 50/50 splits of users (qids), seeds 1000-1019. `c, d` are fitted on the training half only; every loss is evaluated on the held-out half. Baselines: **oracle** picks, at each k, the arm with the lower held-out loss; **random-matched** upgrades the same *number* of k buckets chosen uniformly at random (200 draws per split). Losses are per-item task losses (LaMP-3 |predicted - gold rating|; others 1 - metric, with LaMP-2 on 0/1 error because macro-F1 is not item-decomposable).

LaMP-3, the only task where the rule carries information:

| tau | k* (median over splits) | upgrade rate | loss under the rule | regret vs oracle | regret of random at matched budget |
|---|---|---|---|---|---|
| 0.00 | infinite | 100% | 0.1982 | 0.0057 | 0.0057 |
| 0.01 | 35.7 | 82% | 0.2045 | 0.0119 | 0.0182 |
| 0.02 | 13.8 | 71% | 0.2088 | 0.0162 | 0.0245 |
| 0.05 | 3.7 | 49% | 0.2152 | 0.0226 | 0.0356 |
| 0.10 | 0.8 | 22% | 0.2263 | 0.0338 | 0.0487 |

For reference on the same splits: always-upgrade loss 0.1982, never-upgrade 0.2516, oracle 0.1926.

**Readout, stated against the rule rather than for it.**

1. The rule beats a matched-budget random policy at every tau > 0 on LaMP-3 (0.0119 vs 0.0182 at tau=0.01; 0.0226 vs 0.0356 at tau=0.05), so **the ordering it induces over k is real information, not an artefact of how many upgrades it happens to buy**.
2. **It does not beat always-upgrading on quality.** Always-upgrade has regret 0.0057; the rule at tau=0.02 has 0.0162. As a quality-maximising policy the rule is strictly worse, and the paper must not present it otherwise.
3. **Its value is the compute it declines.** At tau=0.05 on LaMP-3 the rule declines the upgrade on 51% of the k range and pays 0.0170 MAE for it (0.2152 versus 0.1982). The operating point worth quoting: with c=0.167 and d=0.894, requiring a 0.05 MAE improvement to justify the upgrade means **upgrade only when fewer than about 3 of the user's own items are in context (k* = 2.8; median 3.7 across held-out splits)**; requiring 0.02 MAE moves the threshold to k* = 9.7 (median 13.8).
4. **On the other three tasks the rule is not usable.** On LaMP-2 the fitted gain is flat in k (d = 0.010) because the true gain *grows* with k, so k* diverges and the rule degenerates to always-upgrade with regret identical to random. On LaMP-1 the per-k gains are noise (-0.010 to +0.093) and the rule is no better than random. On LaMP-5 it edges random (0.0099 vs 0.0116 at tau=0.01) but every effect is under 0.05 ROUGE-1.

C2 as the data supports it: **on rating prediction, a threshold on observable signal volume identifies when a model-tier upgrade is worth buying, with an out-of-sample regret of 0.016 MAE at tau=0.02 against a clairvoyant oracle and a 30% reduction in upgrades. The rule does not generalise to the other three tasks in this suite, and it is a cost-saving instrument, not a quality-improving one.**

## 6. Mechanism: answer determinism (grid B)

CoT prompting, temperature 1.0, haiku-4-5. **Every cell is exactly 39 items** with a uniform sample count per item; the 39-item LaMP-3 subset is stratum-balanced (13 short / 13 medium / 13 long) and its direct-prompt MAE tracks the full 99 items (k=4: 0.2051 vs 0.2020; k=32: 0.1026 vs 0.1414; k=0 is harder, 0.5385 vs 0.4545). Entropy is normalised by log(number of parsed samples for that item), so 1.0 means every sample differed. Bootstrap CIs are over the 39 items.

| Task | k | items | samples/item | fully deterministic | mean normalised entropy | max distinct answers |
|---|---|---|---|---|---|---|
| LaMP-2 | 0 | 39 | 3 | 0.590 [0.436, 0.744] | 0.270 [0.163, 0.377] | 3 |
| LaMP-2 | 4 | 39 | 4 | 0.564 [0.410, 0.718] | 0.222 [0.137, 0.308] | 3 |
| LaMP-2 | 32 | 39 | 3 | 0.385 [0.231, 0.538] | 0.421 [0.315, 0.540] | 3 |
| LaMP-3 | 0 | 39 | 4 | 0.667 [0.513, 0.821] | 0.142 [0.078, 0.204] | 2 |
| LaMP-3 | 4 | 39 | 4 | 0.513 [0.359, 0.667] | 0.247 [0.163, 0.329] | 3 |
| LaMP-3 | 32 | 39 | 4 | 0.513 [0.359, 0.667] | 0.210 [0.145, 0.279] | 2 |

- **38% to 67% of items return an identical answer on every sample**, and no item ever produced more than 3 distinct answers out of 3-4 draws. Mean normalised entropy runs 0.14 to 0.42, i.e. 58% to 86% below the ceiling.
- **Self-consistency has almost no headroom at low k.** Majority vote over 3-4 samples versus the mean single sample: LaMP-3 k=0 -0.006 MAE [-0.071, +0.058], k=4 0.000 [-0.096, +0.083]; LaMP-2 k=0 +0.009 accuracy [-0.051, +0.068], k=4 +0.013 [-0.051, +0.083]. **None resolvable.** It becomes resolvable only at k=32 (LaMP-3 +0.250 MAE [0.096, 0.423]; LaMP-2 +0.086 accuracy [0.017, 0.162]) -- extra sampling needs signal before it pays, which is the same asymmetry again rather than a counterexample.
- **The k=32 sampling gain is recovery, not progress.** On the same 39 items, 4-sample CoT majority vote at k=32 reaches MAE 0.2564 while **one greedy direct call reaches 0.1026** -- so four times the sampling budget under CoT ends up 2.5x worse than a single deterministic call. Table: `sampling_vs_direct.csv`.
- **Confidently wrong.** Among items the majority vote got wrong, the fraction that were nonetheless fully deterministic is 0.522 [0.348, 0.739] (LaMP-2 k=0), 0.556 [0.333, 0.778] (LaMP-3 k=0), 0.333 [0.108, 0.667] (LaMP-3 k=32). Determinism is lower among wrong answers than right ones in all six cells (e.g. LaMP-3 k=4: 0.083 wrong vs 0.704 correct), so the signal is not useless as a confidence proxy -- but at k=0 roughly half of the errors carry no disagreement at all for a self-consistency method to exploit. **n per group is 9 to 23 items; these CIs are wide and the LaMP-3 k=32 wrong group is 9 items.**

## 7. Controls

### 7.1 Chain-of-thought versus direct, matched k (grid C, haiku)

| Task | k | n | quality change from CoT | resolvable |
|---|---|---|---|---|
| LaMP-1 | 0 | 99 | -0.0404 [-0.1313, 0.0505] | no |
| LaMP-1 | 4 | 99 | 0.0664 [-0.0461, 0.1842] | no |
| LaMP-1 | 32 | 99 | -0.0083 [-0.1408, 0.1253] | no |
| LaMP-2 | 0 | 99 | -0.0816 [-0.1524, 0.0238] | no |
| LaMP-2 | 4 | 99 | 0.0549 [-0.0310, 0.1312] | no |
| LaMP-2 | 32 | 99 | -0.0816 [-0.1529, -0.0033] | yes |
| LaMP-3 | 0 | 99 | 0.0404 [-0.0404, 0.1212] | no |
| LaMP-3 | 4 | 99 | -0.1212 [-0.2323, -0.0000] | no |
| LaMP-3 | 32 | 39 | -0.4872 [-0.8462, -0.1795] | yes |
| LaMP-5 | 0 | 99 | 0.0649 [0.0340, 0.0967] | yes |
| LaMP-5 | 32 | 99 | -0.1646 [-0.2158, -0.1122] | yes |

Four of eleven contrasts resolve: **three are harms** (LaMP-2 k=32 -0.082 macro-F1; LaMP-3 k=32 -0.487 MAE, n=39; LaMP-5 k=32 -0.165 ROUGE-1) and **one is a gain** (LaMP-5 k=0 +0.065 ROUGE-1). The reasoning-token axis buys nothing at matched k on these tasks and costs materially at k=32. Note the LaMP-3 k=32 CoT cell is n=39 (the usage-limit abort destroyed the n=99 single-sample run) and is the temperature-1.0 condition, so it conflates CoT with sampling temperature; the LaMP-2 and LaMP-5 k=32 harms are at n=99 and greedy.

### 7.2 BM25 versus recency, matched k (grid R, haiku)

**No contrast at any k on either task has a CI excluding zero** (12 of 12; see `controls.csv`). The largest point estimate is -0.061 macro-F1 (LaMP-2 k=16, CI [-0.127, +0.013]).

The ceiling question -- does the best attainable quality move with the retriever? Paired bootstrap of `max_k q(recency) - max_k q(bm25)`:

| Task | n | BM25 ceiling | recency ceiling | difference |
|---|---|---|---|---|
| LaMP-1 | 99 | 0.6224 | 0.6237 | 0.0012 [-0.0993, 0.0874] |
| LaMP-2 | 90 | 0.4099 | 0.3883 | -0.0215 [-0.0803, 0.0406] |

**The ceiling does not move detectably with the retriever.** This is the control that lets the paper describe the bound as an information bound rather than a BM25 artefact -- but the claim must be scoped to the resolution actually achieved. Limitations, all of which belong in the paper:

- The CI half-widths are +-0.09 (LaMP-1) and +-0.06 (LaMP-2) in quality units, so **a ceiling shift smaller than about 6-9 points cannot be ruled out.**
- Only 2 of 4 tasks and only the haiku tier were run with a second retriever.
- **Recency is positional, not chronological.** LaMP profiles for these tasks carry no date field; `recency_topk` takes the profile tail and assumes list order is chronological. The data track flagged this as unverified. A genuinely different retrieval *objective* (e.g. a trained or diversity-aware retriever) has not been tested and could move the ceiling.

## 8. Required auxiliary readouts

### 8.1 LaMP-3 accuracy against the constant-prediction floor

The majority class on our 99-item LaMP-3 subset is rating 5 at **0.6364**; the data track measured **0.5912** on the full 2,500-item dev split. Per-stratum floors on our subset: short 0.6970, medium 0.6667, long 0.5455 (the rating track's quoted 0.660 short / 0.600 medium refer to a different subset and are not reproduced here).

| k | haiku | sonnet | opus |
|---|---|---|---|
| 0 | 0.6061 | 0.7551 | 0.7273 |
| 1 | 0.7475 | 0.7980 | 0.7980 |
| 2 | 0.7677 | 0.8351 | 0.8384 |
| 4 | 0.8283 | 0.8687 | 0.8687 |
| 8 | 0.8687 | 0.8889 | 0.8788 |
| 16 | 0.8384 | 0.8788 | 0.8687 |
| 32 | 0.8687 | 0.8776 | 0.8687 |

**20 of 21 cells clear the subset floor of 0.6364. The single exception is haiku at k=0 (0.6061), which is below the constant-prediction baseline** -- with no user signal the small model is worse than always answering '5'. With 8 retrieved reviews it reaches 0.8687. Every cell clears the 0.5912 full-split floor.

### 8.2 effective_k versus k

| k | LaMP-1 | LaMP-2 | LaMP-3 | LaMP-5 |
|---|---|---|---|---|
| 0 | 0.00 (0% capped) | 0.00 (0% capped) | 0.00 (0% capped) | 0.00 (0% capped) |
| 1 | 1.00 (0% capped) | 1.00 (0% capped) | 1.00 (0% capped) | 1.00 (0% capped) |
| 2 | 2.00 (0% capped) | 2.00 (0% capped) | 2.00 (0% capped) | 2.00 (0% capped) |
| 4 | 4.00 (0% capped) | 4.00 (0% capped) | 4.00 (0% capped) | -- |
| 8 | 8.00 (0% capped) | 7.84 (8% capped) | 8.00 (0% capped) | 8.00 (0% capped) |
| 16 | 16.00 (0% capped) | 13.86 (38% capped) | 16.00 (0% capped) | 16.00 (0% capped) |
| 32 | 32.00 (0% capped) | 22.11 (60% capped) | 32.00 (0% capped) | 32.00 (0% capped) |

**Only LaMP-2 saturates the signal axis from profile length.** At k=32, 60% of LaMP-2 users have shorter profiles than the budget and the mean effective_k is 22.1; at k=16, 38% are capped. LaMP-1/3/5 profiles (median 80 / 140 / 79 items) never bind. Two consequences: LaMP-2's flat high-k behaviour is partly a *ceiling on available signal*, not only on the model's use of it; and the decision rule is correctly keyed on effective_k rather than k, because on LaMP-2 the two differ.

### 8.3 Refusals and other measurement losses

- **29 refusals**, all `claude-sonnet-4-5-20250929` with `variant=direct` and `stop_reason="refusal"`: 8 LaMP-1, 16 LaMP-2, 4 LaMP-3, 1 LaMP-5. They are not uniform in k (LaMP-2: 5 at k=0, 4 at k=1, 6 at k=4, 1 at k=32), so Sonnet cell sizes vary between 93 and 99 and **Sonnet's LaMP-2 numbers rest on a mildly non-random subsample.** Direction of any resulting bias is unknown; we do not correct for it.
- **168 usage-limit aborts**, an infrastructure loss with no model content: they removed LaMP-5/haiku/k=4 from grid A entirely, removed LaMP-3/k=32 from grid C entirely, and cut three grid-C cells to n=60.
- **14 unparseable responses of 11,197 (0.13%)**, concentrated in LaMP-1 haiku (13, of which 6 in the recency k=32 cell = 6.1% of that cell). Excluded from metric averages per the harness contract, reported per cell.

### 8.4 Per-stratum signal gains (exploratory)

k=0 -> k=32 gain by profile-length stratum, averaged over the three tiers (n=33 per task/stratum/tier, so single-stratum CIs are ~+-0.15; `stratum_gains.csv` has them):

| Task | short | medium | long |
|---|---|---|---|
| LaMP-1 | 0.073 | 0.131 | 0.323 |
| LaMP-2 | 0.088 | 0.116 | 0.056 |
| LaMP-3 | 0.286 | 0.142 | 0.162 |
| LaMP-5 | 0.051 | -0.004 | 0.017 |

The direction is inconsistent across tasks -- LaMP-1 gains most for long profiles (0.323 vs 0.073 short), LaMP-3 most for short ones (0.286 vs 0.162 long). **At n=33 per cell this is not a result and should not appear as one in the paper.** Report it as exploratory or omit.


## 9. Verdict against the required framing

The framing the paper must serve: *under sparse per-user signal, quality is bounded by available user information, and inference compute is a poor substitute for it.* Component by component:

| Component | Verdict | Evidence |
|---|---|---|
| Retrieved user signal moves quality | **Supported on 3 of 4 tasks** | Pooled k=0->32 gain resolvable on LaMP-1 (0.182 accuracy), LaMP-2 (0.113 macro-F1), LaMP-3 (0.211 MAE); **not** on LaMP-5 (0.021 ROUGE-1, CI includes 0) |
| The signal axis saturates | **Supported** | Knee at k=8 on LaMP-1/2/3; >=83% of the total gain realised by k=8 on all four tasks; the signal term is required for held-out prediction on LaMP-1 and LaMP-3 |
| A cheap model with signal beats an expensive model without | **Supported on 3 of 4 tasks** | LaMP-1 +0.198, LaMP-2 +0.102, LaMP-3 +0.141, all CIs excluding 0; LaMP-5 -0.017, CI includes 0 |
| The compute axis cannot reach what the signal axis reaches | **Supported where a compute effect exists** | LaMP-3 needs 3.65 model-scale steps on a 2-step ladder; LaMP-1/LaMP-2 have a non-positive compute span at k=0, so no number of steps suffices |
| The axes converge to a common floor | **Supported on LaMP-3 only** | Tier spread 0.200 -> 0.021 MAE; a single shared floor improves held-out prediction (t=-4.4), e_inf = 0.121 [0.032, 0.181] MAE. **Spread does not shrink on LaMP-2 or LaMP-5** |
| The marginal value of compute shrinks as signal grows (the cross-term) | **MIXED -- supported on LaMP-3, contradicted on LaMP-2** | LaMP-3 DiD -0.182 MAE of compute advantage lost (p=0.003); LaMP-2 DiD +0.157 macro-F1 of compute advantage *gained* (p=0.016) |
| Self-consistency has little headroom | **Supported at low k** | 38-67% of items fully deterministic; majority-vote gain not resolvable at k=0 or k=4 on either task |
| The bound is informational, not a retrieval artefact | **Consistent with the data, at limited resolution** | No resolvable retriever effect in 12 of 12 matched-k contrasts; ceiling difference 0.001 [-0.099, 0.087] (LaMP-1) and -0.022 [-0.080, 0.041] (LaMP-2) |
| A decision rule keyed on observable signal decides when compute pays | **Supported on LaMP-3 only** | Beats matched-budget random at every tau; useless on LaMP-1/2, marginal on LaMP-5 |

### 9.1 What the data does NOT support -- do not write these sentences

1. **'Inference compute is a poor substitute for user signal.'** Not as a general claim. It is true on LaMP-3 and vacuous on LaMP-1 (no compute effect to substitute) and it is *false* on LaMP-2, where signal and model scale are complementary with a resolvable positive interaction. Write: *'On rating prediction the marginal value of model scale collapses once user history is in context; on movie tagging the two are complementary. The direction of the interaction is task-dependent.'*
2. **'The signal and compute axes converge to a common floor.'** LaMP-3 only. The tier spread is flat on LaMP-2 (0.094 -> 0.082) and LaMP-5 (0.050 -> 0.053).
3. **'We fit a power law with exponent b.'** The per-model 3-parameter fits are under-identified on 3 of 4 tasks (e_inf CIs reaching [0.00, 0.72]; b CIs hitting the optimiser bounds). The exponential form is statistically indistinguishable from the power form on LaMP-3 and slightly preferred on LaMP-1. Quote the knee and, for LaMP-3 only, the shared floor.
4. **'Personalization quality has an irreducible floor of X.'** Only LaMP-3's floor is resolvable (0.121 MAE [0.032, 0.181]), and it is a floor *at k <= 32 with BM25 retrieval and these three model tiers*, not an information-theoretic bound. The survey's recommendation against information-theoretic framing stands: we have no such result.
5. **'Our decision rule improves quality.'** It does not. Always-upgrading has lower regret (0.0057 vs 0.0162 at tau=0.02). The rule trades a measured quantity of quality for a measured reduction in upgrades.
6. **'The retriever does not matter.'** We can say no *resolvable* difference at +-0.06 to +-0.09 resolution, on 2 of 4 tasks, one model tier, with a positional (not dated) recency baseline. A smaller ceiling shift, or one from a different retrieval objective, is not excluded.
7. **Anything about k > 32, chain-of-thought token budgets, thinking budgets, or reward-model-mediated Best-of-N.** Not measured. The compute axis here is model tier plus a 3-4 sample budget, selector-free.
8. **Stratum-level claims.** n=33 per task/stratum/tier; the per-stratum gain direction reverses between LaMP-1 and LaMP-3 and is within noise.
9. **Comparisons to published LaMP baselines.** Our LaMP-2 is the `LaMP_2/new/dev` movie-tag variant on a 99-item stratified subset with 9 gold-leakage/malformed exclusions upstream; the data track flagged that published 'LaMP-2' numbers may refer to the news-categorisation variant. Do not table our numbers against a published baseline.

### 9.2 The single most defensible sentence the data will carry

> Retrieving eight of a user's own past items reduces LaMP-3 rating error by 0.31 MAE [0.19, 0.44] and saturates there, while moving from the smallest to the largest available model at fixed signal buys 0.17 MAE at k=0 and nothing at all by k=32 (tier spread 0.021 [0.011, 0.095]); matching the signal gain along the compute axis would take 3.65 model-scale steps on a ladder that has two.

## 10. Numbers for the paper

Quote these verbatim. **Precision convention: MAE, accuracy, macro-F1 and ROUGE to 4 decimal places as computed; report them rounded to 3 decimals in the paper text and keep 4 in tables. Brackets are 95% percentile bootstrap CIs over items, 2000 resamples, seed 12345. Differences between two cells on the same task are paired.** Every value below is reproduced from the saved artifacts, not retyped.

### Headline surface, LaMP-3 MAE (n=99 unless noted)

| | haiku | sonnet | opus |
|---|---|---|---|
| k=0 | 0.4545 [0.3333, 0.5758] | 0.2653 [0.1735, 0.3574] (n=98) | 0.2828 [0.1919, 0.3838] |
| k=8 | 0.1414 [0.0707, 0.2222] | 0.1111 [0.0505, 0.1818] | 0.1313 [0.0606, 0.2121] |
| k=32 | 0.1414 [0.0707, 0.2222] | 0.1224 [0.0612, 0.1939] (n=98) | 0.1515 [0.0808, 0.2323] |

### The substitution claim

- LaMP-3, haiku at k=8 versus opus at k=0: **-0.1414 MAE [-0.2222, -0.0606], n=99, p=0.0005** (paired). Phrase as: the small model with eight retrieved reviews halves the error of the largest model with none (0.1414 vs 0.2828).
- LaMP-1, haiku at k=16 versus opus at k=0: **+0.1982 accuracy [0.1010, 0.2993], n=99, p=0.0005**.
- LaMP-2, haiku at k=16 versus opus at k=0: **+0.1015 macro-F1 [0.0212, 0.1708], n=99, p=0.0060**.
- LaMP-5, haiku at k=8 versus opus at k=0: **-0.0174 ROUGE-1 [-0.0646, 0.0281], n=99, p=0.4650 -- not resolvable, and must be reported as the negative case.**

### Saturation

- LaMP-1: total k=0->32 gain 0.1821 [0.1100, 0.2543] (accuracy), knee k=8 [4, 16], fraction of gain by k=8 0.962 [0.759, 1.238], n=97 items
- LaMP-2: total k=0->32 gain 0.1128 [0.0335, 0.1730] (macro_f1), knee k=8 [1, 32], fraction of gain by k=8 1.003 [0.693, 1.881], n=77 items
- LaMP-3: total k=0->32 gain 0.2105 [0.1404, 0.2842] (mae), knee k=8 [4, 8], fraction of gain by k=8 1.067 [0.910, 1.255], n=95 items
- LaMP-5: total k=0->32 gain 0.0207 [-0.0096, 0.0528] (rouge_1), knee k=2 [1, 32], fraction of gain by k=8 0.833 [-1.618, 4.063], n=98 items

### The interaction

- LaMP-3, DiD [opus-haiku] x [k=32 vs k=0]: 0.1818 [0.0606, 0.3232] (mae units), n=99, p=0.0030
- LaMP-2, DiD [opus-haiku] x [k=32 vs k=0]: 0.1568 [0.0224, 0.2490] (macro_f1 units), n=99, p=0.0160
- LaMP-1, DiD [opus-haiku] x [k=32 vs k=0]: -0.0101 [-0.1515, 0.1111] (accuracy units), n=99, p=0.8860

### Fitted surface, LaMP-3 (shared-floor form, n=95 items)

- `MAE(k) = e_inf + a_m (k+1)^(-b_m)`, e_inf = **0.1206 [0.0323, 0.1808] MAE** shared across tiers
- haiku: a = 0.348 [0.258, 0.470], b = 0.900 [0.406, 1.627]
- sonnet: a = 0.162 [0.095, 0.265], b = 0.979 [0.326, 2.256]
- opus: a = 0.181 [0.117, 0.277], b = 0.905 [0.283, 1.814]
- In-sample RMSE 0.0194 MAE. Forcing the shared floor **improves** held-out MSE by 1.54e-4 (paired SE 3.5e-5, t=-4.41, 50 folds).
- Dropping the signal term costs +5.01e-3 (SE 4.5e-4, t=+11.26); dropping the compute term costs +9.16e-4 (SE 2.4e-4, t=+3.81).

### Exchange rate

- LaMP-3: signal span (haiku, k=0->32) 0.3131 MAE; compute span (haiku->opus, 2 tiers) 0.1717 MAE at k=0 and -0.0101 at k=32. **3.65 model-scale steps** would be needed to match the signal span; the ladder has 2.
- LaMP-1 and LaMP-2: compute span at k=0 is -0.0202 accuracy and -0.0992 macro-F1 respectively (i.e. non-positive), so the signal gain is **unreachable at any number of model-scale steps**.

### Decision rule

- LaMP-3 gain model: c = 0.1668, d = 0.8942. Thresholds: k*(tau=0.02) = 9.7, k*(tau=0.05) = 2.8 (held-out medians 13.8 and 3.7).
- Held-out regret vs a per-k oracle, 20 splits: tau=0.02 **0.0162 MAE** (random at matched budget 0.0245); tau=0.05 **0.0226** (random 0.0356). Always-upgrade 0.0057; never-upgrade 0.0590 (loss 0.2516 vs oracle 0.1926).
- Upgrade rate: 71% at tau=0.02, 49% at tau=0.05.

### Mechanism

- Fully deterministic items (3-4 samples, temperature 1.0, CoT, haiku): LaMP-3 k=0 **0.667 [0.513, 0.821]**, k=32 0.513 [0.359, 0.667]; LaMP-2 k=0 0.590 [0.436, 0.744], k=32 0.385 [0.231, 0.538]. **n=39 items per cell.**
- Mean normalised answer entropy 0.14-0.42 across the six cells; no item exceeded 3 distinct answers.
- Majority-vote gain over a single sample: not resolvable at k=0 or k=4 on either task; LaMP-3 k=32 +0.250 MAE [0.096, 0.423], LaMP-2 k=32 +0.086 accuracy [0.017, 0.162].
- On the same 39 items at k=32: CoT 4-sample majority 0.2564 MAE versus **one greedy direct call 0.1026 MAE**.

### Controls

- Chain-of-thought at matched k: 4 of 11 contrasts resolvable, three of them harms (LaMP-2 k=32 -0.0816 [-0.1529, -0.0033]; LaMP-3 k=32 -0.4872 [-0.8462, -0.1795], n=39; LaMP-5 k=32 -0.1646 [-0.2158, -0.1122]) and one a gain (LaMP-5 k=0 +0.0649 [0.0340, 0.0967]).
- BM25 vs recency: **0 of 12 matched-k contrasts resolvable.** Ceiling difference (max over k=1..32): LaMP-1 +0.0012 [-0.0993, 0.0874] accuracy, n=99; LaMP-2 -0.0215 [-0.0803, 0.0406] macro-F1, n=90. Recency is positional, not dated.

### Floors and coverage

- LaMP-3 constant-prediction accuracy: **0.6364** on our 99-item subset (rating 5), 0.5912 on the full dev split. 20 of 21 grid-A cells clear the subset floor; the exception is haiku at k=0 (0.6061).
- Only LaMP-2 caps the signal axis: mean effective_k 22.1 at k=32 with 60% of users capped; 13.9 at k=16 with 38% capped. LaMP-1/3/5 never cap.
- Measurement losses of 11,394 responses: 29 refusals (all Sonnet, direct), 168 usage-limit aborts, 14 unparseable (0.13% of the 11,197 usable).

### Figures

- **F1** `F1_signal_compute_surface.png` -- the surface (a-d), the tier gap versus k (e), the small-with-signal versus large-without contrast (f). n=93-99 items per cell in (a-d); n=77-98 in (e) (items shared by all 21 cells of a task); n=99 paired in (f).
- **F2** `F2_saturation_fit.png` -- shared-floor fits and knee CI bands per task (a-d), cumulative gain fraction (e), held-out cost of dropping each term (f).
- **F3** `F3_mechanism_entropy.png` -- distinct-answer distribution (a), normalised entropy (b), determinism split by correctness (c), sampling budget versus one greedy call (d). **n=39 items per cell throughout.**
- **F4** `F4_controls.png` -- CoT contrasts (a), retriever contrasts (b), retriever curves and ceilings (c), measurement losses (d).

## 11. Files

| File | Contents |
|---|---|
| `cell_metrics.csv` | 106 cells x all metrics, n, n_items, parse failures and rates, 95% bootstrap CIs, mean tokens |
| `contrasts.csv` | 294 paired contrasts: signal ladder, signal steps, compute at fixed k, interaction DiD, controls, ceilings; each with delta, CI, SE, bootstrap p, and a `resolved` flag |
| `fits.json` | all six functional forms, paired-CV comparison per task, per-model and shared-floor parameters with CIs, non-parametric knees, gain fractions, exchange rates, substitution table |
| `fit_curves.json` | fitted shared-floor parameters and the 21 observed cell errors per task (what F2 a-d plots) |
| `decision_rule.json` | rule specification, full-data gain fits, and all 400 held-out split evaluations |
| `entropy.csv` | 6 grid-B cells: determinism, normalised entropy, self-consistency gain, determinism by correctness, all with CIs and exact n |
| `entropy_per_item.csv` | per-item answer distributions for the 39-item grid-B cells |
| `sampling_vs_direct.csv` | CoT single / CoT majority / one greedy direct call on the same 39 items |
| `controls.csv` | grid C and grid R contrasts plus the retriever ceiling test |
| `model_spread.csv` | tier spread versus k with CIs (F1e) |
| `substitution_best_k.csv` | haiku at best k versus opus at k=0, paired (F1f) |
| `stratum_gains.csv` | per-stratum k-gains, exploratory |
| `F1-F4 *.png` | 300 dpi |

