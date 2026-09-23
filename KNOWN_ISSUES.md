# Known issues in the released data

Found while packaging this repository, after the paper was written. Nothing here
was known at analysis time. Everything below is reproducible with the scripts in
`scripts/`; no claim in this file rests on memory.

## 1. Two consolidations of the raw responses, and why both are shipped

`results/cells/` holds the 135 per-cell parquet files the sweep actually wrote.
These are the primary record: each file is one cell, and its filename carries
the cell's identity including the sample budget (`__s1__`, `__s4__`).

Two consolidated tables are derived from them:

| File | Usable rows | What it is |
|---|---|---|
| `sweep_all_raw.parquet` | 11,197 | **The exact input the paper's analysis consumed.** Use it to reproduce published numbers. |
| `sweep_all_rebuilt.parquet` | 11,403 | Rebuilt by `scripts/rebuild_merge.py` with a corrected key. |

The original consolidation de-duplicated on
`(task, model, k, variant, retriever, qid, sample_idx)` — **without the sample
budget**. The sampling grid (4 samples, temperature 1.0) and the
chain-of-thought control (1 sample, greedy) share every one of those keys, so
for items present in both, their `sample_idx = 0` rows collided and one was
discarded. It also had no tie-break for cells that were partly re-run, so where
two *successful* responses existed for the same slot, which one survived
depended on file iteration order.

`rebuild_merge.py` includes the sample budget in the key and applies a stable
tie-break (non-errored first, then lexicographic source filename), so it is
deterministic across runs — verified by running it twice and comparing frames.

## 2. What changes between the two, cell by cell

Reproduce with:

    python scripts/score_sweep.py --raw results/sweep_all_rebuilt.parquet --check

Of the 106 cells in `results/cell_metrics.csv`, **17 differ** in headline value.

**Grid C (chain-of-thought control) — 3 cells, and the corrected values are the
better ones.** These are the collision victims: the paper's analysis saw 60
items where 99 were available.

| Task | k | n (paper → rebuilt) | Headline (paper → rebuilt) |
|---|---|---|---|
| LaMP-3 | 4 | 60 → 99 | MAE 0.1833 → 0.2929 |
| LaMP-3 | 0 | 60 → 99 | MAE 0.3500 → 0.4040 |
| LaMP-2 | 4 | 60 → 99 | macro-F1 0.4122 → 0.4370 |

**Grid A (the main surface) — 14 cells, all at identical n, max difference
0.0379.** These come from the missing tie-break, not from lost data: duplicate
*successful* responses exist for those slots and the two consolidations kept
different ones. Neither is more correct than the other. The largest are
LaMP-5/sonnet *k*=1 (0.0379) and LaMP-2/sonnet *k*=8 (0.0303).

## 3. What this does and does not touch in the paper

Checked against the differing-cell list, not assumed:

- **The headline substitution result is unaffected.** It rests on LaMP-3 grid A,
  and no LaMP-3 grid-A cell differs. Small model with history (MAE 0.1414 at
  *k*=8) versus large model without (0.2828 at *k*=0) is unchanged in both
  consolidations.
- **The LaMP-2 cross-term may shift.** The complementarity result uses LaMP-2
  sonnet cells, five of which move by up to 0.0303 macro-F1. The reported
  difference-in-differences was 0.157 with *p* = 0.016; it has **not** been
  recomputed here.
- **The chain-of-thought control at LaMP-3 *k*=4 looks materially worse on the
  corrected data** (MAE 0.293 against 0.202 for direct prompting at the same
  *k*), where the paper reported that contrast as not resolvable. This warrants
  recomputation before the control paragraph is relied on.
- Saturation fits, the exchange rate, the entropy/determinism mechanism, the
  retriever control and the decision rule all draw on cells that are unchanged
  or on grid A cells for tasks whose conclusions were already reported as not
  resolvable.

**The analysis has not been re-run on the rebuilt table.** Figures, CIs and
fitted parameters in `results/` and `docs/analysis_report.md` all correspond to
`sweep_all_raw.parquet`.

## 4. A cell reported as empty retained 11 responses

`docs/analysis_report.md` and the paper describe the LaMP-3 chain-of-thought
cell at *k*=32 as emptied by infrastructure aborts. It in fact retained **11 of
99** responses (88 carry a usage-limit `api_error`). Excluding it was the right
call — 11 surviving items is an uninformative and probably skewed sample, and
`rebuild_merge.py` flags it in its thin-cell report — but "emptied" is a rounding
of 11 down to 0 rather than a literal count. No reported number depends on it:
the cell is absent from `cell_metrics.csv` in both consolidations.

## 5. One cell is genuinely absent

LaMP-5 / small tier / *k*=4 in grid A has 99 response rows and **zero** usable
ones: every call hit a usage limit. Grid A is 83 of 84 cells. This is stated in
the paper and is not an artifact of either consolidation.

## 6. The subset rebuild script is a reimplementation

`data/build_subsets.py` reconstructs the evaluation subsets from the procedure
documented in `docs/data_report.md`. The original selection was done inline and
no script was saved, so this is a faithful re-derivation, **not the original
code**, and it has not been verified to reproduce the shipped parquets
bit-for-bit (doing so needs the ~993 MB of raw dev data). The shipped
`data/subsets/*.parquet` are authoritative; treat the script as documentation of
how they were built.
