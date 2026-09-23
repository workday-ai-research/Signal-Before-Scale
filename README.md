# Signal Before Scale

Code, data subsets and raw model responses for a study of how **per-user signal
volume** and **inference compute** interact in LLM personalization.

The experiment crosses a controlled signal axis (*k* ∈ {0,1,2,4,8,16,32}
retrieved items from the user's own history) with a compute axis (three model
tiers; plus a chain-of-thought sampling budget) on four
[LaMP](https://lamp-benchmark.github.io/) tasks — 11,394 model responses in all.

Anonymized for double-blind review. No author or institution information is
included; see [ANONYMITY.md](ANONYMITY.md).

## What is here

```
data/
  subsets/                 4 evaluation subsets (parquet) -- profiles embedded,
                           so the sweep runs without the ~993 MB of raw LaMP
  download_lamp.py         fetch the raw dev splits (only needed to rebuild)
  build_subsets.py         re-derive the subsets (reimplementation -- see KNOWN_ISSUES)
src/
  lamp_harness.py          retrieval (BM25 / recency), prompt assembly, parsing, metrics
  llm_backend.py           model-call interface + reference implementation
  sweep_runner.py          sharded runner, one parquet per cell, resumable
  acore.py                 per-item metrics, bootstrap, saturating fits, decision rule
scripts/
  run_sweep.py             CLI: run one shard of the grid
  score_sweep.py           recompute per-cell metrics from saved responses (no API needed)
  rebuild_merge.py         consolidate results/cells/ deterministically
  gen_tables.py            LaTeX tables for the paper
  gen_appendix.py          LaTeX appendix tables
  xval_rouge.py            cross-check the ROUGE implementation against rouge-score
results/
  cells/                   135 per-cell parquets -- the primary record
  sweep_all_raw.parquet    consolidation the paper's analysis consumed
  sweep_all_rebuilt.parquet corrected consolidation (see KNOWN_ISSUES.md)
  cell_metrics.csv         per-cell metrics with n and parse-failure rates
  contrasts.csv            paired bootstrap contrasts
  entropy.csv, entropy_per_item.csv, controls.csv, model_spread.csv,
  sampling_vs_direct.csv, substitution_best_k.csv, stratum_gains.csv
  fits.json, fit_curves.json, decision_rule.json
  figures/                 F1-F4 as published
docs/
  analysis_report.md       every reported number with its CI, and what the data does not support
  data_report.md           subset construction, exclusions, metric floors, task quirks
```

## Reproducing without any API access

Every published per-cell number can be recomputed from the saved responses:

```bash
pip install -r requirements.txt
python scripts/score_sweep.py --check
```

This re-scores every cell from the saved responses and compares against
`results/cell_metrics.csv`. It reports, per cell, where the recomputation
differs in value or in item count.

It defaults to `results/sweep_all_rebuilt.parquet`, the **corrected**
consolidation. Of the 106 published cells it reproduces 89 exactly and differs
on 17 — 14 of them by at most 0.038 in metric units (an arbitrary choice between
duplicate successful responses), and 3 chain-of-thought control cells where the
corrected data supplies 99 items instead of 60.

**Read [KNOWN_ISSUES.md](KNOWN_ISSUES.md) before quoting any of these numbers.**
It documents the de-duplication defect behind those differences, lists the
affected cells, and states what does and does not change in the paper's
conclusions — the headline substitution result is unaffected; the LaMP-2
cross-term and the chain-of-thought control are the parts that could move.

To score the exact table the paper's analysis consumed:

```bash
python scripts/score_sweep.py --raw results/sweep_all_raw.parquet --check
```

That file cannot be scored unambiguously by a grid-aware scorer — separating the
sampling grid from the chain-of-thought control is precisely what its key was
missing — so expect it to disagree on the conflated cells. It is shipped for
provenance, not as the recommended input.

## Re-running the sweep

```bash
export ANTHROPIC_API_KEY=...
python scripts/run_sweep.py --task LaMP-3 --model <model-id> \
    --ks 0,1,2,4,8,16,32 --n 99 --outdir runs/gridA
```

Each cell is written to its own parquet as it completes and re-runs skip
finished cells, so an interrupted sweep resumes without repeating work.

To exercise the pipeline with no API calls at all:

```bash
python scripts/run_sweep.py --task LaMP-3 --model dummy --ks 0,4 \
    --n 6 --backend echo --outdir /tmp/smoke
```

### Important about the model backend

The paper's runs went through an internal batching harness that is not public.
`src/llm_backend.py` defines that harness's contract and ships
`AnthropicBackend` as a reference implementation against the public API. **The
published numbers were produced with the internal harness, not with
`AnthropicBackend`**, which was never exercised against the live API. Treat it
as a faithful reimplementation rather than a bit-for-bit reproduction path, and
expect drift from sampling nondeterminism and model versioning regardless.

One non-obvious detail it preserves: a request with **no** `temperature` key is
not the same as `temperature=0.0`. One model generation used here rejects an
explicit temperature argument outright, so the sampling conditions use a
different generation from the greedy ones.

## Grids

| Grid | Varies | Fixed | Cells |
|---|---|---|---|
| A | *k* × 3 model tiers, 4 tasks | direct prompt, BM25, 1 sample | 83 of 84 (one lost to API quota) |
| B | *k* ∈ {0,4,32}, 4 samples, temperature 1.0 | chain-of-thought, small tier, LaMP-2/3 | 6 |
| C | *k* ∈ {0,4,32}, chain-of-thought | 1 sample, small tier | 11 |
| R | *k* ≥ 1, recency retrieval | direct, small tier, LaMP-1/2 | 12 |

Metrics follow LaMP: accuracy (LaMP-1), macro-F1 over the closed 15-tag set
(LaMP-2), MAE (LaMP-3), ROUGE-1 (LaMP-5). Unparseable outputs are **reported and
excluded**, never scored as wrong — a compute-scaling claim should not rest on a
parser artifact. Cells with fewer than 90 usable items are reported at their
true *n* rather than pooled.

## Caveats worth knowing before building on this

- The three model tiers are treated as an ordered compute axis using the
  provider's published tier ordering. **No parameter or FLOP counts are
  available**, so any "model-scale steps" figure is relative to this ladder's
  rungs, not a measured scale multiple.
- The recency retriever is **positional**: LaMP profiles on these tasks carry no
  date field, so it takes the tail of the profile list and assumes list order is
  chronological.
- The task called LaMP-2 here is movie tagging, from `LaMP_2/new/dev`. The plain
  `LaMP_2/dev` path is a different task (news categorization). Published
  "LaMP-2" numbers may refer to either.
- Measurement losses are in the data, not smoothed away: 29 model refusals, 168
  usage-limit aborts, 14 unparseable of 11,394 responses.

## License

Code in this repository: MIT (see [LICENSE](LICENSE)). The LaMP benchmark and
its underlying corpora are governed by their own terms; the subsets under
`data/subsets/` are derived from the public LaMP dev splits.
