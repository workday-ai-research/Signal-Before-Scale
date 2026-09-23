#!/usr/bin/env python3
"""Recompute per-cell metrics from the raw sweep records.

This is the offline reproduction path: it needs no API access and no model
calls. It reads `results/sweep_all_raw.parquet` (every response collected for
the paper, 11,394 rows) and recomputes the headline metric for each cell.

    python scripts/score_sweep.py                     # print the surface
    python scripts/score_sweep.py --check             # compare to the shipped
                                                      # cell_metrics.csv

Metric conventions follow the paper:
  LaMP-1 accuracy, LaMP-2 macro-F1 (label set PINNED to the closed 15-tag set),
  LaMP-3 MAE, LaMP-5 ROUGE-1. Rows whose `api_error` is non-null are dropped;
  rows that parsed to None are reported as parse failures and excluded from the
  metric average rather than scored as wrong.
"""
from __future__ import annotations

import argparse
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))

import lamp_harness as lh  # noqa: E402

RESULTS = os.environ.get("SBS_RESULTS", os.path.join(HERE, "..", "results"))
HEADLINE = {"LaMP-1": "accuracy", "LaMP-2": "macro_f1", "LaMP-3": "mae", "LaMP-5": "rouge_1"}


def score_cells(raw: pd.DataFrame) -> pd.DataFrame:
    """One row per CELL.

    A cell is (task, model, k, variant, retriever, samples). `samples` is part
    of the identity and must not be dropped: the sampling grid (4 samples,
    temperature 1.0) and the chain-of-thought control (1 sample, greedy) share
    every other key, so grouping without it silently merges two different
    experimental conditions -- and can resurrect a cell that was actually lost.
    """
    ok = raw[raw["api_error"].isna()].copy()
    base = ["task", "model", "k", "variant", "retriever"]
    if "samples" in ok.columns:
        # Rebuilt file: the sample budget is recorded per row, so cells are
        # unambiguous and the sampling grid stays separate from the CoT control.
        keys = base + ["samples"]
    else:
        # Paper-faithful file: it carries no sample-budget column and its
        # sampling-grid and CoT-control rows are conflated at sample_idx = 0
        # (see KNOWN_ISSUES.md). Grouping on `base` alone reproduces exactly the
        # view the paper's analysis had -- which is the point of this file.
        print("NOTE: no `samples` column; grouping without it to reproduce the "
              "paper's view. Use results/sweep_all_rebuilt.parquet for the "
              "corrected split (see KNOWN_ISSUES.md).")
        keys = base
    rows = []
    for key, d in ok.groupby(keys):
        task = key[0]
        # One row per item: majority vote when a cell has multiple samples.
        multi = ("samples" in d.columns and int(d["samples"].iloc[0]) > 1) \
            or ("samples" not in d.columns and d["sample_idx"].max() > 0)
        if multi:
            preds, golds = [], []
            for _, g in d.groupby("qid"):
                vals = [p for p in g["parsed"].tolist() if p is not None]
                preds.append(max(set(vals), key=vals.count) if vals else None)
                golds.append(g["gold"].iloc[0])
        else:
            preds, golds = d["parsed"].tolist(), d["gold"].tolist()
        m = lh.score_task(task, preds, golds, already_parsed=True)
        rows.append({
            **dict(zip(keys, key if isinstance(key, tuple) else (key,))),
            "n": len(preds),
            "parse_failures": sum(p is None for p in preds),
            "headline_metric": HEADLINE[task],
            "headline_value": m[HEADLINE[task]],
            **{f"m_{k2}": v for k2, v in m.items()},
        })
    return pd.DataFrame(rows).sort_values(keys).reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=os.path.join(RESULTS, "sweep_all_rebuilt.parquet"),
                    help="default is the corrected consolidation; pass "
                         "results/sweep_all_raw.parquet for the exact input the "
                         "paper's analysis consumed (see KNOWN_ISSUES.md)")
    ap.add_argument("--check", action="store_true",
                    help="compare recomputed headline values against cell_metrics.csv")
    ap.add_argument("--out", default=None, help="write the recomputed table here")
    a = ap.parse_args()

    raw = pd.read_parquet(a.raw)
    cells = score_cells(raw)
    print(f"cells scored: {len(cells)}  (rows in: {len(raw)}, "
          f"api_error dropped: {int(raw['api_error'].notna().sum())})")

    gridA = cells[(cells.variant == "direct") & (cells.retriever == "bm25")]
    for task, g in gridA.groupby("task"):
        piv = g.pivot(index="k", columns="model", values="headline_value")
        print(f"\n== {task} ({HEADLINE[task]}) ==")
        print(piv.round(4).to_string())

    if a.out:
        cells.to_csv(a.out, index=False)
        print(f"\nwrote {a.out}")

    if a.check:
        ref_path = os.path.join(RESULTS, "cell_metrics.csv")
        ref = pd.read_csv(ref_path)
        keys = ["task", "model", "k", "variant", "retriever"]
        refcol = next((c for c in ref.columns if c in ("headline_value", "value", "metric_value")), None)
        if refcol is None:
            print(f"\n[check] no headline column found in {os.path.basename(ref_path)}; "
                  f"columns are: {list(ref.columns)[:12]}")
            return
        # The shipped table keys models by short tier name; raw records carry
        # the full model id. Join on whichever column matches.
        if "model_full" in ref.columns and not set(cells["model"]) & set(ref["model"]):
            ref = ref.drop(columns=["model"]).rename(columns={"model_full": "model"})
        # cell_metrics.csv covers the single-sample grids (A, C, R); the
        # multi-sample grid is analysed separately in entropy.csv.
        jkeys = ["task", "model", "k", "variant", "retriever"]
        single = cells[cells["samples"] == 1] if "samples" in cells.columns else cells
        # Rename the reference column explicitly. Merging two frames that both
        # carry `headline_value` and then reading `merged[refcol]` returns the
        # LEFT column, i.e. compares the recomputed value with itself and always
        # reports a zero difference.
        refc = ref[jkeys + [refcol, "n"]].rename(
            columns={refcol: "ref_value", "n": "ref_n"})
        merged = single.merge(refc, on=jkeys, how="inner")
        merged["n_diff"] = (merged["n"] - merged["ref_n"]).abs()
        refcol = "ref_value"
        merged["absdiff"] = (merged["headline_value"] - merged[refcol]).abs()
        n_val = int((merged["absdiff"] > 1e-9).sum())
        n_n = int((merged["n_diff"] > 0).sum())
        print(f"\n[check] matched {len(merged)} cells; "
              f"max |diff| = {merged['absdiff'].max():.3e}; "
              f"cells differing in value: {n_val}; in item count: {n_n}")
        if n_val or n_n:
            cols = jkeys + ["n", "ref_n", "headline_value", refcol, "absdiff"]
            print(merged[(merged.absdiff > 1e-9) | (merged.n_diff > 0)][cols]
                  .sort_values("absdiff", ascending=False).to_string(index=False))
        else:
            print("all matched cells agree exactly on value and item count.")


if __name__ == "__main__":
    main()
