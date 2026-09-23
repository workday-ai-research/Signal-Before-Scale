#!/usr/bin/env python3
"""Consolidate the per-cell parquets in results/cells/ into one table.

Why this script exists, and why its output is NOT the file the paper used:

The consolidation that fed the paper's analysis de-duplicated on
(task, model, k, variant, retriever, qid, sample_idx) -- WITHOUT the sample
budget. The sampling grid (4 samples) and the chain-of-thought control
(1 sample) share every one of those keys, so for items appearing in both, their
`sample_idx = 0` rows collided and one was dropped. That removed 206 usable
responses and left two chain-of-thought control cells with 60 items instead of
99. See KNOWN_ISSUES.md for the three cells whose metrics change.

This script fixes the key by parsing the declared sample budget out of each
cell's filename and including it in both the cell identity and the dedup key.

Determinism: where two successful responses exist for the same
(cell, qid, sample_idx) -- which happens for cells that were partly re-run --
the surviving row is chosen by a stable rule (non-errored first, then
lexicographic source filename) so repeated runs of this script agree. The
paper's consolidation had no such tie-break, which is the second reason its
output and this one differ.

    python scripts/rebuild_merge.py --out results/sweep_all_rebuilt.parquet
"""
from __future__ import annotations

import argparse
import glob
import os
import re

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CELLS = os.path.join(HERE, "..", "results", "cells")
NAME = re.compile(
    r"(?P<task>LaMP-\d)__(?P<model>.+?)__k(?P<k>\d+)__"
    r"(?P<variant>direct|cot)__s(?P<samples>\d+)__(?P<retriever>bm25|recency)"
)

CELL_KEYS = ["task", "model", "k", "variant", "retriever", "samples"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", default=CELLS)
    ap.add_argument("--out", default=os.path.join(HERE, "..", "results", "sweep_all_rebuilt.parquet"))
    a = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(a.cells, "*.parquet")))
    if not paths:
        raise SystemExit(f"no parquets under {a.cells}")
    frames = []
    for p in paths:
        fn = os.path.basename(p)
        m = NAME.match(fn)
        if not m:
            raise SystemExit(f"cannot parse cell identity from filename: {fn}")
        d = pd.read_parquet(p)
        d["samples"] = int(m.group("samples"))
        d["_src"] = fn
        frames.append(d)
    raw = pd.concat(frames, ignore_index=True)

    n_before = len(raw)
    raw["_ok"] = raw["api_error"].isna().astype(int)
    raw = (raw.sort_values(["_ok", "_src"], ascending=[False, True])
              .drop_duplicates(subset=CELL_KEYS[:-1] + ["samples", "qid", "sample_idx"],
                               keep="first")
              .drop(columns=["_ok"])
              .reset_index(drop=True))

    usable = raw[raw["api_error"].isna()]
    cells = usable.groupby(CELL_KEYS)["qid"].nunique()
    print(f"files {len(paths)} | rows {n_before} -> {len(raw)} | usable {len(usable)}")
    print(f"cells {len(cells)} "
          f"({(cells.reset_index()['samples'] == 1).sum()} single-sample, "
          f"{(cells.reset_index()['samples'] > 1).sum()} multi-sample)")
    thin = cells[cells < 90]
    if len(thin):
        print("\ncells with fewer than 90 usable items (report these, never silently pool):")
        print(thin.to_string())
    raw.to_parquet(a.out, index=False)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
