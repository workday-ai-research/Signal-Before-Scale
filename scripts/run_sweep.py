#!/usr/bin/env python3
"""Run one shard of the signal x compute sweep.

The paper's grid is 4 tasks x 3 model tiers x 7 signal levels (grid A), plus a
sampling grid (B), a chain-of-thought control (C) and a retriever control (R).
Each (task, model, k, variant, retriever, samples) combination is one CELL,
written to its own parquet so a crash costs at most one cell and re-running
skips finished work.

Examples
--------
Grid A, one task/tier, the full k axis:

    python scripts/run_sweep.py --task LaMP-3 --model <model-id> \
        --ks 0,1,2,4,8,16,32 --n 99 --outdir runs/gridA

Grid B (sampling; needs a model that accepts an explicit temperature):

    python scripts/run_sweep.py --task LaMP-3 --model <model-id> \
        --ks 0,4,32 --variant cot --samples 4 --temperature 1.0 \
        --n 39 --outdir runs/gridB

Retriever control:

    python scripts/run_sweep.py --task LaMP-1 --model <model-id> \
        --ks 1,2,4,8,16,32 --retriever recency --n 99 --outdir runs/gridR

Dry run with no API calls (checks prompts, parsing, scoring, checkpointing):

    python scripts/run_sweep.py --task LaMP-3 --model dummy --ks 0,4 \
        --n 6 --backend echo --outdir /tmp/smoke
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import sweep_runner as SR  # noqa: E402
from llm_backend import AnthropicBackend, EchoBackend  # noqa: E402

REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
SUBSETS = {
    "LaMP-1": "lamp_eval_subset_lamp1.parquet",
    "LaMP-2": "lamp_eval_subset_lamp2.parquet",
    "LaMP-3": "lamp_eval_subset_lamp3.parquet",
    "LaMP-5": "lamp_eval_subset_lamp5.parquet",
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", required=True, choices=sorted(SUBSETS))
    ap.add_argument("--model", required=True, help="model identifier passed to the backend")
    ap.add_argument("--ks", required=True, help="comma-separated signal levels, e.g. 0,1,2,4,8,16,32")
    ap.add_argument("--variant", default="direct", choices=["direct", "cot"])
    ap.add_argument("--retriever", default="bm25", choices=["bm25", "recency"])
    ap.add_argument("--samples", type=int, default=1)
    ap.add_argument("--temperature", type=float, default=None,
                    help="omit for greedy decoding; see llm_backend on why absent != 0.0")
    ap.add_argument("--n", type=int, default=99, help="items per cell (paper: 99 for A/C/R, 39 for B)")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--outdir", default="runs")
    ap.add_argument("--subset", default=None, help="override the subset parquet path")
    ap.add_argument("--backend", default="anthropic", choices=["anthropic", "echo"])
    ap.add_argument("--tag", default="")
    a = ap.parse_args()

    backend = EchoBackend() if a.backend == "echo" else AnthropicBackend()
    if a.backend == "echo":
        print("WARNING: echo backend -- pipeline smoke test only, scores are meaningless.")

    subset = a.subset or os.path.join(REPO, "data", "subsets", SUBSETS[a.task])
    paths = SR.run_shard(
        backend,
        task=a.task,
        subset_path=subset,
        model=a.model,
        ks=[int(x) for x in a.ks.split(",")],
        variant=a.variant,
        retriever=a.retriever,
        samples=a.samples,
        temperature=a.temperature,
        n=a.n,
        concurrency=a.concurrency,
        outdir=a.outdir,
        tag=a.tag,
    )
    print("\n".join(paths))


if __name__ == "__main__":
    main()
