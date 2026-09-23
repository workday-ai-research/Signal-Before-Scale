#!/usr/bin/env python3
"""Rebuild the stratified evaluation subsets from the raw LaMP dev splits.

NOT THE ORIGINAL CODE. The original selection was done inline and no script was
saved; this re-derives it from the procedure recorded in docs/data_report.md. It
has not been verified to reproduce data/subsets/*.parquet bit-for-bit, because
that needs the ~993 MB of raw dev data. The shipped parquets are authoritative;
read this as executable documentation of how they were built. See KNOWN_ISSUES.md.

Procedure (docs/data_report.md):
  1. Filter ineligible items:
       - gold leakage: the gold answer appears verbatim in the user's own
         profile, making it retrievable for free at high k and manufacturing
         exactly the k-dependent gain the study measures
         (LaMP-5: 9/2500; LaMP-1: candidate title in profile)
       - malformed input: one LaMP-2 query is missing its instruction and tag
         list entirely
  2. Sort the eligible pool by (profile_len, qid).
  3. Cut at ranks n//3 and 2n//3 into short / medium / long profile-length
     terciles.
  4. Within each stratum, take a seeded sample (seed=0) of 50 items, giving
     150 per task, exactly balanced by construction.

The subset embeds each user's full profile, which is why the sweep never needs
the raw data again.

    python data/build_subsets.py --raw data/raw --out data/subsets_rebuilt
"""
from __future__ import annotations

import argparse
import json
import os
import random

import pandas as pd

TASKS = ("LaMP-1", "LaMP-2", "LaMP-3", "LaMP-5")
PER_STRATUM = 50
SEED = 0

# Profile text fields per task, verified against the dev dumps (see data_report).
PROFILE_TEXT = {
    "LaMP-1": ("title", "abstract"),
    "LaMP-2": ("description",),
    "LaMP-3": ("text",),
    "LaMP-5": ("title", "abstract"),
}


def gold_leaks(task: str, gold: str, profile: list[dict]) -> bool:
    """True when the gold answer appears verbatim in the user's own profile."""
    g = (gold or "").strip().lower()
    if not g:
        return False
    if task == "LaMP-5":
        return any((p.get("title") or "").strip().lower() == g for p in profile)
    if task == "LaMP-1":
        return any(g in (p.get("title") or "").strip().lower() for p in profile)
    return False


def malformed(task: str, text: str) -> bool:
    """LaMP-2's single broken query: missing instruction and tag list."""
    return task == "LaMP-2" and not text.lower().startswith("which tag does this movie")


def build(task: str, qpath: str, opath: str) -> pd.DataFrame:
    questions = json.load(open(qpath, encoding="utf-8"))
    outputs = json.load(open(opath, encoding="utf-8"))
    golds = {d["id"]: d["output"] for d in outputs["golds"]}

    pool, dropped_leak, dropped_bad = [], 0, 0
    for q in questions:
        qid, prof = str(q["id"]), q.get("profile", [])
        gold = golds.get(qid)
        if gold is None:
            continue
        if malformed(task, q["input"]):
            dropped_bad += 1
            continue
        if gold_leaks(task, gold, prof):
            dropped_leak += 1
            continue
        pool.append({"qid": qid, "input": q["input"], "gold": gold,
                     "profile_json": json.dumps(prof), "profile_len": len(prof)})

    df = pd.DataFrame(pool).sort_values(["profile_len", "qid"]).reset_index(drop=True)
    n = len(df)
    cuts = [0, n // 3, 2 * n // 3, n]
    names = ["short", "medium", "long"]
    rng = random.Random(SEED)
    parts = []
    for name, lo, hi in zip(names, cuts[:-1], cuts[1:]):
        block = df.iloc[lo:hi]
        take = min(PER_STRATUM, len(block))
        idx = sorted(rng.sample(range(len(block)), take))
        part = block.iloc[idx].copy()
        part["stratum"] = name
        parts.append(part)
    out = pd.concat(parts, ignore_index=True)
    out["task"] = task
    out["seed"] = SEED
    print(f"{task}: pool {n} (dropped {dropped_leak} gold-leak, {dropped_bad} malformed) "
          f"-> subset {len(out)} ({out.stratum.value_counts().to_dict()})")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    here = os.path.dirname(os.path.abspath(__file__))
    ap.add_argument("--raw", default=os.path.join(here, "raw"))
    ap.add_argument("--out", default=os.path.join(here, "subsets_rebuilt"))
    ap.add_argument("--tasks", default=",".join(TASKS))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    for task in [t.strip() for t in a.tasks.split(",")]:
        stem = task.replace("-", "")
        q = os.path.join(a.raw, f"{stem}_dev_questions.json")
        o = os.path.join(a.raw, f"{stem}_dev_outputs.json")
        if not (os.path.exists(q) and os.path.exists(o)):
            print(f"{task}: raw files missing under {a.raw}; run download_lamp.py first")
            continue
        df = build(task, q, o)
        dest = os.path.join(a.out, f"lamp_eval_subset_{stem.lower()}.parquet")
        df.to_parquet(dest, index=False)
        print(f"  wrote {dest}")


if __name__ == "__main__":
    main()
