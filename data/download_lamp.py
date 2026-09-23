#!/usr/bin/env python3
"""Download the raw LaMP dev splits used in this paper.

You do NOT need this to re-run the sweep or the analysis: the evaluation
subsets in `data/subsets/` embed each user's full profile, so the pipeline runs
without the ~993 MB of raw data. Use this only to rebuild the subsets from
scratch (see build_subsets.py) or to work with the full dev splits.

URL pattern (verified against the official download index):

    https://ciir.cs.umass.edu/downloads/LaMP/LaMP_{N}/{split}/{split}_questions.json
    https://ciir.cs.umass.edu/downloads/LaMP/LaMP_{N}/{split}/{split}_outputs.json

Two things that cost time if you guess instead of reading:

  * The task this paper calls LaMP-2 (movie tagging) lives under
    `LaMP_2/new/dev/`. The plain `LaMP_2/dev/` path serves NEWS CATEGORIZATION
    with a different profile schema. Published "LaMP-2" numbers may refer to
    either; check before comparing.
  * Paths that look plausible but are wrong (e.g. `LaMP_2/user/dev/...`) return
    HTTP 200 with a ~16 KB HTML error body, not a 404. Verify sizes.

LaMP-6 needs the licensed Avocado corpus and is not used here.
Test golds are withheld for the leaderboard, so all scoring uses DEV.
"""
from __future__ import annotations

import argparse
import os
import urllib.request

BASE = "https://ciir.cs.umass.edu/downloads/LaMP"

# Local name -> URL path fragment for the four tasks used in the paper.
TASKS = {
    "LaMP-1": "LaMP_1",
    "LaMP-2": "LaMP_2/new",   # movie tagging -- NOT LaMP_2/dev
    "LaMP-3": "LaMP_3",
    "LaMP-5": "LaMP_5",
}


def fetch(url: str, dest: str) -> None:
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        print(f"[skip] {dest} exists ({os.path.getsize(dest):,} B)")
        return
    print(f"[get ] {url}")
    tmp = dest + ".part"
    urllib.request.urlretrieve(url, tmp)
    size = os.path.getsize(tmp)
    # A wrong-but-live path returns a small HTML error body; real files are large.
    if size < 100_000:
        head = open(tmp, "rb").read(200)
        if b"<html" in head.lower():
            os.remove(tmp)
            raise RuntimeError(f"{url} returned an HTML error body, not JSON")
    os.replace(tmp, dest)
    print(f"[ok  ] {dest} ({size:,} B)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=os.path.join(os.path.dirname(__file__), "raw"))
    ap.add_argument("--split", default="dev", choices=["train", "dev"])
    ap.add_argument("--tasks", default=",".join(TASKS), help="comma-separated task names")
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    for t in [x.strip() for x in a.tasks.split(",")]:
        frag = TASKS[t]
        for kind in ("questions", "outputs"):
            url = f"{BASE}/{frag}/{a.split}/{a.split}_{kind}.json"
            dest = os.path.join(a.outdir, f"{t.replace('-', '')}_{a.split}_{kind}.json")
            fetch(url, dest)


if __name__ == "__main__":
    main()
