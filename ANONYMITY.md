# Anonymity

This repository accompanies a double-blind submission. It carries no author
names, affiliations, emails, funding statements or acknowledgments.

## What was removed when packaging

The code was written inside a hosted research environment and referenced it in
ways that would identify the account. Those references were replaced, not merely
hidden:

- **Absolute home-directory paths** (`/Users/<name>/...`) that appeared in
  `acore.py`, `gen_tables.py` and `gen_appendix.py` as hardcoded input paths.
  Replaced with a `RESULTS` directory resolved relative to the repository, or
  overridable via the `SBS_RESULTS` environment variable.
- **A hardcoded path to a versioned copy of `lamp_harness.py`**, which
  `acore.py` loaded via `importlib` from an artifact store. Replaced with a
  normal `import lamp_harness as lh`.
- **Internal artifact and object identifiers** — UUIDs and version-prefixed
  filenames (e.g. `v217b5c2d_sweep_all_raw.parquet`) — in `gen_tables.py`,
  `gen_appendix.py`, `docs/analysis_report.md` and the provenance fields of
  `results/fits.json`. Replaced with repository-relative filenames.
- **Calls into the hosting platform's SDK** (`host.llm`, `host.artifact_path`).
  The model call is now `src/llm_backend.py`, a documented interface with a
  public-API reference implementation; the file reads are ordinary paths.
- **Two of the authors' own unpublished works** were referenced in the working
  notes as related work under review elsewhere. They are not named, described or
  cited anywhere in this repository or in the paper, which is self-contained.

## Verification

A scan over every text file in the repository (31 files: sources, docs, CSV,
JSON, configs) for home-directory paths, email addresses, author surnames,
platform SDK calls, project identifiers, store UUIDs, version-prefixed
filenames, editor/service URLs and credential-shaped strings returns no matches.
Reproduce it after any edit, before pushing:

```bash
# Excludes this file, which necessarily quotes the patterns it searches for.
grep -rInE '/Users/|[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}|host\.(llm|artifact_path)|proj_[0-9a-f]+|v[0-9a-f]{8}_' \
  --include='*.py' --include='*.md' --include='*.json' --include='*.csv' \
  --include='*.txt' --include='*.sh' --exclude='ANONYMITY.md' . \
  && echo 'MATCHES FOUND' || echo 'clean'
```

Binary files (`*.parquet`, `*.png`) were not text-scanned. The parquets contain
benchmark inputs, model responses and scores; the response tables carry a
`_src` column with the sweep's own per-cell filenames (task, model, *k*,
variant, sample budget), which encode experimental configuration only.

## What is deliberately retained

- **Model identifiers** (`claude-haiku-4-5-...` and the other two tiers). These
  are public product identifiers and appear in the paper; they are not
  identifying, and removing them would make the compute axis unreadable.
- **The LaMP download host and URL pattern**, which is the public benchmark
  distribution point.
