"""
Sharded LaMP sweep runner.

Why it is built this way:
  * The harness the paper's runs used billed a FIXED ~3,500-token overhead per
    call on top of the prompt, and enforced a ~2.0M input-token ceiling per
    worker process. One worker could not hold the full grid, so work is sharded,
    each shard sized to fit under that ceiling. Those constants are properties of
    that harness, not of the benchmark -- with a different backend you may not
    need sharding at all, but the shard boundaries are preserved here because
    they are what the saved per-cell outputs correspond to.
  * Prompt caching is active: calls sharing a long identical PREFIX bill ~13
    fresh input tokens instead of ~3,500. Every prompt is emitted as
    [STABLE_PREAMBLE][item text], so the overhead amortizes across a shard.
  * Each cell is written to its own parquet as soon as it finishes, so a frame
    that dies mid-shard loses at most one cell, and re-running skips finished
    cells.
"""
from __future__ import annotations
import json, os, sys, time
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) or ".")
import lamp_harness as LH

# Stable, cacheable preamble. Also suppresses a real failure mode seen on
# LaMP-1, where the model emitted web-API code to look the answer up instead of
# answering. That failure concentrates at low k -- exactly where the signal axis
# lives -- so leaving it in would manufacture a spurious k-gradient.
STABLE_PREAMBLE = (
    "You are completing a personalization benchmark item offline.\n"
    "You have no internet access, no tools, and no ability to run code.\n"
    "Answer directly from the text provided in this prompt alone.\n"
    "Do not write code. Do not describe how you would look the answer up.\n"
    "Do not ask clarifying questions. Always commit to a single best answer,\n"
    "even when the evidence provided is thin or absent.\n"
    "----------------------------------------------------------------\n"
)

MAX_TOKENS = {"direct": 12, "cot": 400}


def subsample(df, n):
    """Deterministic subsample preserving exact stratum balance."""
    if n >= len(df):
        return df.reset_index(drop=True)
    per = n // df["stratum"].nunique()
    return (df.sort_values("qid").groupby("stratum", group_keys=False)
              .head(per).reset_index(drop=True))


def build_calls(df, task, k, variant, retriever="bm25"):
    calls = []
    for _, r in df.iterrows():
        if k > 0:
            ret = (LH.bm25_topk(r["query_text"], r["profile"], k, task=task)
                   if retriever == "bm25" else LH.recency_topk(r["profile"], k))
        else:
            ret = []
        body = LH.build_prompt(task, r["input"], ret, variant)
        calls.append({
            "qid": r["qid"], "stratum": r["stratum"], "profile_len": int(r["profile_len"]),
            "gold": r["gold"], "k": k, "effective_k": len(ret), "retriever": retriever,
            "variant": variant, "prompt": STABLE_PREAMBLE + body,
            "prompt_chars": len(STABLE_PREAMBLE + body),
        })
    return calls


def run_cell(backend, task, calls, model, variant, samples, temperature, concurrency):
    reqs, meta = [], []
    for c in calls:
        for s in range(samples):
            req = {"prompt": c["prompt"], "model": model, "max_tokens": MAX_TOKENS[variant]}
            # temperature is REJECTED by the claude-*-5 generation; only send it
            # when asked for (sampling conditions must use 4-5 models).
            if temperature is not None:
                req["temperature"] = temperature
            reqs.append(req)
            meta.append((c, s))

    out = backend.complete_batch(reqs, max_concurrency=concurrency)

    recs, errs = [], 0
    for (c, s), r in zip(meta, out):
        rec = {kk: vv for kk, vv in c.items() if kk != "prompt"}
        rec.update({"task": task, "model": model, "sample_idx": s,
                    "temperature": temperature if temperature is not None else float("nan")})
        if isinstance(r, dict) and r.get("text") is not None:
            rec["raw_text"] = r["text"]
            rec["parsed"] = LH.parse_prediction(task, r["text"])
            u = r.get("usage") or {}
            rec["input_tokens"] = u.get("input_tokens")
            rec["cache_read_input_tokens"] = u.get("cache_read_input_tokens")
            rec["output_tokens"] = u.get("output_tokens")
            rec["api_error"] = None
        else:
            errs += 1
            rec["raw_text"] = rec["parsed"] = None
            rec["input_tokens"] = rec["cache_read_input_tokens"] = rec["output_tokens"] = None
            rec["api_error"] = str(r)[:300]
        recs.append(rec)
    return recs, errs


def run_shard(backend, task, subset_path, model, ks, variant="direct", retriever="bm25",
              samples=1, temperature=None, n=100, concurrency=10, outdir="out", tag=""):
    """Run one shard. Returns list of parquet paths written.

    `backend` is any object implementing llm_backend.LLMBackend.
    """
    os.makedirs(outdir, exist_ok=True)
    df = subsample(LH.load_subset(subset_path), n)
    written = []
    for k in ks:
        stem = (f"{task}__{model}__k{k}__{variant}__s{samples}"
                f"__{retriever}{tag}").replace("/", "_")
        path = os.path.join(outdir, stem + ".parquet")
        if os.path.exists(path):
            print(f"[skip] {stem}", flush=True)
            written.append(path)
            continue
        calls = build_calls(df, task, k, variant, retriever)
        t0 = time.time()
        recs, errs = run_cell(backend, task, calls, model, variant, samples,
                              temperature, concurrency)
        rdf = pd.DataFrame(recs)
        rdf.to_parquet(path, index=False)
        tok = float(pd.to_numeric(rdf["input_tokens"], errors="coerce").fillna(0).sum())
        print(f"[done] {stem} n={len(rdf)} parsed={int(rdf['parsed'].notna().sum())} "
              f"api_err={errs} {time.time()-t0:.0f}s in_tok={tok:.0f}", flush=True)
        written.append(path)
    return written
