"""Model-call boundary for the sweep.

The experiments in the paper were run through an internal batching harness whose
`complete_batch` contract is reproduced exactly below. That harness is not
public, so this module defines the interface and ships a reference
implementation against the public Anthropic API.

IMPORTANT, for anyone reproducing: the numbers in `results/` were produced with
the internal harness, NOT with `AnthropicBackend`. The two are intended to be
semantically identical -- same prompts, same decoding settings, same
request/response shape -- but `AnthropicBackend` was never exercised against the
live API in the run that produced those files, so treat it as a faithful
reference implementation rather than a bit-for-bit reproduction path. Expect
small differences from sampling nondeterminism and model-version drift even
where the settings match.

Contract
--------
`complete_batch(requests)` takes a list of dicts, each with:

    prompt       str   the full prompt text
    model        str   model identifier
    max_tokens   int   generation cap
    temperature  float OPTIONAL. When absent, the call must use the provider's
                       default greedy/near-greedy decoding. Do not substitute
                       0.0 -- during the original runs one model generation
                       REJECTED an explicit temperature argument, so "absent"
                       and "0.0" are not interchangeable.

and returns a list of the same length, positionally aligned, each element:

    {"text": str, "usage": {...}}      on success
    {"error": str}                     on failure (never raise for one item)

A failed item must not abort the batch: the runner records `api_error` per row
and reports per-cell error counts, which is how measurement losses stay visible
instead of silently shrinking the sample.
"""
from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor


class LLMBackend:
    """Interface the sweep runner depends on."""

    def complete_batch(self, requests: list[dict], max_concurrency: int = 8) -> list[dict]:
        raise NotImplementedError


class AnthropicBackend(LLMBackend):
    """Reference implementation over the public Anthropic Messages API.

    Requires `pip install anthropic` and ANTHROPIC_API_KEY in the environment.
    Concurrency is a simple thread pool; the original harness batched
    server-side, so throughput will differ.
    """

    def __init__(self, api_key: str | None = None, max_retries: int = 4):
        try:
            import anthropic
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "AnthropicBackend needs the `anthropic` package: pip install anthropic"
            ) from e
        key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise RuntimeError("Set ANTHROPIC_API_KEY or pass api_key=")
        self._client = anthropic.Anthropic(api_key=key, max_retries=max_retries)

    def _one(self, req: dict) -> dict:
        kwargs = {
            "model": req["model"],
            "max_tokens": req.get("max_tokens", 16),
            "messages": [{"role": "user", "content": req["prompt"]}],
        }
        # Only forward temperature when the caller set it (see module docstring).
        if req.get("temperature") is not None:
            kwargs["temperature"] = req["temperature"]
        try:
            r = self._client.messages.create(**kwargs)
            text = "".join(b.text for b in r.content if getattr(b, "type", None) == "text")
            return {
                "text": text,
                "usage": {
                    "input_tokens": r.usage.input_tokens,
                    "output_tokens": r.usage.output_tokens,
                    "cache_read_input_tokens": getattr(
                        r.usage, "cache_read_input_tokens", None
                    ),
                },
            }
        except Exception as e:  # one bad item must not kill the batch
            return {"error": f"{type(e).__name__}: {e}"[:300]}

    def complete_batch(self, requests: list[dict], max_concurrency: int = 8) -> list[dict]:
        if not requests:
            return []
        with ThreadPoolExecutor(max_workers=max_concurrency) as ex:
            return list(ex.map(self._one, requests))


class EchoBackend(LLMBackend):
    """Offline stand-in for testing the pipeline without any API calls.

    Returns a fixed parseable answer per task family so the runner, parser,
    scorer and checkpointing can be exercised end to end. Produces meaningless
    scores by construction -- never report numbers obtained from it.
    """

    def __init__(self, reply: str = "3"):
        self.reply = reply

    def complete_batch(self, requests: list[dict], max_concurrency: int = 8) -> list[dict]:
        return [
            {"text": self.reply, "usage": {"input_tokens": 0, "output_tokens": 0}}
            for _ in requests
        ]
