"""lamp_harness.py — retrieval, prompting, parsing and metrics for LaMP-1/2/3/5.

Self-contained: standard library only (no nltk / rouge-score / sklearn needed).
Every metric is unit-tested in `_selftest()` against hand-computed cases; run
`python lamp_harness.py` to execute the assertions.

Task conventions used throughout. The profile schemas below were measured by
inspecting the downloaded dev dumps (key sets tallied over >=400 queries per
task; 100% homogeneous within each task) AFTER an initial draft of this file
had guessed them -- the LaMP-2 shape in that draft was wrong and is corrected
here. Source split for each task is recorded in the eval-subset parquet's
`source_split` column.

  LaMP-1  citation identification, from LaMP_1/dev. `input` embeds the target
          paper title and two candidate references "[1]" / "[2]"; gold output
          is the string "[1]" or "[2]". Profile items: {id, title, abstract}.
  LaMP-2  movie tagging, from LaMP_2/NEW/dev. `input` embeds a movie
          description; gold output is one tag from a closed 15-way set.
          Profile items: {id, tag, description} -- there is NO `title` key on
          this task. Note that LaMP_2/dev (without /new/) is a DIFFERENT task
          (news categorization, profile items {id, title, text, category});
          this module targets the movie-tagging variant.
  LaMP-3  product rating, from LaMP_3/dev. `input` embeds a review; gold
          output is a digit string "1".."5". Profile items: {id, text, score}.
  LaMP-5  scholarly title generation, from LaMP_5/dev. `input` embeds a paper
          abstract; gold output is the free-text title. Profile items:
          {id, title, abstract}.

Retrieval operates over the raw profile list; each task supplies the field(s)
that make up an item's retrievable text via PROFILE_TEXT_FIELDS.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any, Callable, Dict, Iterable, List, Sequence, Tuple

# --------------------------------------------------------------------------
# task registry
# --------------------------------------------------------------------------

TASKS = ("LaMP-1", "LaMP-2", "LaMP-3", "LaMP-5")

# Which profile keys carry retrievable text, in the order they are concatenated.
PROFILE_TEXT_FIELDS: Dict[str, Tuple[str, ...]] = {
    "LaMP-1": ("title", "abstract"),
    "LaMP-2": ("description",),   # verified: movie profile items are {id, tag, description}
    "LaMP-3": ("text",),
    "LaMP-5": ("abstract", "title"),
}

# LaMP-2 closed label set (populated from the dev gold distribution at load
# time by `set_lamp2_labels`; the default below is the standard 15-tag set that
# the dev split was verified to use).
LAMP2_TAGS: List[str] = [
    "sci-fi", "based on a book", "comedy", "action", "twist ending",
    "dystopia", "dark comedy", "classic", "psychology", "fantasy",
    "romance", "thought-provoking", "social commentary", "violence",
    "true story",
]


def set_lamp2_labels(tags: Sequence[str]) -> None:
    """Override the LaMP-2 tag set with the labels actually observed in gold."""
    global LAMP2_TAGS
    LAMP2_TAGS = list(tags)


# --------------------------------------------------------------------------
# tokenization
# --------------------------------------------------------------------------

_WORD_RE = re.compile(r"[a-z0-9]+")

# A small, fixed stopword list. Kept inline so the harness has no data
# dependency and results are reproducible across machines.
STOPWORDS = frozenset("""
a an the and or but if while of to in on at by for with from as is are was were
be been being this that these those it its into over under then than so such
no not do does did doing have has had having i you he she they we them his her
their our your my me us who whom which what when where why how all any both
each few more most other some only own same too very can will just should now
""".split())


def tokenize(text: str, remove_stopwords: bool = True) -> List[str]:
    """Lowercase alphanumeric tokenization; optional stopword removal."""
    toks = _WORD_RE.findall((text or "").lower())
    if remove_stopwords:
        toks = [t for t in toks if t not in STOPWORDS]
    return toks


def profile_item_text(task: str, item: Dict[str, Any]) -> str:
    """Concatenate the retrievable text fields of one profile item."""
    fields = PROFILE_TEXT_FIELDS[task]
    return " ".join(str(item.get(f, "")) for f in fields).strip()


# --------------------------------------------------------------------------
# retrieval
# --------------------------------------------------------------------------

def bm25_topk(
    query_text: str,
    profile: Sequence[Dict[str, Any]],
    k: int,
    task: str = "LaMP-3",
    k1: float = 1.5,
    b: float = 0.75,
) -> List[Dict[str, Any]]:
    """Okapi BM25 retrieval over a user profile.

    Returns the top-`k` profile items, highest score first. Ties break by
    original profile order (stable), so the function is deterministic. `k <= 0`
    returns [] (the no-signal condition). `k >= len(profile)` returns the whole
    profile in BM25 order.
    """
    if k <= 0 or not profile:
        return []

    docs = [tokenize(profile_item_text(task, it)) for it in profile]
    N = len(docs)
    dls = [len(d) for d in docs]
    avgdl = (sum(dls) / N) if N else 0.0

    df: Counter = Counter()
    for d in docs:
        df.update(set(d))

    idf = {
        t: math.log(1.0 + (N - n + 0.5) / (n + 0.5))
        for t, n in df.items()
    }

    q = tokenize(query_text)
    qtf = Counter(q)

    scored: List[Tuple[float, int]] = []
    for i, d in enumerate(docs):
        if not d:
            scored.append((0.0, i))
            continue
        tf = Counter(d)
        dl = dls[i]
        s = 0.0
        for t, qn in qtf.items():
            f = tf.get(t)
            if not f:
                continue
            denom = f + k1 * (1.0 - b + b * (dl / avgdl if avgdl else 0.0))
            s += idf.get(t, 0.0) * qn * (f * (k1 + 1.0)) / denom
        scored.append((s, i))

    order = sorted(range(N), key=lambda i: (-scored[i][0], i))
    return [profile[i] for i in order[:k]]


def recency_topk(
    profile: Sequence[Dict[str, Any]],
    k: int,
    date_key: str | None = None,
) -> List[Dict[str, Any]]:
    """Most-recent-k retrieval.

    LaMP profiles are stored in chronological order and (for the tasks used
    here) carry no date field, so "recent" == the tail of the profile list.
    Returned newest-first. If `date_key` is given and present on every item,
    items are sorted by that key descending instead (lexicographic on ISO
    dates, which is order-preserving).
    """
    if k <= 0 or not profile:
        return []
    if date_key and all(date_key in it for it in profile):
        idx = sorted(range(len(profile)),
                     key=lambda i: (str(profile[i][date_key]), i), reverse=True)
        return [profile[i] for i in idx[:k]]
    return list(reversed(list(profile)[-k:]))


RETRIEVERS: Dict[str, Callable[..., List[Dict[str, Any]]]] = {
    "bm25": bm25_topk,
    "recency": recency_topk,
}


def retrieve(
    retriever: str,
    query_text: str,
    profile: Sequence[Dict[str, Any]],
    k: int,
    task: str = "LaMP-3",
) -> List[Dict[str, Any]]:
    """Dispatch to a named retriever with a uniform signature."""
    if retriever == "bm25":
        return bm25_topk(query_text, profile, k, task=task)
    if retriever == "recency":
        return recency_topk(profile, k)
    raise ValueError(f"unknown retriever {retriever!r}")


# --------------------------------------------------------------------------
# prompt construction
# --------------------------------------------------------------------------

_MAX_ITEM_CHARS = 1200  # truncate individual profile items to bound prompt size

COT_SUFFIX = {
    "LaMP-1": (
        "Think step by step about which reference matches this author's own "
        "citing habits, then give your final answer on the last line in the "
        "exact form 'Answer: [1]' or 'Answer: [2]'."
    ),
    "LaMP-2": (
        "Think step by step about how this user tags movies, then give your "
        "final answer on the last line in the exact form 'Answer: <tag>'."
    ),
    "LaMP-3": (
        "Think step by step about how harshly or generously this user rates, "
        "then give your final answer on the last line in the exact form "
        "'Answer: <integer 1-5>'."
    ),
    "LaMP-5": (
        "Think step by step about this author's titling style, then give the "
        "final title on the last line in the exact form 'Answer: <title>'."
    ),
}

DIRECT_SUFFIX = {
    "LaMP-1": "Respond with exactly '[1]' or '[2]' and nothing else.",
    "LaMP-2": "Respond with exactly one tag from the list and nothing else.",
    "LaMP-3": "Respond with a single integer from 1 to 5 and nothing else.",
    "LaMP-5": "Respond with only the title, on one line, and nothing else.",
}


def _fmt_item(task: str, item: Dict[str, Any], n: int) -> str:
    """Render one retrieved profile item for the prompt."""
    if task == "LaMP-3":
        txt = str(item.get("text", ""))[:_MAX_ITEM_CHARS]
        return f"{n}. [rating {item.get('score')}] {txt}"
    if task == "LaMP-2":
        # Movie profile items carry only {id, tag, description} -- no title.
        desc = str(item.get("description", ""))[:_MAX_ITEM_CHARS]
        return f"{n}. [tag: {item.get('tag')}] description: {desc}"
    # LaMP-1 / LaMP-5: title + abstract
    abst = str(item.get("abstract", ""))[:_MAX_ITEM_CHARS]
    return f"{n}. title: {item.get('title')} | abstract: {abst}"


def build_prompt(
    task: str,
    question_input: str,
    retrieved: Sequence[Dict[str, Any]],
    variant: str = "direct",
) -> str:
    """Build a plain-string prompt for one query.

    `question_input` is the LaMP `input` field verbatim (it already contains
    the task instruction and the payload). `retrieved` is the output of a
    retriever; an empty list yields the k=0 / no-profile prompt. `variant` is
    "direct" or "cot".
    """
    if task not in TASKS:
        raise ValueError(f"unknown task {task!r}")
    if variant not in ("direct", "cot"):
        raise ValueError(f"unknown variant {variant!r}")

    parts: List[str] = []
    if retrieved:
        parts.append(
            f"Here are {len(retrieved)} items from this user's history, most "
            "relevant first:"
        )
        parts.extend(_fmt_item(task, it, i + 1)
                     for i, it in enumerate(retrieved))
        parts.append("")
        parts.append("Now complete the following task for the SAME user.")
    else:
        parts.append(
            "No history is available for this user. Complete the following "
            "task."
        )
    parts.append("")
    parts.append(question_input.strip())

    if task == "LaMP-2":
        parts.append("")
        parts.append("Allowed tags: " + ", ".join(LAMP2_TAGS) + ".")

    parts.append("")
    parts.append(COT_SUFFIX[task] if variant == "cot" else DIRECT_SUFFIX[task])
    return "\n".join(parts)


# --------------------------------------------------------------------------
# prediction parsing
# --------------------------------------------------------------------------

_ANSWER_RE = re.compile(
    # Allow markdown/bullet decoration before the marker ("**Answer:**",
    # "- Final answer:", "### Answer -") and after the colon.
    r"(?:^|\n)[\s*_#>\-`]*(?:final\s+)?answer[\s*_`]*[:\-][\s*_`]*(.+)",
    re.IGNORECASE)


def _last_answer_span(raw: str) -> str:
    """Return the text after the last 'Answer:' marker, else the last
    non-empty line. Strips markdown emphasis and trailing punctuation."""
    matches = list(_ANSWER_RE.finditer(raw))
    if matches:
        seg = matches[-1].group(1)
    else:
        lines = [ln.strip() for ln in raw.strip().splitlines() if ln.strip()]
        seg = lines[-1] if lines else ""
    seg = seg.strip()
    seg = re.sub(r"^[*_`\s]+|[*_`\s]+$", "", seg)
    return seg.strip()


def parse_prediction(task: str, raw_text: str) -> str | None:
    """Extract a task-valid prediction from a raw model response.

    Robust to CoT preambles, "Answer:"/"Final answer:" markers, markdown
    emphasis, code fences, and trailing prose. Returns None when nothing
    valid can be recovered (callers must count these as parse failures rather
    than silently scoring them).
    """
    if raw_text is None:
        return None
    raw = str(raw_text).replace("```", " ")
    if not raw.strip():
        return None
    seg = _last_answer_span(raw)

    if task == "LaMP-1":
        # Prefer the bracketed form anywhere in the answer span, then the
        # whole response, then a bare 1/2.
        for hay in (seg, raw):
            m = re.findall(r"\[\s*([12])\s*\]", hay)
            if m:
                return f"[{m[-1]}]"
        m = re.fullmatch(r"[^0-9]*([12])[^0-9]*", seg)
        return f"[{m.group(1)}]" if m else None

    if task == "LaMP-3":
        for hay in (seg, raw):
            if not hay.strip():
                continue
            # 1. the whole span is just the rating
            m = re.fullmatch(r"[^0-9]*([1-5])[^0-9]*", hay.strip())
            if m:
                return m.group(1)
            # 2. explicit "X out of 5" / "X/5" / "X stars" -> take X, not the 5
            m = re.search(
                r"\b([1-5])\s*(?:/\s*5|out\s+of\s+5|stars?\b|\-?\s*star\b)",
                hay, re.IGNORECASE)
            if m:
                return m.group(1)
            # 3. "rating/score/rate ... X"
            m = re.search(r"(?:rating|score|rate[sd]?)\D{0,12}?([1-5])\b",
                          hay, re.IGNORECASE)
            if m:
                return m.group(1)
            # 4. fall back to the first in-range digit in the span
            m = re.search(r"\b([1-5])\b", hay)
            if m:
                return m.group(1)
        return None

    if task == "LaMP-2":
        low = seg.lower()
        # exact match first, then longest containing tag (so "dark comedy"
        # wins over "comedy"), then scan the whole response.
        for t in LAMP2_TAGS:
            if low == t:
                return t
        for hay in (low, raw.lower()):
            hits = [t for t in LAMP2_TAGS if t in hay]
            if hits:
                return max(hits, key=len)
        return None

    if task == "LaMP-5":
        # Free text: drop a leading "Title:" label, collapse whitespace,
        # strip wrapping quotes.
        # Strip a residual "Answer:"/"Title:" label that survived the span
        # extraction (e.g. when it arrived wrapped in markdown emphasis).
        seg = re.sub(r"^[\s*_`#>\-]*(?:final\s+)?answer[\s*_`]*[:\-]\s*", "",
                     seg, flags=re.IGNORECASE)
        seg = re.sub(r"^\s*title\s*[:\-]\s*", "", seg, flags=re.IGNORECASE)
        seg = re.sub(r"\s+", " ", seg).strip().strip('"').strip("'").strip()
        return seg or None

    raise ValueError(f"unknown task {task!r}")


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------

def mae(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    assert len(y_true) == len(y_pred) and y_true, "empty or mismatched inputs"
    return sum(abs(a - b) for a, b in zip(y_true, y_pred)) / len(y_true)


def rmse(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    assert len(y_true) == len(y_pred) and y_true, "empty or mismatched inputs"
    return math.sqrt(
        sum((a - b) ** 2 for a, b in zip(y_true, y_pred)) / len(y_true))


def accuracy(y_true: Sequence[Any], y_pred: Sequence[Any]) -> float:
    assert len(y_true) == len(y_pred) and y_true, "empty or mismatched inputs"
    return sum(1 for a, b in zip(y_true, y_pred) if a == b) / len(y_true)


def macro_f1(y_true: Sequence[Any], y_pred: Sequence[Any],
             labels: Sequence[Any] | None = None) -> float:
    """Unweighted mean per-class F1 over `labels` (default: labels present in
    y_true or y_pred). A class with no predictions and no gold contributes
    F1 = 0, matching sklearn's zero_division=0 default."""
    assert len(y_true) == len(y_pred) and y_true, "empty or mismatched inputs"
    if labels is None:
        labels = sorted(set(y_true) | set(y_pred), key=str)
    f1s = []
    for c in labels:
        tp = sum(1 for a, b in zip(y_true, y_pred) if a == c and b == c)
        fp = sum(1 for a, b in zip(y_true, y_pred) if a != c and b == c)
        fn = sum(1 for a, b in zip(y_true, y_pred) if a == c and b != c)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if (prec + rec) else 0.0)
    return sum(f1s) / len(f1s) if f1s else 0.0


# ---- ROUGE (own implementation) ------------------------------------------

def _rouge_tokens(text: str) -> List[str]:
    """ROUGE tokenization: lowercase, non-alphanumeric -> space. Matches
    google-research/rouge_score's default (no stopword removal, no stemming)."""
    return re.sub(r"[^a-z0-9]+", " ", (text or "").lower()).split()


def _prf(match: int, n_pred: int, n_ref: int) -> Dict[str, float]:
    p = match / n_pred if n_pred else 0.0
    r = match / n_ref if n_ref else 0.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return {"precision": p, "recall": r, "fmeasure": f}


def rouge_1(prediction: str, reference: str) -> Dict[str, float]:
    """ROUGE-1: unigram overlap with clipped (multiset) counts."""
    p_t, r_t = _rouge_tokens(prediction), _rouge_tokens(reference)
    pc, rc = Counter(p_t), Counter(r_t)
    match = sum(min(pc[t], rc[t]) for t in pc.keys() & rc.keys())
    return _prf(match, len(p_t), len(r_t))


def _lcs_len(a: Sequence[str], b: Sequence[str]) -> int:
    """Length of the longest common subsequence (O(len(a)*len(b)) time,
    O(min) space)."""
    if not a or not b:
        return 0
    if len(a) < len(b):
        a, b = b, a
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b, 1):
            cur.append(prev[j - 1] + 1 if x == y else max(prev[j], cur[j - 1]))
        prev = cur
    return prev[-1]


def rouge_l(prediction: str, reference: str) -> Dict[str, float]:
    """ROUGE-L: LCS-based F-measure over the whole sequence (sentence-level,
    matching rouge_score's 'rougeL')."""
    p_t, r_t = _rouge_tokens(prediction), _rouge_tokens(reference)
    return _prf(_lcs_len(p_t, r_t), len(p_t), len(r_t))


def rouge_scores(predictions: Sequence[str],
                 references: Sequence[str]) -> Dict[str, float]:
    """Corpus-level ROUGE: mean of per-example F-measures (LaMP convention)."""
    assert len(predictions) == len(references) and predictions, \
        "empty or mismatched inputs"
    r1 = [rouge_1(p, r)["fmeasure"] for p, r in zip(predictions, references)]
    rl = [rouge_l(p, r)["fmeasure"] for p, r in zip(predictions, references)]
    return {"rouge_1": sum(r1) / len(r1), "rouge_L": sum(rl) / len(rl)}


# ---- per-task scorers ----------------------------------------------------

def score_task(
    task: str,
    raw_predictions: Sequence[str],
    golds: Sequence[str],
    already_parsed: bool = False,
) -> Dict[str, float]:
    """Score one task from raw model outputs (or pre-parsed predictions).

    Parse failures are reported in `parse_failures` / `n_parsed` and are
    EXCLUDED from the metric averages, so a metric is never silently
    contaminated by unparseable rows. For LaMP-3 a failure also cannot be
    scored as a rating, so callers reporting headline numbers should quote
    both the metric and the failure count.
    """
    assert len(raw_predictions) == len(golds) and raw_predictions, \
        "empty or mismatched inputs"
    preds = list(raw_predictions) if already_parsed else [
        parse_prediction(task, r) for r in raw_predictions]
    keep = [i for i, p in enumerate(preds) if p is not None]
    fails = len(preds) - len(keep)
    out: Dict[str, float] = {
        "n": float(len(preds)),
        "n_parsed": float(len(keep)),
        "parse_failures": float(fails),
        "parse_failure_rate": fails / len(preds),
    }
    if not keep:
        return out
    P = [preds[i] for i in keep]
    G = [golds[i] for i in keep]

    if task == "LaMP-3":
        pi = [float(int(p)) for p in P]
        gi = [float(int(g)) for g in G]
        out.update(mae=mae(gi, pi), rmse=rmse(gi, pi),
                   accuracy=accuracy([int(g) for g in gi],
                                     [int(p) for p in pi]))
    elif task in ("LaMP-1", "LaMP-2"):
        labels = (["[1]", "[2]"] if task == "LaMP-1"
                  else sorted(set(G) | set(P), key=str))
        out.update(accuracy=accuracy(G, P), macro_f1=macro_f1(G, P, labels))
    elif task == "LaMP-5":
        out.update(rouge_scores(P, G))
    else:
        raise ValueError(f"unknown task {task!r}")
    return out


# --------------------------------------------------------------------------
# eval-subset loading (for sweep tracks)
# --------------------------------------------------------------------------

# Payload separators verified against the dev `input` strings of each task.
_PAYLOAD_SEP = {
    "LaMP-1": None,          # no clean separator; the whole input is the query
    "LaMP-2": "tags: [",     # description follows the closing bracket
    "LaMP-3": "review:",
    "LaMP-5": "abstract of a paper:",   # verified wording; NOT "abstract:"
}


def query_text(task: str, question_input: str) -> str:
    """Extract the retrieval query (the payload, instruction stripped).

    Using the full `input` as a BM25 query would let the shared instruction
    boilerplate and, for LaMP-2, the tag list dominate the term statistics, so
    the payload is isolated per task.
    """
    sep = _PAYLOAD_SEP.get(task)
    if sep is None:
        # LaMP-1: drop the leading instruction clause, keep title + candidates.
        m = re.search(r'the title\s*"(.*)$', question_input, re.DOTALL)
        return m.group(1) if m else question_input
    if sep in question_input:
        tail = question_input.split(sep, 1)[1]
        if task == "LaMP-2":
            tail = tail.split("]", 1)[-1]
        return tail.strip()
    return question_input


def load_subset(path: str):
    """Load one eval-subset parquet. Returns a pandas DataFrame with the
    `profile` column decoded from JSON into a list of dicts.

    Requires pandas; kept out of the module's import-time dependencies so the
    metric/retrieval core stays standard-library only.
    """
    import json as _json

    import pandas as _pd

    df = _pd.read_parquet(path)
    df["profile"] = df["profile_json"].map(_json.loads)
    df["query_text"] = [query_text(t, i) for t, i in zip(df["task"], df["input"])]
    return df


# --------------------------------------------------------------------------
# unit tests
# --------------------------------------------------------------------------

def _close(a: float, b: float, tol: float = 1e-9) -> bool:
    return abs(a - b) <= tol


def _selftest() -> None:
    checks: List[str] = []

    def ok(name: str, cond: bool) -> None:
        assert cond, f"FAILED: {name}"
        checks.append(name)

    # ---- regression metrics, hand-computed -----------------------------
    # errors: |3-1|=2, |1-2|=1, |5-5|=0, |2-4|=2  -> sum 5, MAE 5/4 = 1.25
    yt, yp = [3, 1, 5, 2], [1, 2, 5, 4]
    ok("mae hand-computed = 1.25", _close(mae(yt, yp), 1.25))
    # squares: 4,1,0,4 -> 9/4 = 2.25 -> sqrt = 1.5
    ok("rmse hand-computed = 1.5", _close(rmse(yt, yp), 1.5))
    ok("accuracy hand-computed = 0.25", _close(accuracy(yt, yp), 0.25))
    ok("mae zero on identical", _close(mae(yt, yt), 0.0))
    ok("rmse >= mae", rmse(yt, yp) >= mae(yt, yp))

    # ---- macro F1, hand-computed ---------------------------------------
    # gold: a a b b c ; pred: a b b b c
    # a: tp1 fp0 fn1 -> P=1  R=.5  F=2/3
    # b: tp2 fp1 fn0 -> P=2/3 R=1  F=0.8
    # c: tp1 fp0 fn0 -> F=1
    # macro = (2/3 + 0.8 + 1)/3 = 0.8222...
    g = ["a", "a", "b", "b", "c"]
    p = ["a", "b", "b", "b", "c"]
    ok("macro_f1 hand-computed = 0.822222",
       _close(macro_f1(g, p), (2 / 3 + 0.8 + 1.0) / 3))
    ok("accuracy of that case = 0.8", _close(accuracy(g, p), 0.8))
    ok("macro_f1 perfect = 1.0", _close(macro_f1(g, g), 1.0))
    # a class present in labels but absent from data contributes 0
    ok("macro_f1 unused label contributes 0",
       _close(macro_f1(["a", "a"], ["a", "a"], labels=["a", "z"]), 0.5))

    # ---- ROUGE-1, hand-computed ---------------------------------------
    # pred "the cat sat on the mat" (6 tok), ref "the cat sat on a mat" (6)
    # clipped unigram matches: the(1 of 2 in pred vs 1 in ref -> 1), cat 1,
    # sat 1, on 1, mat 1  => 5 ; P = R = 5/6 ; F = 5/6
    r1 = rouge_1("the cat sat on the mat", "the cat sat on a mat")
    ok("rouge_1 precision = 5/6", _close(r1["precision"], 5 / 6))
    ok("rouge_1 recall = 5/6", _close(r1["recall"], 5 / 6))
    ok("rouge_1 fmeasure = 5/6", _close(r1["fmeasure"], 5 / 6))
    # clipping: repeated pred token must not double-count a single ref token
    r1c = rouge_1("cat cat", "cat")
    ok("rouge_1 clips duplicate matches (P=0.5)",
       _close(r1c["precision"], 0.5) and _close(r1c["recall"], 1.0))
    ok("rouge_1 disjoint = 0", _close(rouge_1("abc", "xyz")["fmeasure"], 0.0))
    ok("rouge_1 identical = 1",
       _close(rouge_1("a b c", "a b c")["fmeasure"], 1.0))

    # ---- ROUGE-L, hand-computed ---------------------------------------
    # pred "a b c d" / ref "a c b d": LCS = "a b d" or "a c d" -> 3
    # P = R = 3/4 -> F = 0.75
    rl = rouge_l("a b c d", "a c b d")
    ok("rouge_l LCS hand-computed F = 0.75", _close(rl["fmeasure"], 0.75))
    ok("lcs_len direct check = 3",
       _lcs_len(["a", "b", "c", "d"], ["a", "c", "b", "d"]) == 3)
    # order sensitivity: ROUGE-L must be < ROUGE-1 on a pure reordering
    perm_1 = rouge_1("a b c d", "d c b a")["fmeasure"]
    perm_l = rouge_l("a b c d", "d c b a")["fmeasure"]
    ok("rouge_1 order-insensitive on permutation = 1.0", _close(perm_1, 1.0))
    ok("rouge_l order-sensitive on permutation (0.25)", _close(perm_l, 0.25))
    ok("rouge_l identical = 1",
       _close(rouge_l("x y z", "x y z")["fmeasure"], 1.0))
    # tokenization: punctuation and case must not matter
    ok("rouge tokenization ignores case/punct",
       _close(rouge_1("The Cat, sat!", "the cat sat")["fmeasure"], 1.0))

    # ---- parse_prediction ---------------------------------------------
    ok("L3 bare digit", parse_prediction("LaMP-3", "4") == "4")
    ok("L3 CoT preamble",
       parse_prediction("LaMP-3", "The user rates 5s often.\nAnswer: 2") == "2")
    ok("L3 final-answer marker",
       parse_prediction("LaMP-3", "blah\nFinal answer: 3") == "3")
    ok("L3 markdown emphasis",
       parse_prediction("LaMP-3", "Answer: **5**") == "5")
    ok("L3 'X out of 5' takes X not the scale max",
       parse_prediction("LaMP-3", "I would rate this a 4 out of 5 stars.")
       == "4")
    ok("L3 'X/5' form", parse_prediction("LaMP-3", "Answer: 2/5") == "2")
    ok("L3 'X stars' form",
       parse_prediction("LaMP-3", "Probably 3 stars.") == "3")
    ok("L3 'rating: X' form",
       parse_prediction("LaMP-3", "My rating would be 1.") == "1")
    ok("L3 multi-line CoT with distractor digits",
       parse_prediction(
           "LaMP-3",
           "They gave 5 and 4 in the past, on a 1-5 scale.\nAnswer: 3") == "3")
    ok("L3 rejects out-of-range only",
       parse_prediction("LaMP-3", "9") is None)
    ok("L3 empty -> None", parse_prediction("LaMP-3", "") is None)
    ok("L3 None -> None", parse_prediction("LaMP-3", None) is None)

    ok("L1 bracket", parse_prediction("LaMP-1", "[2]") == "[2]")
    ok("L1 CoT",
       parse_prediction("LaMP-1", "Reasoning...\nAnswer: [1]") == "[1]")
    ok("L1 bare digit", parse_prediction("LaMP-1", "1") == "[1]")
    ok("L1 spaced bracket", parse_prediction("LaMP-1", "[ 2 ]") == "[2]")
    ok("L1 junk -> None", parse_prediction("LaMP-1", "neither") is None)

    ok("L2 exact tag", parse_prediction("LaMP-2", "comedy") == "comedy")
    ok("L2 longest-match beats substring",
       parse_prediction("LaMP-2", "Answer: dark comedy") == "dark comedy")
    ok("L2 CoT with prose",
       parse_prediction("LaMP-2", "Let me think.\nAnswer: sci-fi") == "sci-fi")
    ok("L2 case-insensitive", parse_prediction("LaMP-2", "Action") == "action")
    ok("L2 junk -> None", parse_prediction("LaMP-2", "zzz") is None)

    ok("L5 title label stripped",
       parse_prediction("LaMP-5", "Title: A Study of Things")
       == "A Study of Things")
    ok("L5 CoT marker",
       parse_prediction("LaMP-5", "thinking...\nAnswer: Deep Nets")
       == "Deep Nets")
    ok("L5 quotes stripped",
       parse_prediction("LaMP-5", '"Quoted Title"') == "Quoted Title")
    ok("L5 empty -> None", parse_prediction("LaMP-5", "   ") is None)
    # Regression: a real haiku-4-5 CoT response ended with the marker wrapped
    # in markdown bold, which a line-anchored regex missed and which leaked
    # "Answer: " into the scored title.
    ok("L5 markdown-bold answer marker (regression)",
       parse_prediction(
           "LaMP-5",
           "- reasoning bullet\n\n**Answer: When Superintelligences Play: "
           "Games, Ethics, and the Future**")
       == "When Superintelligences Play: Games, Ethics, and the Future")
    ok("L5 bolded label with colon inside emphasis",
       parse_prediction("LaMP-5", "**Answer:** Deep Nets Revisited")
       == "Deep Nets Revisited")
    ok("L5 bullet-prefixed marker",
       parse_prediction("LaMP-5", "- Final answer: A Short Title")
       == "A Short Title")
    ok("L5 no 'Answer' leaks into any parsed title",
       not parse_prediction(
           "LaMP-5", "**Answer: X Y Z**").lower().startswith("answer"))
    ok("L3 markdown-bold marker still parses",
       parse_prediction("LaMP-3", "reasoning\n**Answer: 4**") == "4")
    ok("L2 markdown-bold marker still parses",
       parse_prediction("LaMP-2", "reasoning\n**Answer: sci-fi**") == "sci-fi")
    ok("L1 markdown-bold marker still parses",
       parse_prediction("LaMP-1", "reasoning\n**Answer: [2]**") == "[2]")

    # ---- retrieval ----------------------------------------------------
    prof3 = [
        {"id": "a", "text": "the coffee maker leaked all over the counter",
         "score": "2"},
        {"id": "b", "text": "great espresso machine, love the coffee",
         "score": "5"},
        {"id": "c", "text": "this novel was a page turner", "score": "4"},
    ]
    top = bm25_topk("coffee espresso machine", prof3, 2, task="LaMP-3")
    ok("bm25 returns k items", len(top) == 2)
    ok("bm25 ranks the espresso item first", top[0]["id"] == "b")
    ok("bm25 excludes the unrelated novel item",
       "c" not in [t["id"] for t in top])
    ok("bm25 k=0 -> empty", bm25_topk("x", prof3, 0, task="LaMP-3") == [])
    ok("bm25 k>len returns whole profile",
       len(bm25_topk("coffee", prof3, 99, task="LaMP-3")) == 3)
    ok("bm25 empty profile -> empty", bm25_topk("x", [], 3) == [])
    ok("bm25 deterministic",
       [t["id"] for t in bm25_topk("coffee", prof3, 3, task="LaMP-3")]
       == [t["id"] for t in bm25_topk("coffee", prof3, 3, task="LaMP-3")])
    # no query overlap at all -> all zero scores -> stable original order
    ok("bm25 all-zero scores keep profile order",
       [t["id"] for t in bm25_topk("zzzz", prof3, 3, task="LaMP-3")]
       == ["a", "b", "c"])

    rec = recency_topk(prof3, 2)
    ok("recency newest-first from tail",
       [t["id"] for t in rec] == ["c", "b"])
    ok("recency k=0 -> empty", recency_topk(prof3, 0) == [])
    ok("recency k>len returns all", len(recency_topk(prof3, 99)) == 3)
    dated = [{"id": "x", "date": "2020-01-01"}, {"id": "y", "date": "2023-05-05"},
             {"id": "z", "date": "2021-01-01"}]
    ok("recency honours date_key when present",
       [t["id"] for t in recency_topk(dated, 2, date_key="date")] == ["y", "z"])
    ok("retrieve() dispatch matches bm25_topk",
       [t["id"] for t in retrieve("bm25", "coffee", prof3, 2, task="LaMP-3")]
       == [t["id"] for t in bm25_topk("coffee", prof3, 2, task="LaMP-3")])

    # ---- prompts ------------------------------------------------------
    p_direct = build_prompt("LaMP-3", "Rate this review: it was fine",
                            top, variant="direct")
    p_cot = build_prompt("LaMP-3", "Rate this review: it was fine",
                         top, variant="cot")
    ok("prompt is a plain string", isinstance(p_direct, str))
    ok("direct prompt carries the payload",
       "Rate this review: it was fine" in p_direct)
    ok("direct prompt embeds retrieved ratings", "[rating 5]" in p_direct)
    ok("direct prompt states the format constraint",
       "single integer" in p_direct)
    ok("cot prompt asks for step-by-step", "step by step" in p_cot)
    ok("cot prompt differs from direct", p_cot != p_direct)
    p_k0 = build_prompt("LaMP-3", "Rate this review: it was fine", [])
    ok("k=0 prompt says no history", "No history" in p_k0)
    ok("k=0 prompt has no history block", "[rating" not in p_k0)
    p2 = build_prompt("LaMP-2", "Which tag applies?", [], variant="direct")
    ok("LaMP-2 prompt lists the allowed tags", "Allowed tags:" in p2)
    for t in TASKS:
        for v in ("direct", "cot"):
            s = build_prompt(t, "payload", [], variant=v)
            ok(f"prompt builds for {t}/{v}", isinstance(s, str) and len(s) > 20)

    # ---- query_text payload extraction --------------------------------
    qt3 = query_text(
        "LaMP-3",
        "What is the score of the following review on a scale of 1 to 5? just "
        "answer with 1, 2, 3, 4, or 5 without further explanation. review: "
        "One was broken and unusable.")
    ok("query_text LaMP-3 strips instruction",
       qt3 == "One was broken and unusable." )
    qt5 = query_text(
        "LaMP-5",
        "Generate a title for the following abstract of a paper: We propose X.")
    ok("query_text LaMP-5 keeps the abstract", qt5 == "We propose X.")
    qt2 = query_text(
        "LaMP-2",
        "Which tag does this movie relate to among the following tags? Just "
        "answer with the tag name without further explanation. tags: "
        "[sci-fi, comedy] description: A princess tours Rome.")
    ok("query_text LaMP-2 drops the tag list", "sci-fi" not in qt2)
    ok("query_text LaMP-2 keeps the description", "princess" in qt2)
    qt1 = query_text(
        "LaMP-1",
        'For an author who has written the paper with the title "Visual audio '
        'integration", which reference is related? [1]: "A" [2]: "B"')
    ok("query_text LaMP-1 keeps title and candidates",
       "Visual audio" in qt1 and "[2]" in qt1)
    ok("query_text falls back to full input when separator absent",
       query_text("LaMP-3", "no separator here") == "no separator here")

    # ---- LaMP-2 profile shape (verified: {id, tag, description}, no title)
    prof2 = [{"id": "1", "tag": "sci-fi",
              "description": "a spaceship crew fights an alien"},
             {"id": "2", "tag": "comedy",
              "description": "two friends open a failing restaurant"}]
    ok("LaMP-2 retrievable text is the description",
       profile_item_text("LaMP-2", prof2[0]) == prof2[0]["description"])
    t2 = bm25_topk("alien spaceship", prof2, 1, task="LaMP-2")
    ok("LaMP-2 bm25 ranks the sci-fi item first", t2[0]["id"] == "1")
    p2r = build_prompt("LaMP-2", "Which tag?", t2, variant="direct")
    ok("LaMP-2 prompt shows the item tag", "[tag: sci-fi]" in p2r)
    ok("LaMP-2 prompt has no empty title field", "title: None" not in p2r)

    # ---- documented schema matches the code's field map ---------------
    # Guards against the docstring and PROFILE_TEXT_FIELDS drifting apart
    # (an earlier draft of this file documented a `title` key for LaMP-2 that
    # does not exist in the data).
    ok("LaMP-2 field map has no title", "title" not in PROFILE_TEXT_FIELDS["LaMP-2"])
    ok("LaMP-2 field map is description only",
       PROFILE_TEXT_FIELDS["LaMP-2"] == ("description",))
    ok("LaMP-3 field map is text only",
       PROFILE_TEXT_FIELDS["LaMP-3"] == ("text",))
    ok("LaMP-1/5 field maps use title+abstract",
       set(PROFILE_TEXT_FIELDS["LaMP-1"]) == {"title", "abstract"}
       and set(PROFILE_TEXT_FIELDS["LaMP-5"]) == {"title", "abstract"})
    ok("every task has a field map",
       all(t in PROFILE_TEXT_FIELDS for t in TASKS))
    ok("docstring documents the LaMP-2 no-title finding",
       "NO `title` key" in (__doc__ or ""))
    # A LaMP-2 item shaped like the (wrong) old guess must still not
    # contribute a title to retrievable text.
    ok("LaMP-2 ignores a stray title field if present",
       profile_item_text("LaMP-2",
                         {"id": "1", "tag": "comedy", "description": "d",
                          "title": "SHOULD_NOT_APPEAR"}) == "d")

    # ---- score_task end to end ---------------------------------------
    s3 = score_task("LaMP-3", ["Answer: 3", "2", "garbage"], ["3", "4", "5"])
    ok("score_task counts one parse failure", s3["parse_failures"] == 1.0)
    ok("score_task excludes failures from n_parsed", s3["n_parsed"] == 2.0)
    # parsed: pred 3 vs gold 3 (0), pred 2 vs gold 4 (2) -> MAE 1.0
    ok("score_task LaMP-3 MAE = 1.0", _close(s3["mae"], 1.0))
    ok("score_task LaMP-3 accuracy = 0.5", _close(s3["accuracy"], 0.5))
    s1 = score_task("LaMP-1", ["[1]", "[2]"], ["[1]", "[1]"])
    ok("score_task LaMP-1 accuracy = 0.5", _close(s1["accuracy"], 0.5))
    ok("score_task LaMP-1 reports macro_f1", "macro_f1" in s1)
    s5 = score_task("LaMP-5", ["a b c"], ["a b c"])
    ok("score_task LaMP-5 perfect ROUGE-1 = 1.0", _close(s5["rouge_1"], 1.0))
    ok("score_task LaMP-5 perfect ROUGE-L = 1.0", _close(s5["rouge_L"], 1.0))

    print(f"{len(checks)} assertions passed")
    for c in checks:
        print("  PASS", c)


if __name__ == "__main__":
    _selftest()
