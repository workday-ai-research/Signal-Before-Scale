"""Analysis core for the signal x compute sweep.

Metrics come from lamp_harness (imported, not reimplemented). Two deliberate
deviations from `score_task`, both documented in analysis_report.md:

  1. LaMP-2 macro-F1 is computed with the label set PINNED to the closed
     15-tag set (`lh.LAMP2_TAGS`). `score_task` defaults to
     sorted(set(gold) | set(pred)), whose denominator changes with the
     predicted label set -- that makes macro-F1 non-comparable across cells
     and unstable under bootstrap resampling.
  2. Metrics are additionally computed per item so bootstraps over items are
     cheap. Mean-type metrics are verified to equal `score_task` exactly.
"""
import json, math

import numpy as np

import lamp_harness as lh

TASKS = ("LaMP-1", "LaMP-2", "LaMP-3", "LaMP-5")
MODELS = ("haiku", "sonnet", "opus")
# Headline metric per task and its direction (+1 = higher is better).
HEADLINE = {"LaMP-1": ("accuracy", +1), "LaMP-2": ("macro_f1", +1),
            "LaMP-3": ("mae", -1), "LaMP-5": ("rouge_1", +1)}
# The error scale used for the saturating fits: error = 0 is perfect.
# LaMP-3 MAE is already an error on 0..4; the others are 1 - metric on 0..1.
ERR_MAX = {"LaMP-1": 1.0, "LaMP-2": 1.0, "LaMP-3": 4.0, "LaMP-5": 1.0}
LABELS = {"LaMP-1": ["[1]", "[2]"], "LaMP-2": list(lh.LAMP2_TAGS)}


def item_scores(task, preds, golds):
    """Per-item metric contributions. None preds -> nan (parse failure).

    Returns dict of metric -> np.array(float) aligned with `preds`.
    """
    n = len(preds)
    ok = np.array([p is not None for p in preds])
    out = {}
    if task == "LaMP-3":
        pi = np.full(n, np.nan); gi = np.array([float(int(g)) for g in golds])
        for i, p in enumerate(preds):
            if p is not None:
                pi[i] = float(int(p))
        out["mae"] = np.abs(pi - gi)
        out["sqerr"] = (pi - gi) ** 2
        out["accuracy"] = np.where(ok, (pi == gi).astype(float), np.nan)
    elif task in ("LaMP-1", "LaMP-2"):
        out["accuracy"] = np.array([np.nan if p is None else float(p == g)
                                    for p, g in zip(preds, golds)])
    elif task == "LaMP-5":
        r1 = np.full(n, np.nan); rl = np.full(n, np.nan)
        for i, (p, g) in enumerate(zip(preds, golds)):
            if p is None:
                continue
            r1[i] = lh.rouge_1(p, g)["fmeasure"]
            rl[i] = lh.rouge_l(p, g)["fmeasure"]
        out["rouge_1"] = r1; out["rouge_L"] = rl
    else:
        raise ValueError(task)
    return out


def macro_f1_fixed(task, preds, golds, idx=None):
    """Macro-F1 over the pinned label set; parse failures dropped."""
    labels = LABELS[task]
    if idx is None:
        idx = range(len(preds))
    P, G = [], []
    for i in idx:
        if preds[i] is not None:
            P.append(preds[i]); G.append(golds[i])
    if not P:
        return float("nan")
    return lh.macro_f1(G, P, labels)


def macro_f1_factory(task, preds, golds):
    """Fast fn(idx)->macro-F1 over the pinned label set (bincount confusion).

    Verified elementwise against lh.macro_f1 in the verification cell.
    """
    labels = LABELS[task]; L = len(labels)
    code = {c: i for i, c in enumerate(labels)}
    keep = np.array([i for i, p in enumerate(preds) if p is not None])
    gp = np.array([code[golds[i]] * L + code[preds[i]] for i in keep], dtype=np.int64)
    pos = np.full(len(preds), -1, dtype=np.int64)
    pos[keep] = np.arange(len(keep))

    def fn(idx):
        sel = pos[np.asarray(idx)]
        sel = sel[sel >= 0]
        if sel.size == 0:
            return float("nan")
        cm = np.bincount(gp[sel], minlength=L * L).reshape(L, L)
        tp = np.diag(cm).astype(float)
        fp = cm.sum(0) - tp
        fn_ = cm.sum(1) - tp
        denom = 2 * tp + fp + fn_
        f1 = np.where(denom > 0, 2 * tp / np.where(denom > 0, denom, 1), 0.0)
        return float(f1.mean())
    return fn


def cell_metrics(task, preds, golds):
    """Point estimates for one cell (all metrics), plus n / parse failures."""
    n = len(preds)
    fails = sum(1 for p in preds if p is None)
    out = {"n": n, "n_parsed": n - fails, "parse_failures": fails,
           "parse_failure_rate": fails / n if n else float("nan")}
    it = item_scores(task, preds, golds)
    for m, v in it.items():
        if m == "sqerr":
            out["rmse"] = float(np.sqrt(np.nanmean(v)))
        else:
            out[m] = float(np.nanmean(v))
    if task in ("LaMP-1", "LaMP-2"):
        out["macro_f1"] = macro_f1_fixed(task, preds, golds)
    return out


def boot_ci(fn, n_items, n_boot=2000, seed=0, alpha=0.05):
    """Percentile bootstrap over item indices. fn(idx)->float (nan allowed)."""
    rng = np.random.default_rng(seed)
    vals = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n_items, n_items)
        vals[b] = fn(idx)
    vals = vals[~np.isnan(vals)]
    if vals.size < 100:
        return float("nan"), float("nan"), float("nan")
    lo, hi = np.percentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi), float(vals.std(ddof=1))


def metric_fn(task, metric, preds, golds):
    """Return fn(idx)->metric value, for bootstrapping."""
    if metric == "macro_f1":
        return lambda idx: macro_f1_fixed(task, preds, golds, idx)
    it = item_scores(task, preds, golds)
    if metric == "rmse":
        v = it["sqerr"]
        return lambda idx: float(np.sqrt(np.nanmean(v[idx])))
    v = it[metric]
    return lambda idx: float(np.nanmean(v[idx]))


def headline_err(task, metrics):
    """Convert a cell's headline metric to an error (0 = perfect)."""
    m, sign = HEADLINE[task]
    return metrics[m] if sign < 0 else 1.0 - metrics[m]
