"""Ways to turn N sampled solutions into one answer, and how to score them without bias.

For each problem we hold ``n`` samples, each with an extracted answer and a confidence
score (mean token log-probability). For a budget of ``k ≤ n`` samples:

- ``pass@k`` (oracle): is *any* of k samples correct? It uses the unbiased estimator
  ``1 − C(n−c, k) / C(n, k)`` (Chen et al. 2021), where c is the number of correct samples
  among n. This is an upper bound for any selection method that only sees the k samples.
- ``majority@k`` (self-consistency, Wang et al. 2023): the most common answer wins.
- ``weighted@k``: answers are voted with weight ``exp(mean logprob)``, so confident chains
  count more.
- ``best_of@k``: the single sample with the highest mean log-probability. This is
  best-of-N with the model's own likelihood as the verifier.

Expectations over which k of the n samples were drawn are estimated by Monte-Carlo over
random size-k subsets (fixed seed), except for pass@k, which is exact.
"""

from __future__ import annotations

from collections import defaultdict
from math import comb

import numpy as np


def pass_at_k(n: int, c: int, k: int) -> float:
    if n - c < k:
        return 1.0
    return 1.0 - comb(n - c, k) / comb(n, k)


def vote(answers, weights=None):
    """Answer with the largest total weight (None answers never win). Ties go to the answer
    seen first, which is a random choice among the tied answers since samples are i.i.d."""
    tot: dict[str, float] = defaultdict(float)
    order: dict[str, int] = {}
    for i, a in enumerate(answers):
        if a is None:
            continue
        tot[a] += 1.0 if weights is None else float(weights[i])
        order.setdefault(a, i)
    if not tot:
        return None
    return max(tot, key=lambda a: (tot[a], -order[a]))


def best_of(answers, scores):
    return answers[int(np.argmax(scores))]


STRATEGIES = ("majority", "weighted", "best_of")


def select(strategy: str, answers, scores):
    if strategy == "majority":
        return vote(answers)
    if strategy == "weighted":
        return vote(answers, np.exp(np.asarray(scores, dtype=float)))
    if strategy == "best_of":
        return best_of(answers, scores)
    raise ValueError(strategy)


def curves(problems, ks, n_subsets: int = 200, seed: int = 0) -> dict[str, list[float]]:
    """Accuracy at each k for pass@k and every strategy.

    ``problems``: iterable of dicts with ``gold`` and ``samples``: a list of
    ``{"answer": str | None, "score": float}``, all with the same number of samples.
    """
    rng = np.random.default_rng(seed)
    out = {name: [] for name in ("pass", *STRATEGIES)}
    probs = list(problems)
    for k in ks:
        acc = defaultdict(float)
        for p in probs:
            ans = [s["answer"] for s in p["samples"]]
            sc = np.array([s["score"] for s in p["samples"]], dtype=float)
            n = len(ans)
            c = sum(a == p["gold"] for a in ans)
            acc["pass"] += pass_at_k(n, c, k)
            reps = 1 if k == n else n_subsets
            for name in STRATEGIES:
                hits = 0
                for _ in range(reps):
                    idx = rng.choice(n, k, replace=False) if k < n else np.arange(n)
                    hits += select(name, [ans[i] for i in idx], sc[idx]) == p["gold"]
                acc[name] += hits / reps
        for name in out:
            out[name].append(acc[name] / len(probs))
    return out


def calibration(problems, bins: int = 5):
    """Accuracy of single samples by mean-logprob quantile bin: is confidence informative?"""
    pts = [(s["score"], s["answer"] == p["gold"]) for p in problems for s in p["samples"]]
    sc = np.array([x for x, _ in pts])
    ok = np.array([y for _, y in pts], dtype=float)
    edges = np.quantile(sc, np.linspace(0, 1, bins + 1))
    rows = []
    for i in range(bins):
        m = (sc >= edges[i]) & (sc <= edges[i + 1] if i == bins - 1 else sc < edges[i + 1])
        rows.append(
            {
                "lo": float(edges[i]),
                "hi": float(edges[i + 1]),
                "n": int(m.sum()),
                "accuracy": float(ok[m].mean()) if m.any() else float("nan"),
            }
        )
    return rows
