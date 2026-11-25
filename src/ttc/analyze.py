"""Scaling curves from ``samples.jsonl``: accuracy vs samples (and tokens) per strategy.

python -m ttc.analyze --out results
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

from .strategies import STRATEGIES, calibration, curves


def report(problems, ks, meta: str = "") -> str:
    ks = [k for k in ks if k <= len(problems[0]["samples"])]
    cur = curves(problems, ks)
    greedy = np.mean([p["greedy"]["answer"] == p["gold"] for p in problems if "greedy" in p])
    tok = np.mean([s["tokens"] for p in problems for s in p["samples"]])
    lines = [
        meta,
        "",
        f"Greedy decoding: **{100 * greedy:.1f}%**. Mean sampled solution length "
        f"{tok:.0f} tokens, so k samples cost about {tok:.0f}·k generated tokens.",
        "",
        "| k | " + " | ".join(f"{k}" for k in ks) + " |",
        "|---|" + "---|" * len(ks),
    ]
    names = {
        "pass": "pass@k (oracle)",
        "majority": "majority vote",
        "weighted": "confidence-weighted vote",
        "best_of": "best-of-k by mean logprob",
    }
    for n in ("pass", *STRATEGIES):
        lines.append(f"| {names[n]} | " + " | ".join(f"{100 * v:.1f}" for v in cur[n]) + " |")
    lines += [
        "",
        "Single-sample accuracy by mean-logprob quintile (is the model's confidence informative?):",
        "",
        "| mean logprob | samples | accuracy |",
        "|---|---|---|",
    ]
    for r in calibration(problems):
        lines.append(f"| {r['lo']:.3f} … {r['hi']:.3f} | {r['n']} | {100 * r['accuracy']:.1f}% |")
    return "\n".join(lines) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results")
    ap.add_argument("--ks", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32])
    a = ap.parse_args(argv)
    problems = [json.loads(x) for x in open(os.path.join(a.out, "samples.jsonl"))]
    cfg_path = os.path.join(a.out, "config.json")
    cfg = json.load(open(cfg_path)) if os.path.exists(cfg_path) else {}
    meta = (
        f"{cfg.get('model', '?')} on the first {len(problems)} GSM8K test problems, "
        f"{len(problems[0]['samples'])} samples each at T={cfg.get('temperature', '?')}."
    )
    text = report(problems, a.ks, meta)
    with open(os.path.join(a.out, "summary.md"), "w", encoding="utf-8") as f:
        f.write(text)
    print(text)


if __name__ == "__main__":
    main()
