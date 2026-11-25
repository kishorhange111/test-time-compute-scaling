"""Answer extraction for GSM8K-style numeric problems."""

from __future__ import annotations

import re

_NUM = r"-?\d[\d,]*(?:\.\d+)?"


def normalize(x: str | None) -> str | None:
    """Canonical string for a number: no commas, no trailing zeros, '-0' -> '0'."""
    if x is None:
        return None
    x = x.replace(",", "").replace("$", "").strip().rstrip(".")
    try:
        v = float(x)
    except ValueError:
        return None
    if v.is_integer():
        return str(int(v)) if v != 0 else "0"
    return f"{v:.6f}".rstrip("0").rstrip(".")


def gold_answer(solution: str) -> str | None:
    """GSM8K reference solutions end with '#### <answer>'."""
    m = re.search(r"####\s*(" + _NUM + ")", solution)
    return normalize(m.group(1)) if m else None


def extract(text: str) -> str | None:
    """Final answer from a model completion, in order of reliability:
    ``\\boxed{...}``, then 'the answer is ...', then '#### ...', then the last number."""
    for pat in (
        r"\\boxed\{\s*\$?(" + _NUM + r")\s*\}",
        r"[Tt]he (?:final )?answer is[:\s]*\$?\s*(" + _NUM + ")",
        r"####\s*\$?(" + _NUM + ")",
    ):
        hits = re.findall(pat, text)
        if hits:
            return normalize(hits[-1])
    nums = re.findall(_NUM, text)
    return normalize(nums[-1]) if nums else None
