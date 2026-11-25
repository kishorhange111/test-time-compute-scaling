import json
import random
from itertools import combinations

import pytest

from ttc.analyze import main as analyze_main
from ttc.answers import extract, gold_answer, normalize
from ttc.strategies import best_of, calibration, curves, pass_at_k, select, vote


@pytest.mark.parametrize(
    "text,want",
    [
        ("so 3 + 4 = 7. \\boxed{7}", "7"),
        ("First 12, then \\boxed{1,250}. Check: 1250", "1250"),
        ("The answer is $18.50.", "18.5"),
        ("The final answer is: -3", "-3"),
        ("He has 5 apples and buys 3 more, so 8", "8"),
        ("#### 42", "42"),
        ("no numbers here", None),
        ("\\boxed{12} wait, \\boxed{13}", "13"),
    ],
)
def test_extract(text, want):
    assert extract(text) == want


def test_normalize_and_gold():
    assert normalize("1,000.00") == "1000" and normalize("-0") == "0" and normalize("x") is None
    assert gold_answer("Natalia sold 48/2 = 24 clips.\n#### 72") == "72"


def test_pass_at_k_matches_brute_force():
    for n in (5, 8):
        for c in range(n + 1):
            for k in range(1, n + 1):
                hits = [any(i < c for i in s) for s in combinations(range(n), k)]
                assert pass_at_k(n, c, k) == pytest.approx(sum(hits) / len(hits))


def test_vote_and_best_of():
    assert vote(["3", "4", "4", None, None, None]) == "4"  # None never wins
    assert vote(["3", "4"]) == "3"  # tie: first seen
    assert vote([None, None]) is None
    assert vote(["3", "4", "4"], [5.0, 1.0, 1.0]) == "3"
    assert best_of(["3", "4", "5"], [-0.5, -0.1, -0.9]) == "4"
    with pytest.raises(ValueError):
        select("oracle", ["1"], [0.0])


def _problem(gold, answers, scores):
    return {
        "gold": gold,
        "samples": [
            {"answer": a, "score": s, "tokens": 10} for a, s in zip(answers, scores, strict=True)
        ],
        "greedy": {"answer": answers[0], "score": scores[0], "tokens": 10},
    }


def test_curves_exact_cases():
    # always right / always wrong
    probs = [_problem("1", ["1"] * 4, [-1.0] * 4), _problem("2", ["9"] * 4, [-1.0] * 4)]
    c = curves(probs, [1, 2, 4])
    for name in c:
        assert c[name] == pytest.approx([0.5, 0.5, 0.5])
    # majority over all 4 samples with 3 right: deterministic
    p = _problem("5", ["5", "5", "5", "7"], [-1.0, -1.0, -1.0, -0.1])
    c = curves([p], [4])
    assert c["majority"] == [1.0] and c["best_of"] == [0.0] and c["pass"] == [1.0]
    assert c["weighted"] == [1.0]  # 3·e^-1 > e^-0.1


def test_pass_upper_bounds_every_strategy_and_majority_scales():
    rnd = random.Random(0)
    probs = []
    for i in range(40):
        p_right = rnd.random()
        ans = [
            str(i) if rnd.random() < p_right else str(rnd.choice([i + 1, i + 2, i + 3]))
            for _ in range(16)
        ]
        probs.append(_problem(str(i), ans, [rnd.gauss(-0.5, 0.1) for _ in ans]))
    c = curves(probs, [1, 4, 16], n_subsets=100)
    for name in ("majority", "weighted", "best_of"):
        assert all(c[name][j] <= c["pass"][j] + 1e-9 for j in range(3))
    assert c["pass"][0] == pytest.approx(c["majority"][0], abs=0.03)  # k = 1: all coincide
    assert c["pass"][2] > c["pass"][0]


def test_calibration_bins_cover_all_samples():
    probs = [_problem("1", ["1", "2", "1", "3", "1"], [-0.1, -0.9, -0.2, -0.8, -0.3])]
    rows = calibration(probs, bins=2)
    assert sum(r["n"] for r in rows) == 5
    assert rows[-1]["accuracy"] >= rows[0]["accuracy"]


def test_analyze_end_to_end(tmp_path):
    rnd = random.Random(1)
    with open(tmp_path / "samples.jsonl", "w") as f:
        for i in range(10):
            ans = [str(i) if rnd.random() < 0.6 else "0" for _ in range(8)]
            f.write(json.dumps({"id": i, **_problem(str(i), ans, [-0.3] * 8)}) + "\n")
    analyze_main(["--out", str(tmp_path), "--ks", "1", "2", "8", "64"])
    text = (tmp_path / "summary.md").read_text(encoding="utf-8")
    assert "majority vote" in text and "| 64 |" not in text
