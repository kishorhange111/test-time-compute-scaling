# test-time-compute-scaling

[![CI](https://github.com/kishorhange111/test-time-compute-scaling/actions/workflows/ci.yml/badge.svg)](https://github.com/kishorhange111/test-time-compute-scaling/actions/workflows/ci.yml)

How much accuracy does spending more inference compute buy on GSM8K, and which way of
spending it is best? A small instruct model samples 32 solutions per problem. The samples
are then combined in different ways, and each is compared with the **oracle** that knows
which sample is right.

## Strategies ([`strategies.py`](src/ttc/strategies.py))

| strategy | answer from k samples | needs |
|---|---|---|
| **pass@k** (oracle) | correct if *any* sample is | the label; an upper bound for every selector |
| **majority vote** (self-consistency, Wang et al. 2023) | most common extracted answer | nothing |
| **confidence-weighted vote** | answers weighted by `exp(mean token logprob)` | logprobs |
| **best-of-k** | the single sample with the highest mean logprob | logprobs |

The evaluation is careful about bias:
- pass@k uses the **unbiased estimator** `1 − C(n−c, k)/C(n, k)` from all n samples
  (Chen et al. 2021). Taking the first k samples would be noisier, and taking the best k
  would be biased.
- The voting strategies are averaged over 200 random size-k subsets per problem.
- The gap between pass@k and each selector shows how much of the samples' latent accuracy
  that selector recovers.
- A calibration table checks whether the model's own confidence predicts correctness. That
  is what logprob-based selection relies on.

Answer extraction ([`answers.py`](src/ttc/answers.py)) prefers `\boxed{}`, then
"the answer is", then `####`, then the last number. It normalises `1,250.00` to `1250`.

The tests cover extraction edge cases and pass@k against brute-force enumeration over all
subsets. They also check voting tie-breaks and None handling, exact curve values in
degenerate cases, and that the oracle bounds every strategy.

## Running it

```
pytest -q                                                 # 15 tests, no GPU
python -m ttc.generate --out results --n 32 --problems 300   # vLLM on a GPU, resumable
python -m ttc.analyze  --out results
```

The defaults are Qwen2.5-1.5B-Instruct, the first 300 GSM8K test problems, 32 samples at
T = 0.8 and top-p 0.95, plus one greedy solution per problem.

## Results (Tesla T4, Colab, vLLM 0.34)

Qwen2.5-1.5B-Instruct on the first 300 GSM8K test problems, with 32 samples each at T = 0.8
(9,600 solutions) and up to 512 tokens. The summary is in
[`results/summary.md`](results/summary.md). Greedy decoding scores **75.0%**. A sampled
solution averages 273 tokens, so k samples cost about 273·k generated tokens.

| accuracy (%) at k = | 1 | 2 | 4 | 8 | 16 | 32 |
|---|---|---|---|---|---|---|
| pass@k (oracle) | 69.3 | 81.4 | 88.8 | 93.3 | 96.0 | **98.3** |
| majority vote | 69.1 | 69.1 | 78.5 | 82.7 | 85.0 | 87.0 |
| confidence-weighted vote | 69.5 | 73.3 | 79.6 | 83.2 | 85.2 | **87.3** |
| best-of-k by mean logprob | 69.2 | 73.4 | 75.7 | 76.8 | 77.2 | 77.0 |

Accuracy of a single sample, by quintile of its mean token logprob:

| mean logprob | samples | accuracy |
|---|---|---|
| −0.920 … −0.220 | 1,920 | 46.7% |
| −0.220 … −0.159 | 1,920 | 61.3% |
| −0.159 … −0.121 | 1,920 | 70.8% |
| −0.121 … −0.090 | 1,920 | 80.0% |
| −0.090 … −0.028 | 1,920 | 87.9% |

What the numbers show:
- **Voting turns 32× compute into +12 points over greedy decoding** (87.3% against 75.0%).
  It needs no verifier, but the curve flattens: from k = 16 to k = 32, doubling compute
  adds 0.2–2 points depending on the strategy.
- **A sample at T = 0.8 is worse than greedy** (69.3% against 75.0%), so sampling only pays
  off at about k ≥ 4.
- **Majority vote gains nothing from k = 1 to k = 2.** When two samples disagree, the tie
  goes to whichever answer was seen first, which is a random sample. Weighting the vote by
  confidence breaks ties sensibly and gains 4.2 points at k = 2. Above k = 8 the two votes
  are within 0.5 points.
- **The model's confidence is informative but a poor selector on its own.** Accuracy rises
  steadily with mean logprob, from 46.7% to 87.9% across quintiles. But picking the single
  most confident sample plateaus at 77%, barely above greedy. Confident wrong answers exist,
  and agreement across samples is the stronger signal.
- **The 11-point gap between pass@32 (98.3%) and the best vote (87.3%) is the headroom for a
  verifier** or reward model. The right answer is almost always among the samples; voting
  just can't always find it.
