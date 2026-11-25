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

**Results are pending** from the Colab run.
