Qwen/Qwen2.5-1.5B-Instruct on the first 300 GSM8K test problems, 32 samples each at T=0.8.

Greedy decoding: **75.0%**. Mean sampled solution length 273 tokens, so k samples cost about 273·k generated tokens.

| k | 1 | 2 | 4 | 8 | 16 | 32 |
|---|---|---|---|---|---|---|
| pass@k (oracle) | 69.3 | 81.4 | 88.8 | 93.3 | 96.0 | 98.3 |
| majority vote | 69.1 | 69.1 | 78.5 | 82.7 | 85.0 | 87.0 |
| confidence-weighted vote | 69.5 | 73.3 | 79.6 | 83.2 | 85.2 | 87.3 |
| best-of-k by mean logprob | 69.2 | 73.4 | 75.7 | 76.8 | 77.2 | 77.0 |

Single-sample accuracy by mean-logprob quintile (is the model's confidence informative?):

| mean logprob | samples | accuracy |
|---|---|---|
| -0.920 … -0.220 | 1920 | 46.7% |
| -0.220 … -0.159 | 1920 | 61.3% |
| -0.159 … -0.121 | 1920 | 70.8% |
| -0.121 … -0.090 | 1920 | 80.0% |
| -0.090 … -0.028 | 1920 | 87.9% |
