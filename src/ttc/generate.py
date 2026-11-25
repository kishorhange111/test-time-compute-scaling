"""Sample N solutions per GSM8K test problem with vLLM (GPU).

    python -m ttc.generate --out results --n 32 --problems 300

For each problem it stores one greedy solution plus ``n`` samples at temperature 0.8, each
with its extracted answer, token count and mean token log-probability. Problems are written
in chunks to ``<out>/samples.jsonl`` and skipped on restart, so the run survives Colab's
session cap.
"""

from __future__ import annotations

import argparse
import json
import os
import time

from .answers import extract, gold_answer

SYSTEM = "Please reason step by step, and put your final answer within \\boxed{}."


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results")
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--problems", type=int, default=300)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--chunk", type=int, default=25)
    a = ap.parse_args(argv)

    from datasets import load_dataset
    from vllm import LLM, SamplingParams

    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, "samples.jsonl")
    done = {json.loads(x)["id"] for x in open(path)} if os.path.exists(path) else set()
    data = load_dataset("openai/gsm8k", "main", split="test").select(range(a.problems))
    llm = LLM(a.model, dtype="half", max_model_len=1024, gpu_memory_utilization=0.9, seed=0)
    tok = llm.get_tokenizer()
    sample = SamplingParams(
        n=a.n, temperature=a.temperature, top_p=0.95, max_tokens=a.max_tokens, logprobs=0, seed=0
    )
    greedy = SamplingParams(n=1, temperature=0.0, max_tokens=a.max_tokens, logprobs=0)

    def rec(o):
        n_tok = max(1, len(o.token_ids))
        return {
            "answer": extract(o.text),
            "score": o.cumulative_logprob / n_tok,
            "tokens": n_tok,
            "finished": o.finish_reason == "stop",
        }

    todo = [i for i in range(len(data)) if i not in done]
    for c in range(0, len(todo), a.chunk):
        ids = todo[c : c + a.chunk]
        prompts = [
            tok.apply_chat_template(
                [
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": data[i]["question"]},
                ],
                tokenize=False,
                add_generation_prompt=True,
            )
            for i in ids
        ]
        t0 = time.time()
        outs, gouts = llm.generate(prompts, sample), llm.generate(prompts, greedy)
        with open(path, "a") as f:
            for i, o, g in zip(ids, outs, gouts, strict=True):
                f.write(
                    json.dumps(
                        {
                            "id": i,
                            "gold": gold_answer(data[i]["answer"]),
                            "greedy": rec(g.outputs[0]),
                            "samples": [rec(x) for x in o.outputs],
                        }
                    )
                    + "\n"
                )
        n_tok = sum(len(x.token_ids) for o in outs for x in o.outputs)
        print(
            f"[{time.strftime('%H:%M:%S')}] problems {ids[0]}–{ids[-1]}: "
            f"{n_tok / (time.time() - t0):.0f} sampled tok/s",
            flush=True,
        )
    with open(os.path.join(a.out, "config.json"), "w") as f:
        json.dump(vars(a), f, indent=1)


if __name__ == "__main__":
    main()
