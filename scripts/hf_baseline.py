"""Baseline: plain Hugging Face transformers generate(), no serving engine.

Static batching: a batch is padded, runs until its longest member finishes, and
nothing joins mid-way. This is what continuous batching is compared against.
"""
import argparse, json, statistics, time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from servelab import env
from servelab.workloads import short_chat_prompts

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
ap.add_argument("--batches", default="1,4,8,16,32")
ap.add_argument("--new-tokens", type=int, default=128)
ap.add_argument("--repeats", type=int, default=3)
ap.add_argument("--out", default="results/hf_baseline.json")
a = ap.parse_args()

tok = AutoTokenizer.from_pretrained(a.model, padding_side="left")
model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.float16).to("cuda").eval()
prompts = short_chat_prompts(64)

def chat(p):
    return tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True)

rows = []
for bs in [int(x) for x in a.batches.split(",")]:
    batch = [chat(p) for p in prompts[:bs]]
    enc = tok(batch, return_tensors="pt", padding=True).to("cuda")
    gen = dict(max_new_tokens=a.new_tokens, min_new_tokens=a.new_tokens, do_sample=False, pad_token_id=tok.pad_token_id)
    with torch.no_grad():
        model.generate(**enc, **{**gen, "max_new_tokens": 8, "min_new_tokens": 8})  # warm up
        times = []
        for _ in range(a.repeats):
            torch.cuda.synchronize(); t = time.perf_counter()
            out = model.generate(**enc, **gen)
            torch.cuda.synchronize(); times.append(time.perf_counter() - t)
    new = out.shape[1] - enc["input_ids"].shape[1]
    wall = statistics.median(times)
    rows.append({
        "batch_size": bs, "new_tokens_each": new, "wall_s_median": round(wall, 3),
        "output_tok_per_s": round(bs * new / wall, 2),
        "time_per_token_ms": round(1000 * wall / new, 2),
        "peak_mem_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2),
    })
    print(rows[-1], flush=True)

json.dump({"env": env.capture(), "model": a.model, "rows": rows}, open(a.out, "w"), indent=2)
