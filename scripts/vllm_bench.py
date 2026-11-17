"""Benchmark a vLLM server: concurrency sweep, plus prefix-caching on vs off."""
import argparse, json, os

from servelab import env
from servelab.loadgen import run_sync
from servelab.server import VLLMServer
from servelab.workloads import shared_prefix_prompts, short_chat_prompts

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
ap.add_argument("--levels", default="1,2,4,8,16,32,64,128")
ap.add_argument("--max-tokens", type=int, default=128)
ap.add_argument("--only", default="sweep,prefix", help="comma list: sweep,prefix")
ap.add_argument("--out", default="results/vllm_bench.json")
ap.add_argument("--tag", default="fp16")
ap.add_argument("--server-args", default="", help="extra vllm args, space separated")
a = ap.parse_args()

extra = a.server_args.split()
result = {"env": env.capture(), "model": a.model, "tag": a.tag, "max_tokens": a.max_tokens}
base = "http://127.0.0.1:8000"
ignore_eos = {"ignore_eos": True}  # fixed output length so every request does equal decode work

if "sweep" in a.only:
    with VLLMServer(a.model, extra_args=extra, log_path=f"vllm_{a.tag}_sweep.log") as srv:
        result["server_facts"] = srv.facts()
        sweep = []
        for c in [int(x) for x in a.levels.split(",")]:
            n = max(16, 4 * c)
            r = run_sync(base, a.model, short_chat_prompts(n), c, max_tokens=a.max_tokens, extra=ignore_eos)
            sweep.append(r)
            print(f"c={c:3d} n={n:3d} out_tok/s={r['output_tok_per_s']:8.1f} ttft_p50={r['ttft_s']['p50']*1000:7.1f}ms "
                  f"ttft_p95={r['ttft_s']['p95']*1000:7.1f}ms tpot_p50={r['tpot_s'].get('p50', float('nan'))*1000:6.1f}ms failed={r['failed']}", flush=True)
        result["sweep"] = sweep

if "prefix" in a.only:
    system, qs = shared_prefix_prompts(33)
    pref = {}
    for label, flag in (("prefix_cache_on", "--enable-prefix-caching"), ("prefix_cache_off", "--no-enable-prefix-caching")):
        with VLLMServer(a.model, extra_args=extra + [flag], log_path=f"vllm_{a.tag}_{label}.log") as srv:
            cold = run_sync(base, a.model, qs[:1], 1, max_tokens=32, system=system, extra=ignore_eos, warmup=0)
            r = run_sync(base, a.model, qs[1:], 4, max_tokens=32, system=system, extra=ignore_eos, warmup=0)
            r["first_request_ttft_s"] = cold["ttft_s"]["p50"]
            r["prompt_tokens_each"] = r["input_tokens"] // max(1, r["requests"] - r["failed"])
            pref[label] = r
            print(label, f"first={cold['ttft_s']['p50']*1000:.0f}ms later_p50={r['ttft_s']['p50']*1000:.0f}ms prompt_tokens={r['prompt_tokens_each']}", flush=True)
    result["prefix"] = pref

os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
json.dump(result, open(a.out, "w"), indent=2)
print("wrote", a.out)
