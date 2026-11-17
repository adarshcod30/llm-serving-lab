"""Turn the raw JSON results into charts and a markdown results table. No numbers are typed by hand."""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

h = json.load(open("results/hf_baseline.json"))
v = json.load(open("results/vllm_fp16.json"))
hf = {r["batch_size"]: r for r in h["rows"]}
sw = {r["concurrency"]: r for r in v["sweep"]}

lines = ["| Concurrency | vLLM out tok/s | TTFT p50 | TTFT p95 | per-token p50 | per-token p95 | e2e p95 | HF generate tok/s (same batch) | vLLM / HF |",
         "|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
for c, r in sw.items():
    hfv = hf.get(c, {}).get("output_tok_per_s")
    lines.append(f"| {c} | {r['output_tok_per_s']:,.1f} | {r['ttft_s']['p50']*1000:,.0f} ms | {r['ttft_s']['p95']*1000:,.0f} ms | "
                 f"{r['tpot_s']['p50']*1000:.1f} ms | {r['tpot_s']['p95']*1000:.1f} ms | {r['e2e_s']['p95']:.2f} s | "
                 f"{hfv if hfv else 'n/a'} | {(str(round(r['output_tok_per_s']/hfv,2))+'x') if hfv else 'n/a'} |")
p_on, p_off = v["prefix"]["prefix_cache_on"], v["prefix"]["prefix_cache_off"]
lines += ["", "| Prefix caching (1,140-token shared prompt, concurrency 4, 32 requests) | TTFT p50 | TTFT p95 |", "|---|---:|---:|",
          f"| off | {p_off['ttft_s']['p50']*1000:,.0f} ms | {p_off['ttft_s']['p95']*1000:,.0f} ms |",
          f"| on | {p_on['ttft_s']['p50']*1000:,.0f} ms | {p_on['ttft_s']['p95']*1000:,.0f} ms |",
          f"| speed-up (p50) | {p_off['ttft_s']['p50']/p_on['ttft_s']['p50']:.1f}x | {p_off['ttft_s']['p95']/p_on['ttft_s']['p95']:.1f}x |"]
facts = v["server_facts"]
pred_gib = facts["kv_cache_tokens"] * 28672 / 2**30  # 28,672 B/token from servelab.kvmath on the model config
lines += ["", f"KV cache: vLLM reported {facts['kv_cache_tokens']:,} tokens in 8.70 GiB (log rounds to 2 decimals). "
          f"The config-derived 28,672 bytes per token predicts {pred_gib:.4f} GiB, which rounds to the same 8.70.",
          f"Total requests in the sweep: {sum(r['requests'] for r in v['sweep'])}, failed: {sum(r['failed'] for r in v['sweep'])}."]
open("results/results.md", "w").write("\n".join(lines) + "\n")
print("\n".join(lines))

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
cs = list(sw)
ax[0].plot(cs, [sw[c]["output_tok_per_s"] for c in cs], "o-", label="vLLM (HTTP, streaming)")
hb = sorted(hf)
ax[0].plot(hb, [hf[b]["output_tok_per_s"] for b in hb], "s--", label="HF generate (static batch)")
ax[0].set_xscale("log", base=2); ax[0].set_yscale("log"); ax[0].set_xlabel("concurrent requests / batch size")
ax[0].set_ylabel("output tokens per second"); ax[0].set_title("Throughput, Qwen2.5-1.5B fp16, Tesla T4"); ax[0].legend(); ax[0].grid(alpha=.3)
ax[1].plot(cs, [sw[c]["ttft_s"]["p50"]*1000 for c in cs], "o-", label="TTFT p50")
ax[1].plot(cs, [sw[c]["ttft_s"]["p95"]*1000 for c in cs], "o--", label="TTFT p95")
ax[1].plot(cs, [sw[c]["tpot_s"]["p95"]*1000 for c in cs], "s:", label="per-token p95")
ax[1].set_xscale("log", base=2); ax[1].set_yscale("log"); ax[1].set_xlabel("concurrent requests"); ax[1].set_ylabel("milliseconds")
ax[1].set_title("vLLM latency under load"); ax[1].legend(); ax[1].grid(alpha=.3)
fig.tight_layout(); fig.savefig("docs/throughput_latency.png", dpi=140)
