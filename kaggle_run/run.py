"""Kaggle GPU kernel: clone the repo at a pinned commit, run the whole suite, keep logs and JSON."""
import os, subprocess, sys, time

REPO = "https://github.com/adarshcod30/llm-serving-lab.git"
COMMIT = os.environ.get("LAB_COMMIT", "main")
W = "/kaggle/working"

def sh(cmd, check=True):
    print(f"\n$ {cmd}", flush=True)
    t = time.time()
    r = subprocess.run(cmd, shell=True, cwd=W)
    print(f"[exit {r.returncode} in {time.time()-t:.0f}s]", flush=True)
    if check and r.returncode:
        raise SystemExit(r.returncode)

sh(f"rm -rf lab && git clone {REPO} lab && cd lab && git checkout {COMMIT} && git rev-parse HEAD > ../COMMIT.txt && cat ../COMMIT.txt")
sh("nvidia-smi --query-gpu=name,memory.total,compute_cap,driver_version --format=csv")
sh("pip install -q vllm aiohttp tiktoken 2>&1 | tail -3")
sh("cd lab && pip install -q -e . 2>&1 | tail -2")
sh("cd lab && python -m pytest -q 2>&1 | tail -3", check=False)
os.makedirs(f"{W}/results", exist_ok=True)
M = "Qwen/Qwen2.5-1.5B-Instruct"
# Order matters: the HF baseline needs the GPU to itself, so it runs before any vLLM server starts.
sh(f"cd lab && python scripts/hf_baseline.py --model {M} --out {W}/results/hf_baseline.json")
sh(f"cd lab && python scripts/vllm_bench.py --model {M} --tag fp16 --out {W}/results/vllm_fp16.json")
sh(f"cd lab && python -m servelab.tokens {W}/results/tokenization.json", check=False)
sh(f"cp lab/vllm_*.log {W}/results/ 2>/dev/null; ls -la {W}/results")
