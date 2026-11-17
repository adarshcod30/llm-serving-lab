"""Probe: what does a Kaggle GPU kernel give us, and does vLLM install and run on it?"""
import subprocess, sys, time, platform, json

def sh(cmd, timeout=1500):
    t = time.time()
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
    return r.returncode, (r.stdout + r.stderr), round(time.time() - t, 1)

print("python", sys.version.replace("\n", " "), platform.platform())
print(sh("nvidia-smi --query-gpu=name,memory.total,compute_cap,driver_version --format=csv")[1])
print(sh("nvcc --version | tail -2")[1])
print(sh("pip list 2>/dev/null | grep -i -E '^(torch|transformers|vllm|aiohttp|xformers|flashinfer|numpy) '")[1])
print("disk", sh("df -h /kaggle/working | tail -1")[1])
rc, out, secs = sh("pip install -q vllm 2>&1 | tail -15")
print("pip install vllm: rc", rc, "secs", secs); print(out[-1500:])
print(sh("pip list 2>/dev/null | grep -i -E '^(torch|vllm|transformers|xformers|flashinfer-python) '")[1])
code = r'''
import time, torch, vllm
print("vllm", vllm.__version__, "torch", torch.__version__, "cuda", torch.version.cuda)
from vllm import LLM, SamplingParams
t=time.time()
llm = LLM("Qwen/Qwen2.5-0.5B-Instruct", dtype="half", max_model_len=2048, gpu_memory_utilization=0.85, enforce_eager=False)
print("load+init secs", round(time.time()-t,1))
sp = SamplingParams(temperature=0, max_tokens=64)
t=time.time(); o = llm.generate(["Explain PagedAttention in one sentence."], sp); dt=time.time()-t
print(o[0].outputs[0].text); print("one request secs", round(dt,2), "tokens", len(o[0].outputs[0].token_ids))
'''
open("/kaggle/working/p.py","w").write(code)
rc, out, secs = sh("cd /kaggle/working && python p.py 2>&1", timeout=1200)
print("vllm smoke: rc", rc, "secs", secs); print(out[-6000:])
