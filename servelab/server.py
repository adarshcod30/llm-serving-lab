"""Start and stop a vLLM OpenAI-compatible server, and read facts out of its log.

The log is where vLLM states things worth checking: the attention backend it
chose, the KV cache size in tokens, the maximum concurrency it can hold.
Those lines are parsed and stored next to every result.
"""
from __future__ import annotations

import os
import re
import signal
import subprocess
import sys
import time
import urllib.request


class VLLMServer:
    def __init__(self, model: str, port: int = 8000, log_path: str = "vllm_server.log", extra_args: list[str] | None = None,
                 max_model_len: int = 4096, gpu_util: float = 0.88, dtype: str = "half"):
        self.model, self.port, self.log_path = model, port, log_path
        self.cmd = [
            sys.executable, "-m", "vllm.entrypoints.openai.api_server",
            "--model", model, "--port", str(port), "--dtype", dtype,
            "--max-model-len", str(max_model_len), "--gpu-memory-utilization", str(gpu_util),
            *(extra_args or []),
        ]
        self.proc: subprocess.Popen | None = None

    def __enter__(self):
        self._log = open(self.log_path, "w")
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=os.environ.get("CUDA_VISIBLE_DEVICES", "0"))
        self.proc = subprocess.Popen(self.cmd, stdout=self._log, stderr=subprocess.STDOUT, env=env, start_new_session=True)
        self._wait_ready()
        return self

    def _wait_ready(self, timeout: float = 900.0):
        t0 = time.time()
        url = f"http://127.0.0.1:{self.port}/v1/models"
        while time.time() - t0 < timeout:
            if self.proc.poll() is not None:
                raise RuntimeError(f"vLLM exited early (code {self.proc.returncode}). Last log:\n{self.tail(60)}")
            try:
                urllib.request.urlopen(url, timeout=2).read()
                self.ready_s = round(time.time() - t0, 1)
                return
            except Exception:
                time.sleep(3)
        raise TimeoutError(f"vLLM not ready after {timeout}s. Last log:\n{self.tail(60)}")

    def tail(self, n: int = 40) -> str:
        try:
            return "".join(open(self.log_path).readlines()[-n:])
        except OSError:
            return ""

    def facts(self) -> dict:
        """Facts vLLM printed about itself. Missing keys mean the line was not found."""
        text = open(self.log_path).read()
        out: dict = {"startup_s": getattr(self, "ready_s", None)}
        m = re.search(r"GPU KV cache size:\s*([\d,]+)\s*tokens", text)
        if m:
            out["kv_cache_tokens"] = int(m.group(1).replace(",", ""))
        m = re.search(r"Maximum concurrency for ([\d,]+) tokens per request:\s*([\d.]+)x", text)
        if m:
            out["max_concurrency_at_len"] = {"tokens": int(m.group(1).replace(",", "")), "x": float(m.group(2))}
        m = re.search(r"Model loading took ([\d.]+)\s*GiB", text)
        if m:
            out["weights_gib"] = float(m.group(1))
        m = re.search(r"Available KV cache memory:\s*([\d.]+)\s*GiB", text)
        if m:
            out["kv_cache_gib"] = float(m.group(1))
        m = re.search(r"(?:Using|Selected)[^\n]*?(FLASH[_A-Z]*|FLASHINFER|TRITON[_A-Z]*|XFORMERS|FLEX[_A-Z]*|TORCH_SDPA)[^\n]*", text, re.I)
        if m:
            out["attention_line"] = m.group(0)[:200]
        out["prefix_caching_line"] = next((l.strip()[:200] for l in text.splitlines() if "refix cach" in l), None)
        return out

    def __exit__(self, *exc):
        if self.proc and self.proc.poll() is None:
            os.killpg(os.getpgid(self.proc.pid), signal.SIGTERM)
            try:
                self.proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(os.getpgid(self.proc.pid), signal.SIGKILL)
        self._log.close()
        time.sleep(5)  # let the GPU memory actually free before the next server starts
