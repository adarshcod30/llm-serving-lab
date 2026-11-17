"""Async load generator for any OpenAI-compatible streaming endpoint.

For every request it records:
  ttft     seconds from send to the first streamed content chunk
  e2e      seconds from send to the last chunk
  tpot     (e2e - ttft) / (output_tokens - 1): mean time per output token after the first
  n_out    output tokens as reported by the server (usage), the ground truth
Closed-loop: C workers each keep exactly one request in flight, so concurrency
is C for the whole run. That is the standard way to read a throughput/latency curve.
"""
from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field

import aiohttp

from .stats import summarize


@dataclass
class ReqResult:
    ok: bool
    ttft: float = float("nan")
    e2e: float = float("nan")
    tpot: float = float("nan")
    n_out: int = 0
    n_in: int = 0
    error: str = ""
    gaps: list[float] = field(default_factory=list)


async def _one(session, url, model, messages, max_tokens, extra) -> ReqResult:
    body = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
        **extra,
    }
    t0 = time.perf_counter()
    t_first = None
    t_prev = None
    gaps: list[float] = []
    n_out = n_in = 0
    try:
        async with session.post(url, json=body) as resp:
            if resp.status != 200:
                return ReqResult(False, error=f"HTTP {resp.status}: {(await resp.text())[:200]}")
            async for raw in resp.content:
                line = raw.decode("utf-8", "ignore").strip()
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                evt = json.loads(payload)
                if evt.get("usage"):
                    n_out = evt["usage"].get("completion_tokens", n_out)
                    n_in = evt["usage"].get("prompt_tokens", n_in)
                choices = evt.get("choices") or []
                if choices and (choices[0].get("delta") or {}).get("content"):
                    now = time.perf_counter()
                    if t_first is None:
                        t_first = now
                    elif t_prev is not None:
                        gaps.append(now - t_prev)
                    t_prev = now
    except Exception as e:  # network, JSON, timeout: count it, do not hide it
        return ReqResult(False, error=f"{type(e).__name__}: {e}")
    t_end = time.perf_counter()
    if t_first is None:
        return ReqResult(False, error="no content received")
    ttft, e2e = t_first - t0, t_end - t0
    tpot = (e2e - ttft) / (n_out - 1) if n_out > 1 else float("nan")
    return ReqResult(True, ttft, e2e, tpot, n_out, n_in, gaps=gaps)


async def run_level(
    base_url: str,
    model: str,
    prompts: list[str],
    concurrency: int,
    max_tokens: int = 128,
    system: str | None = None,
    extra: dict | None = None,
    warmup: int = 2,
    timeout_s: float = 600.0,
) -> dict:
    """Send every prompt once at a fixed concurrency and summarise."""
    url = base_url.rstrip("/") + "/v1/chat/completions"
    extra = extra or {}

    def msgs(p: str):
        m = [{"role": "user", "content": p}]
        return ([{"role": "system", "content": system}] + m) if system else m

    timeout = aiohttp.ClientTimeout(total=timeout_s)
    conn = aiohttp.TCPConnector(limit=0)
    async with aiohttp.ClientSession(timeout=timeout, connector=conn) as session:
        for p in prompts[:warmup]:  # warm CUDA graphs, tokenizer, connection pool
            await _one(session, url, model, msgs(p), min(16, max_tokens), extra)

        queue: asyncio.Queue = asyncio.Queue()
        for p in prompts:
            queue.put_nowait(p)
        results: list[ReqResult] = []

        async def worker():
            while True:
                try:
                    p = queue.get_nowait()
                except asyncio.QueueEmpty:
                    return
                results.append(await _one(session, url, model, msgs(p), max_tokens, extra))

        t0 = time.perf_counter()
        await asyncio.gather(*[worker() for _ in range(concurrency)])
        wall = time.perf_counter() - t0

    good = [r for r in results if r.ok]
    total_out = sum(r.n_out for r in good)
    return {
        "concurrency": concurrency,
        "requests": len(results),
        "failed": len(results) - len(good),
        "errors": sorted({r.error for r in results if not r.ok})[:3],
        "wall_s": round(wall, 3),
        "output_tokens": total_out,
        "input_tokens": sum(r.n_in for r in good),
        "output_tok_per_s": round(total_out / wall, 2) if wall else 0.0,
        "req_per_s": round(len(good) / wall, 3) if wall else 0.0,
        "ttft_s": summarize(r.ttft for r in good),
        "tpot_s": summarize(r.tpot for r in good if r.tpot == r.tpot),
        "e2e_s": summarize(r.e2e for r in good),
        "chunk_gap_s": summarize(g for r in good for g in r.gaps),
    }


def run_sync(*a, **k) -> dict:
    return asyncio.run(run_level(*a, **k))
