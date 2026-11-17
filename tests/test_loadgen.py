import asyncio

import pytest
from aiohttp import web

from servelab.loadgen import run_level
from servelab.stats import percentile
from servelab.workloads import short_chat_prompts, shared_prefix_prompts
from tests.fake_server import TOKEN_S, TTFT_S, make_app


def test_percentile_interpolates():
    assert percentile([1, 2, 3, 4], 50) == 2.5
    assert percentile([5], 95) == 5
    assert percentile([], 50) != percentile([], 50)  # NaN


def test_prompts_are_deterministic_and_distinct():
    a, b = short_chat_prompts(40), short_chat_prompts(40)
    assert a == b and len(set(a)) == 40


def test_shared_prefix_is_shared():
    system, qs = shared_prefix_prompts(10)
    assert len(system) > 5000 and len(set(qs)) == 10


@pytest.mark.parametrize("c", [1, 4])
def test_loadgen_recovers_known_server_speed(c):
    async def go():
        runner = web.AppRunner(make_app())
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        port = site._server.sockets[0].getsockname()[1]
        try:
            return await run_level(f"http://127.0.0.1:{port}", "fake", short_chat_prompts(8), c, max_tokens=21)
        finally:
            await runner.cleanup()

    r = asyncio.run(go())
    assert r["failed"] == 0 and r["output_tokens"] == 8 * 21
    # server: 50 ms to first token, 10 ms per token after. Allow scheduler jitter.
    assert TTFT_S * 0.9 < r["ttft_s"]["p50"] < TTFT_S + 0.04
    assert TOKEN_S * 0.9 < r["tpot_s"]["p50"] < TOKEN_S + 0.01


def test_kv_bytes_per_token_matches_vllm_report():
    """Qwen2.5-1.5B: 28 layers, 2 KV heads, head_dim 128, fp16 -> 28,672 B/token.
    vLLM 0.34 on a T4 reported 325,632 tokens in 8.70 GiB, which is the same number."""
    from servelab.kvmath import KVSpec
    s = KVSpec(layers=28, attn_heads=12, kv_heads=2, head_dim=128)
    assert s.bytes_per_token == 28672
    assert s.mha_bytes_per_token / s.bytes_per_token == 6
    assert round(325632 * s.bytes_per_token / 2**30, 2) == 8.70
