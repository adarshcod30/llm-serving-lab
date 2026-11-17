"""A fake OpenAI-compatible streaming server with a known, fixed speed.

Used to check the load generator itself: if the server takes 50 ms to the first
token and 10 ms per token, the measured numbers must come back close to that.
"""
from __future__ import annotations

import asyncio
import json

from aiohttp import web

TTFT_S = 0.050
TOKEN_S = 0.010


async def chat(request: web.Request) -> web.StreamResponse:
    body = await request.json()
    n = int(body.get("max_tokens", 16))
    resp = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
    await resp.prepare(request)
    await asyncio.sleep(TTFT_S)
    for i in range(n):
        if i:
            await asyncio.sleep(TOKEN_S)
        chunk = {"choices": [{"delta": {"content": "tok "}}]}
        await resp.write(f"data: {json.dumps(chunk)}\n\n".encode())
    usage = {"choices": [], "usage": {"prompt_tokens": 20, "completion_tokens": n}}
    await resp.write(f"data: {json.dumps(usage)}\n\n".encode())
    await resp.write(b"data: [DONE]\n\n")
    return resp


def make_app() -> web.Application:
    app = web.Application()
    app.router.add_post("/v1/chat/completions", chat)
    return app
