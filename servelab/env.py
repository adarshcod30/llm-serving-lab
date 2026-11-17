"""Record exactly what a result was measured on, so it can be checked later."""
from __future__ import annotations

import datetime as dt
import importlib.metadata as md
import platform
import subprocess
import sys


def _run(cmd: list[str]) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=30).stdout.strip()
    except Exception:
        return ""


def capture(extra_packages: tuple[str, ...] = ("vllm", "torch", "transformers", "aiohttp", "tiktoken")) -> dict:
    pkgs = {}
    for p in extra_packages:
        try:
            pkgs[p] = md.version(p)
        except md.PackageNotFoundError:
            pass
    gpu = _run(["nvidia-smi", "--query-gpu=name,memory.total,compute_cap,driver_version", "--format=csv,noheader"])
    return {
        "utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "gpu": gpu.splitlines() if gpu else [],
        "packages": pkgs,
    }
