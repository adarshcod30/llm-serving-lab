"""Small, dependency-free statistics helpers."""
from __future__ import annotations

import math
from typing import Iterable


def percentile(values: Iterable[float], q: float) -> float:
    """Linear-interpolated percentile, q in [0, 100]. NaN for empty input."""
    xs = sorted(values)
    if not xs:
        return math.nan
    if len(xs) == 1:
        return xs[0]
    pos = (len(xs) - 1) * q / 100.0
    lo, hi = math.floor(pos), math.ceil(pos)
    return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)


def summarize(values: Iterable[float]) -> dict[str, float]:
    xs = list(values)
    if not xs:
        return {"n": 0}
    return {
        "n": len(xs),
        "mean": sum(xs) / len(xs),
        "p50": percentile(xs, 50),
        "p95": percentile(xs, 95),
        "p99": percentile(xs, 99),
        "max": max(xs),
    }
