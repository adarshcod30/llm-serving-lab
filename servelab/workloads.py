"""Deterministic prompt sets. Same seed gives the same prompts on every run.

Distinct prompts matter: if every request were identical, prefix caching would
inflate the numbers and the benchmark would measure the cache, not the server.
"""
from __future__ import annotations

import random

TOPICS = [
    "photosynthesis", "the TCP handshake", "inflation", "binary search", "vaccines",
    "the French Revolution", "garbage collection", "black holes", "supply and demand",
    "public key cryptography", "the water cycle", "recursion", "plate tectonics",
    "the immune system", "database indexing", "the Mughal empire", "neural networks",
    "monsoons", "load balancing", "the theory of evolution", "compound interest",
    "hash tables", "the Indian Constitution", "solar panels", "caching",
]
TASKS = [
    "Explain {t} to a curious high-school student.",
    "Write a short, clear summary of {t} with one concrete example.",
    "List the three most common misconceptions about {t} and correct each one.",
    "Give a step-by-step explanation of how {t} works.",
    "Compare {t} with a related idea and say when each one matters.",
]


def short_chat_prompts(n: int, seed: int = 7) -> list[str]:
    """n distinct short chat prompts (roughly 15 to 30 tokens each)."""
    rng = random.Random(seed)
    combos = [(t, k) for t in TOPICS for k in range(len(TASKS))]
    rng.shuffle(combos)
    out = []
    for i in range(n):
        t, k = combos[i % len(combos)]
        suffix = "" if i < len(combos) else f" (variation {i // len(combos)})"
        out.append(TASKS[k].format(t=t) + suffix)
    return out


_FILLER = (
    "The quarterly operations report records the status of every service in the "
    "fleet, the incidents opened and closed, the capacity headroom for each region, "
    "and the owners responsible for follow-up work. "
)


def shared_prefix_prompts(n: int, prefix_chars: int = 6000, seed: int = 11) -> tuple[str, list[str]]:
    """A long shared system prompt plus n distinct short questions.

    Returns (system_prompt, questions). The shared part is what prefix caching reuses.
    """
    rng = random.Random(seed)
    body = (_FILLER * (prefix_chars // len(_FILLER) + 1))[:prefix_chars]
    system = "You answer questions about the following report.\n\n" + body
    qs = short_chat_prompts(n, seed=seed)
    rng.shuffle(qs)
    return system, qs
