"""Tokenization study: same text, different tokenizers, very different costs."""
from __future__ import annotations

import json
import sys

from .corpus import SAMPLES, SENTENCE_FOR_SPLIT

HF_TOKENIZERS = {
    "qwen2.5": "Qwen/Qwen2.5-1.5B-Instruct",
    "llama3.2": "unsloth/Llama-3.2-1B-Instruct",
    "gpt2": "openai-community/gpt2",
}
TIKTOKEN = {"cl100k_base (GPT-4)": "cl100k_base", "o200k_base (GPT-4o)": "o200k_base"}


def load_encoders() -> tuple[dict, dict]:
    """Return ({name: encode_fn}, {name: why_it_failed}). Never silently drop one."""
    enc, failed = {}, {}
    for name, repo in HF_TOKENIZERS.items():
        try:
            from transformers import AutoTokenizer
            tok = AutoTokenizer.from_pretrained(repo)
            enc[name] = (lambda t, tok=tok: tok.encode(t, add_special_tokens=False), tok)
        except Exception as e:
            failed[name] = f"{type(e).__name__}: {str(e)[:120]}"
    for name, key in TIKTOKEN.items():
        try:
            import tiktoken
            e = tiktoken.get_encoding(key)
            enc[name] = (lambda t, e=e: e.encode(t), e)
        except Exception as ex:
            failed[name] = f"{type(ex).__name__}: {str(ex)[:120]}"
    return enc, failed


def study() -> dict:
    enc, failed = load_encoders()
    rows = []
    for tname, (fn, obj) in enc.items():
        for sname, text in SAMPLES.items():
            ids = fn(text)
            # round-trip check: decode(encode(x)) must give x back
            if hasattr(obj, "decode"):
                back = obj.decode(ids)
                ok = back == text or back.strip() == text.strip()
            else:
                ok = None
            rows.append({
                "tokenizer": tname, "sample": sname, "chars": len(text),
                "bytes": len(text.encode("utf-8")), "words": len(text.split()),
                "tokens": len(ids), "chars_per_token": round(len(text) / len(ids), 2),
                "tokens_per_word": round(len(ids) / max(1, len(text.split())), 2),
                "roundtrip_ok": ok,
            })
    # how one sentence is actually split, so the numbers are not abstract
    splits = {}
    for tname, (fn, obj) in enc.items():
        ids = fn(SENTENCE_FOR_SPLIT)
        splits[tname] = [obj.decode([i]) if hasattr(obj, "decode") else str(i) for i in ids]
    # chat-template overhead: tokens added around a one-word user message
    overhead = {}
    if "qwen2.5" in enc:
        tok = enc["qwen2.5"][1]
        msgs = [{"role": "user", "content": "Hi"}]
        templated = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True)
        ids = templated["input_ids"] if isinstance(templated, dict) or hasattr(templated, "keys") else templated
        overhead["qwen2.5_one_word_prompt"] = {"content_tokens": len(enc["qwen2.5"][0]("Hi")), "total_prompt_tokens": len(ids)}
    return {"rows": rows, "splits": splits, "chat_template_overhead": overhead, "failed": failed}


if __name__ == "__main__":
    out = study()
    json.dump(out, open(sys.argv[1] if len(sys.argv) > 1 else "results/tokenization.json", "w"), indent=2, ensure_ascii=False)
    for r in out["rows"]:
        print(f'{r["tokenizer"]:22s} {r["sample"]:8s} tokens={r["tokens"]:4d} chars/tok={r["chars_per_token"]:5.2f} tok/word={r["tokens_per_word"]:5.2f} rt={r["roundtrip_ok"]}')
    print("failed:", out["failed"])
