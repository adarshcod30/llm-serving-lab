"""Small fixed texts for the tokenization study. Kept in code so results are reproducible."""

ENGLISH = (
    "Large language models read text as tokens, not characters. A tokenizer splits the "
    "input into pieces from a fixed vocabulary, and the model only ever sees the integer "
    "ids of those pieces. The number of tokens decides what a request costs, how much "
    "memory its key-value cache needs, and how long generation takes."
)
HINDI = (
    "बड़े भाषा मॉडल पाठ को अक्षरों के रूप में नहीं बल्कि टोकन के रूप में पढ़ते हैं। "
    "टोकनाइज़र इनपुट को एक निश्चित शब्दावली के टुकड़ों में बाँटता है, और मॉडल केवल उन "
    "टुकड़ों की संख्याएँ देखता है। टोकनों की संख्या तय करती है कि अनुरोध की लागत कितनी होगी।"
)
CODE = '''def percentile(values, q):
    xs = sorted(values)
    if not xs:
        return float("nan")
    pos = (len(xs) - 1) * q / 100.0
    lo, hi = int(pos), min(int(pos) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)
'''
JSON = (
    '{"request_id": "req_8f3a21", "model": "qwen2.5-1.5b", "usage": {"prompt_tokens": 412, '
    '"completion_tokens": 128, "total_tokens": 540}, "latency_ms": {"ttft": 83.4, "e2e": 1914.2}, '
    '"cache_hit": false, "timestamp": "2026-11-17T09:41:07Z"}'
)
NUMBERS = "Invoice 2026-11-17: 1,234,567.89 INR on 17/11/2026, ref 0045-8821-3390, GST 18%."
SENTENCE_FOR_SPLIT = "Tokenization decides cost. Unbelievably, unhappiness is tokenized differently."

SAMPLES = {
    "english": ENGLISH,
    "hindi": HINDI,
    "python": CODE,
    "json": JSON,
    "numbers": NUMBERS,
}
