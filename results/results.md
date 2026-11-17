| Concurrency | vLLM out tok/s | TTFT p50 | TTFT p95 | per-token p50 | per-token p95 | e2e p95 | HF generate tok/s (same batch) | vLLM / HF |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 65.0 | 30 ms | 31 ms | 15.2 ms | 15.9 ms | 2.05 s | 33.32 | 1.95x |
| 2 | 129.3 | 47 ms | 49 ms | 15.2 ms | 15.3 ms | 1.99 s | n/a | n/a |
| 4 | 247.6 | 53 ms | 69 ms | 15.8 ms | 15.9 ms | 2.07 s | 114.25 | 2.17x |
| 8 | 459.6 | 87 ms | 98 ms | 16.9 ms | 17.1 ms | 2.23 s | 222.61 | 2.06x |
| 16 | 820.2 | 152 ms | 191 ms | 18.5 ms | 19.0 ms | 2.54 s | 433.89 | 1.89x |
| 32 | 1,235.5 | 244 ms | 308 ms | 24.2 ms | 25.2 ms | 3.37 s | 810.97 | 1.52x |
| 64 | 1,940.3 | 325 ms | 545 ms | 30.6 ms | 32.0 ms | 4.26 s | n/a | n/a |
| 128 | 2,660.4 | 604 ms | 1,002 ms | 43.4 ms | 46.6 ms | 6.33 s | n/a | n/a |

| Prefix caching (1,140-token shared prompt, concurrency 4, 32 requests) | TTFT p50 | TTFT p95 |
|---|---:|---:|
| off | 984 ms | 1,383 ms |
| on | 138 ms | 150 ms |
| speed-up (p50) | 7.1x | 9.2x |

KV cache: vLLM reported 325,632 tokens in 8.70 GiB (log rounds to 2 decimals). The config-derived 28,672 bytes per token predicts 8.6953 GiB, which rounds to the same 8.70.
Total requests in the sweep: 1040, failed: 0.
