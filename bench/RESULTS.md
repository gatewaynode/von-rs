# Python vs Rust server benchmarks

`tools/bench_compare.sh` starts the Python server (`von serve` from the Python package) and
the Rust `von serve` one after the other on the same port, then sends both the same requests
over HTTP:

- **Cold start:** time from process launch until `/health` answers, and until the first
  `/v1/systemone` answer. Python loads the model on the first request and Rust loads it at
  startup, so the second number is the one to compare. With `PURGE=1` the OS page cache is
  purged before each launch, so the weights come off disk (a true cold start).
- **Latency:** p50 and p95 of sequential requests, after 5 warm-up requests. The requests are
  golden cases also used by `examples/latency.rs`.
- **Throughput:** 8 clients sending 200 short requests in total.
- **Memory:** the server's resident set size after the run.

These numbers are a measurement baseline and involve no tuning. Each section below is rewritten
when its device is re-run: `just bench-compare cpu` or `just bench-compare metal`.

<!-- bench_compare:cpu -->
## CPU: Python (CPU) vs Rust (CPU)

Apple M3 Ultra · macOS 26.6.2 · torch 2.14.0 · rustc 1.98.0 · 2026-09-26 · true cold start (page cache purged before each launch).

| Metric | Python | Rust | Rust / Python |
|---|---|---|---|
| Launch → /health ready (s) | 2.58 | 2.87 | 1.11× |
| Launch → first answer (s) | 3.72 | 2.96 | 0.80× |
| route-account-access, p50 / p95 (ms) | 71.56 / 72.61 | 79.79 / 81.50 | 1.12× |
| many-options, p50 / p95 (ms) | 95.12 / 96.30 | 98.58 / 100.07 | 1.04× |
| noul-zero-shot, p50 / p95 (ms) | 127.78 / 129.97 | 149.50 / 152.20 | 1.17× |
| score-detailed, p50 / p95 (ms) | 72.90 / 75.38 | 87.39 / 89.46 | 1.20× |
| long-600-tokens, p50 / p95 (ms) | 278.25 / 284.03 | 1027.83 / 1041.22 | 3.69× |
| Throughput, 8 clients (req/s) | 13.9 | 54.3 | 3.91× |
| Errors | 0 | 0 | |
| RSS after the run (MB) | 2011 | 1976 | 0.98× |
<!-- /bench_compare:cpu -->

<!-- bench_compare:metal -->
## Metal: Python (MPS) vs Rust (Metal)

Apple M3 Ultra · macOS 26.6.2 · torch 2.14.0 · rustc 1.98.0 · 2026-09-26 · true cold start (page cache purged before each launch).

| Metric | Python | Rust | Rust / Python |
|---|---|---|---|
| Launch → /health ready (s) | 2.49 | 2.90 | 1.17× |
| Launch → first answer (s) | 4.13 | 2.95 | 0.71× |
| route-account-access, p50 / p95 (ms) | 14.81 / 15.18 | 21.55 / 21.75 | 1.45× |
| many-options, p50 / p95 (ms) | 16.46 / 17.52 | 22.68 / 22.89 | 1.38× |
| noul-zero-shot, p50 / p95 (ms) | 27.26 / 27.83 | 45.39 / 45.79 | 1.66× |
| score-detailed, p50 / p95 (ms) | 15.11 / 15.52 | 22.20 / 22.43 | 1.47× |
| long-600-tokens, p50 / p95 (ms) | 47.75 / 48.07 | 174.46 / 175.48 | 3.65× |
| Throughput, 8 clients (req/s) | 66.3 | 45.8 | 0.69× |
| Errors | 0 | 0 | |
| RSS after the run (MB) | 762 | 1621 | 2.13× |
<!-- /bench_compare:metal -->
