# Baseline measurements

> The first sections are a historical record from the feasibility spike. `examples/spike.rs` measured single forward passes and has since
> been replaced by `examples/latency.rs`, which times whole `evaluate()` calls against the
> same requests `tools/bench_python.py` uses. Take new measurements with `tools/metal_check.sh`.

Machine: Apple M3 Ultra (80-core GPU), 512 GB, macOS 26.6.2.
Python: torch 2.14.0, transformers 5.17.0, 24 threads. Rust: rustc 1.98.0, candle 0.11.0.
Weights: `wfzyx/von` snapshot `d8bb5e07`, converted with `tools/convert_weights.py`.
Golden set: `tests/fixtures/golden/v0.json`, which has 12 cases and 13 forward passes and was exported on torch CPU fp32.

Latency is end to end per forward pass: tokenize, encode, gather, score, host copy. Figures are
p50 after warmup. Python numbers cover the whole `backend.evaluate` call.

## Parity (candle vs torch CPU fp32)

| Device | Token ids | Worst mask-hidden Δ | Worst logit Δ | Argmax |
|---|---|---|---|---|
| CPU (gemm) | 13/13 identical | 6.0e-5 | 2.7e-5 | 13/13 |
| CPU (Accelerate) | 13/13 identical | 6.0e-5 | 3.5e-5 | 13/13 |
| Metal | 13/13 identical | 6.1e-5 | 3.3e-5 | 13/13 |

The parity tolerance is 1e-3. CPU and Metal are both about 30× inside it.

## Latency, p50 ms

| Case | tokens | Python CPU | Rust CPU (gemm) | Rust CPU (Accelerate) | Python MPS | Rust Metal |
|---|---|---|---|---|---|---|
| route-account-access | 46 | 73.5 | 245.0 | 80.4 | 14.8 | 21.0 |
| many-options | 57 | 75.1 | 259.4 | 87.9 | 15.2 | 21.5 |
| score-detailed | 57 | 74.7 | 258.3 | 91.2 | 14.8 | 21.5 |
| noul-zero-shot (2 passes) | 35+23 | 132.3 | 432.3 | 152.4 | 26.9 | 44.3 |

## Notes

- Metal and MPS numbers were taken in a normal terminal session (the script is now
  `tools/metal_check.sh`). Rust Metal model load took 3.72 s, against a cold-start target of ≤ 3 s.
- **Metal latency: 1.41–1.45× Python MPS** on single passes. Tuning is deferred until the port is complete.
  Latency is nearly flat, 20.5–22.6 ms across 23–81 tokens, which points to fixed per-call overhead
  (op dispatch and mask construction) rather than compute.
- Candle's default CPU matmul (`gemm`) is about 3.3× slower than torch. The `accelerate` feature
  (Apple AMX via Accelerate.framework) closes most of the gap and is required for the CPU fallback.
- The remaining ~1.15× CPU gap is candle's ModernBERT itself. It rebuilds the seq×seq sliding-window
  mask on the host every forward, re-adds it in each of the 18 local layers, and uses unfused
  attention.

## Whole-request latency, p50 / p95 ms (2026-09-22)

`examples/latency.rs` (Rust) and `tools/bench_python.py` (Python) time the same golden requests
end to end with `evaluate()`. Recorded only; tuning comes after the port is complete.

| Request | Python MPS | Rust Metal | Ratio | Rust CPU (Accelerate) |
|---|---|---|---|---|
| route-account-access | 14.42 / 14.60 | 21.24 / 21.47 | 1.47× | 80.66 / 82.36 |
| many-options (12 options) | 16.13 / 16.42 | 22.35 / 22.53 | 1.39× | 97.93 / 100.19 |
| noul-zero-shot (2 passes) | 26.83 / 27.19 | 45.22 / 45.46 | 1.69× | 152.34 / 155.85 |
| score-detailed | 14.73 / 14.97 | 21.88 / 22.04 | 1.49× | 87.94 / 89.81 |
| long-600-tokens (778 tokens) | 47.01 / 47.35 | 173.85 / 174.75 | **3.70×** | 910.36 / 916.84 |

Rust Metal model load: 0.27 s, with the page cache warm.
