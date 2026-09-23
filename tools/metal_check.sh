#!/usr/bin/env bash
# Metal checks, for a normal terminal session with GPU access:
# golden parity, the unsafe mapping test and the model-backed pattern tests on
# Metal, the cross-SDK check against `von serve` on Metal, then latency for Rust
# and Python (skip it with SKIP_LATENCY=1).
# Run from a normal terminal:  bash von-rs/tools/metal_check.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
export HF_HOME="${HF_HOME:-$ROOT/.hf-cache}"
cd "$ROOT/von-rs"
echo "== Rust parity + unsafe audit + patterns on Metal =="
VON_WEIGHTS=checkpoints/von-1.1 VON_DEVICE=metal \
  cargo test --release -q --test parity --test unsafe_audit --test patterns -- --ignored --nocapture 2>&1 \
  | grep -E 'parity device|logits:|responses:|^test |test result|panicked|FAILED'
echo "== Cross-SDK check against von serve on Metal =="
VON_DEVICE=metal bash tools/cross_sdk_check.sh 2>&1 | grep -E '^==|golden over HTTP|SDK:|FAILED|Error'
[ "${SKIP_LATENCY:-0}" = 1 ] && exit 0
echo "== Rust latency (Metal) =="
VON_DEVICE=metal cargo run --release -q --example latency
echo "== Python latency (MPS) =="
cd "$ROOT" && uv run python von-rs/tools/bench_python.py --devices mps 2>&1 | grep -E '^#|^\|'
