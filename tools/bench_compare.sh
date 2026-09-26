#!/usr/bin/env bash
# Python server vs Rust `von serve` on identical payloads: cold start, latency,
# throughput and memory. Results replace this device's section of bench/RESULTS.md.
#
# Usage (from anywhere):  bash von-rs/tools/bench_compare.sh [cpu|metal]
# Env: PURGE=1 purges the page cache before each server launch (true cold start;
#      runs `sudo purge`, so use a normal terminal). VON_PY_SRC (default: the fork).
# Metal needs a GPU, so run it from a normal terminal too.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
RS="$ROOT/von-rs"
DEVICE="${1:-cpu}"
export HF_HOME="${HF_HOME:-$ROOT/.hf-cache}"
export VON_PY_SRC="${VON_PY_SRC:-$ROOT/bug-fix-fork-von/src}"

(cd "$RS" && cargo build --release -q --bin von)
PURGE_FLAG=()
if [ "${PURGE:-0}" = 1 ]; then
  sudo -v  # ask for the password up front, not in the middle of a measurement
  PURGE_FLAG=(--purge)
fi
cd "$ROOT"
uv run python "$RS/tools/bench_compare.py" --device "$DEVICE" --von "$RS/target/release/von" ${PURGE_FLAG[@]+"${PURGE_FLAG[@]}"}
