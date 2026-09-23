# von-rs task runner. Recipes run from this directory; the scripts they call
# find the repo root themselves.

weights := "checkpoints/von-1.1"

# List the recipes.
default:
    @just --list

# Default suite (no weights): tests, clippy with and without default features, fmt.
test:
    cargo test
    cargo clippy --all-targets -- -D warnings
    cargo clippy --no-default-features --all-targets -- -D warnings
    cargo fmt --check

# Weights suite on CPU: golden parity, unsafe mapping audit, model-backed pattern/client tests.
gate-cpu:
    VON_WEIGHTS={{weights}} VON_DEVICE=cpu cargo test --release -- --ignored --nocapture

# Metal gate (needs a GPU, so run it in a normal terminal): parity, audit, patterns and the cross-SDK check on Metal.
gate-metal:
    SKIP_LATENCY=1 bash tools/metal_check.sh

# Server gate: the real `von serve` against the unmodified Python and JS SDKs (device: cpu or metal).
gate-server device="cpu":
    VON_DEVICE={{device}} bash tools/cross_sdk_check.sh

# Everything that runs without a GPU.
gate-all: test gate-cpu gate-server

# Metal gate plus Rust (Metal) vs Python (MPS) latency.
bench-metal:
    bash tools/metal_check.sh

# Rust latency only (device: cpu or metal).
latency device="cpu":
    VON_DEVICE={{device}} cargo run --release --example latency

# Regenerate the Python fixtures from the fork (the golden set also needs the model).
fixtures:
    cd .. && VON_PY_SRC=bug-fix-fork-von/src uv run python von-rs/tools/export_pyfixtures.py
    cd .. && VON_PY_SRC=bug-fix-fork-von/src uv run python von-rs/tools/export_protocol.py
    cd .. && VON_PY_SRC=bug-fix-fork-von/src HF_HOME=.hf-cache uv run python von-rs/tools/export_golden.py --out von-rs/tests/fixtures/golden/v1.json
