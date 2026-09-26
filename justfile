# von-rs task runner. Recipes run from this directory; the scripts they call
# find the repo root themselves.

# Model storage: set VON_MODELS_DIR (environment, or a `VON_MODELS_DIR=/path` line in
# ~/.config/von/von.env or the gitignored `.env`) to keep models outside the checkout.
# See tools/model_paths.sh.
export VON_MODELS_DIR := `. tools/model_paths.sh && printf %s "${VON_MODELS_DIR:-}"`
export VON_WEIGHTS := `. tools/model_paths.sh && printf %s "$VON_WEIGHTS"`
export VON_WEIGHTS_V11 := `. tools/model_paths.sh && printf %s "$VON_WEIGHTS_V11"`
export HF_HOME := `. tools/model_paths.sh && printf %s "$HF_HOME"`

# List the recipes.
default:
    @just --list

# Default suite (no weights): tests, clippy with and without default features, fmt.
test:
    cargo test
    cargo clippy --all-targets -- -D warnings
    cargo clippy --no-default-features --all-targets -- -D warnings
    cargo fmt --check

# Weights suite on CPU: golden parity (1.2, plus 1.1 when converted), option invariance, unsafe mapping audit, model-backed pattern/client tests.
gate-cpu:
    VON_DEVICE=cpu cargo test --release -- --ignored --nocapture

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

# Optional filter, e.g. `just bench metal /512`.
# Criterion latency benchmarks: choice, zero-shot noul and 10-level score at 64/512/4096 tokens.
bench device="cpu" filter="":
    VON_DEVICE={{device}} cargo bench --bench latency -- {{filter}}

# `PURGE=1 just bench-compare metal` measures a true cold start (sudo purge; normal terminal).
# Python server vs `von serve`: cold start, latency, throughput, memory -> bench/RESULTS.md.
bench-compare device="cpu":
    bash tools/bench_compare.sh {{device}}

# Soak `von serve`: 10k requests at concurrency 64, checking statuses and RSS growth.
soak device="cpu" requests="10000":
    cargo build --release -q --bin von
    cd .. && uv run python von-rs/tools/soak.py --device {{device}} --requests {{requests}}

# Show where the models live (VON_MODELS_DIR, VON_WEIGHTS, VON_WEIGHTS_V11, HF_HOME).
models:
    @echo "VON_MODELS_DIR=${VON_MODELS_DIR:-(unset: models stay in the checkout)}"
    @echo "VON_WEIGHTS=$VON_WEIGHTS"
    @echo "VON_WEIGHTS_V11=$VON_WEIGHTS_V11"
    @echo "HF_HOME=$HF_HOME"

# Download the pinned Python checkpoint into HF_HOME and convert it: 1.2 into VON_WEIGHTS, 1.1 into VON_WEIGHTS_V11.
convert version="1.2":
    cd .. && uv run python von-rs/tools/convert_weights.py --revision {{version}} \
        --out "{{ if version == "1.1" { "$VON_WEIGHTS_V11" } else { "$VON_WEIGHTS" } }}"

# Regenerate the Python fixtures from the fork (the golden set also needs the model).
fixtures:
    cd .. && VON_PY_SRC=bug-fix-fork-von/src uv run python von-rs/tools/export_pyfixtures.py
    cd .. && VON_PY_SRC=bug-fix-fork-von/src uv run python von-rs/tools/export_protocol.py
    cd .. && dir="$(uv run python von-rs/tools/hub_pins.py 1.2)" && VON_PY_SRC=bug-fix-fork-von/src \
        uv run python von-rs/tools/export_golden.py --checkpoint-dir "$dir" --out von-rs/tests/fixtures/golden/v1_2.json
    cd .. && dir="$(uv run python von-rs/tools/hub_pins.py 1.1)" && VON_PY_SRC=bug-fix-fork-von/src \
        uv run python von-rs/tools/export_golden.py --checkpoint-dir "$dir" --out von-rs/tests/fixtures/golden/v1.json
