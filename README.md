# von-rs

A native Rust runtime for [Von](https://github.com/wfzyx/von), the open-source
System One decision model, for **macOS** (Apple Silicon Metal, or CPU with
Accelerate). Von's Python package (`src/von/` in that repository) is the
reference implementation, and von-rs matches it numerically: see [Parity](#parity).

**Status:** a complete port of the Python runtime: the inference engine, the
`decide`/`judge`/`rate` helpers, patterns, presets, the remote client, and a
drop-in `von` server and CLI, all verified against Python on CPU and Metal.
It is not tuned yet: on Metal it is 1.4–1.7× slower than PyTorch MPS on short
requests and 3.7× at about 800 tokens (see `bench/BASELINE.md`). Performance
work comes after the port is complete, followed by an async API.

## Setup

von-rs loads a converted checkpoint directory. `option_marker.safetensors` is not
yet published to the Hugging Face repo, so create one from the Python weights
(see [Development layout](#development-layout)), from the Von checkout's root:

```bash
uv run python von-rs/tools/convert_weights.py --out von-rs/checkpoints/von-1.1
```

The conversion is verified bit for bit. `checkpoints/` is gitignored.

## Server and CLI

`cargo install --path .` (or `cargo build --release`) builds the `von` binary, a
drop-in replacement for the Python `von` command:

```bash
von serve --port 8000                        # POST /v1/systemone, GET /health, GET /v1/models
von decide "I was charged twice" -c refund,bug_report,feature_request
von judge "Prod is down" -i "Is there an outage?" --pos Outage --neg Healthy
von rate "Minor typo" -l Low,Medium,High
von eval request.json                        # {"state": ..., "questions": {...}, "model": ...}
```

Flags, defaults, messages, exit codes and JSON output match `src/von/cli.py`, and
the server's responses match the FastAPI app, byte for byte where clients can see
them. `von serve` loads the model before it binds the port.

## Library use

```rust
use indexmap::IndexMap;
use von::{Choice, LoadOptions, Question, Von};

let von = Von::load(LoadOptions::default())?;          // or Von::global()
let questions: IndexMap<String, Question> = serde_json::from_str(r#"{
    "intent": {"type": "choice", "instructions": "What does the user want?",
               "criteria": {"refund": "Wants money back", "bug": "Reports a defect"}}
}"#)?;
let response = von.evaluate(&"I was charged twice".into(), &questions)?;
println!("{}", serde_json::to_string_pretty(&response)?);
```

Requests and responses use the same JSON shapes as `/v1/systemone`.

The helpers, patterns and presets take any `Decider`: a loaded `Von`, or a
`VonClient` for a remote server (feature `client`, on by default).

```rust
use serde_json::json;
use von::patterns::confidence_gate;
use von::presets::triage_preset;
use von::{ClientOptions, VonClient, decide, judge};

let answer = decide(&von, &json!("I was charged twice"), ["refund", "bug"], None, None)?;
let p_urgent = judge(&von, &json!("Prod is down!"), "Is this urgent?", None, None)?;
let gated = confidence_gate(&von, &json!("Refund please"), &triage_preset(), 0.8)?;

let remote = VonClient::new(ClientOptions::default())?;   // VON_BASE_URL, VON_API_KEY
let answer = decide(&remote, &json!("I was charged twice"), ["refund", "bug"], None, None)?;
```

`VonClient` is blocking; do not call it from inside an async runtime.

## Configuration

| Variable | Meaning | Default |
|---|---|---|
| `VON_CHECKPOINT_DIR` | Checkpoint directory. If it is set, it must be complete: there is no fallback | unset |
| `VON_DEVICE` | `auto`, `metal` (alias `mps`) or `cpu`. `cuda`/`rocm`/`dml` are rejected | `auto` (Metal if visible, else CPU) |
| `VON_BACKEND` | Model alias: `von-1.1`, `1.1`, `von`, `default`, `latest`, `von-latest` | `von-1.1` |
| `HF_HOME` / `HF_HUB_CACHE` | Hub cache, shared with Python | `~/.cache/huggingface` |
| `VON_API_KEY` (server) | Bearer token `von serve` requires on `/v1/systemone`; unset or empty disables auth | unset |
| `VON_CORS_ORIGINS` (server) | Comma-separated allowed origins. `*` alone allows any origin without credentials; a list enables credentials | `*` |
| `VON_BASE_URL` | `VonClient` server root | `http://localhost:8000` |
| `VON_API_KEY`, then `TYPESAFE_API_KEY` | `VonClient` Bearer token | none |

Checkpoint search order: `VON_CHECKPOINT_DIR`, then `checkpoints/von-option-marker-universal`,
`checkpoints/von-option-marker` and `checkpoints/von-1.1` (relative to the working
directory), then the Hub repo `wfzyx/von`. A failed load lists every location tried.

## Tests

```bash
cargo test                                             # unit + Python-oracle + unsafe inventory; no weights needed
VON_WEIGHTS=checkpoints/von-1.1 cargo test --release -- --ignored --nocapture   # golden parity + mapping test
VON_WEIGHTS=checkpoints/von-1.1 VON_DEVICE=metal cargo test --release -- --ignored --nocapture
```

Fixtures are generated from the Python runtime by `tools/export_pyfixtures.py`
(number and text formatting, calibration, presets), `tools/export_golden.py`
(86 requests, 94 forward passes, on torch CPU fp32) and `tools/export_protocol.py`
(the FastAPI server's and click CLI's exact responses, with a fake engine).

The end-to-end check runs the real `von serve` against the unmodified Python and JS
SDKs (needs the converted checkpoint, `uv` and `bun`):

```bash
bash tools/cross_sdk_check.sh                 # VON_DEVICE=metal for the GPU path
```

## Development layout

The Rust crate builds and tests on its own. The tools in `tools/` that talk to
Python (weight conversion, fixture export, the cross-SDK check and the benchmark)
expect this repository to be cloned as `von-rs/` inside a checkout of the Python
Von repository, whose `uv` environment they run in. Two variables point them at
other sources:

| Variable | Used by | Default |
|---|---|---|
| `VON_PY_SRC` | fixture exporters, cross-SDK check | the enclosing checkout's `src/` (`bug-fix-fork-von/src` for the cross-SDK check) |
| `SDK_JS` | cross-SDK check | `bug-fix-fork-von/js` next to `von-rs/` |

The `justfile` wraps the common runs: `just test`, `just gate-cpu`,
`just gate-server [cpu|metal]`, `just gate-metal`, `just bench-metal` and
`just fixtures`.

## Parity

| Stage | CPU (2026-09-22) | Metal (2026-09-22) |
|---|---|---|
| Packed model input | 94/94 identical to Python | 94/94 |
| Token ids | 94/94 identical | 94/94 |
| Raw logits (tolerance 1e-3) | worst Δ 6.4e-5 | worst Δ 9.0e-5 |
| Responses | 86/86 match; worst probability Δ 1.0e-4 | 86/86; worst Δ 1.0e-4 |
| Over HTTP (`von serve` + Python SDK) | 86/86; worst probability Δ 1.0e-4 | 86/86; worst Δ 1.0e-4 |

## Differences from the Python runtime

- **Loading is eager.** `Von::load` loads everything up front. Python loads on the first request.
- **Over-length input is an error.** Inputs over 8,192 tokens (the model's trained
  context) return `InputTooLong` (HTTP 422, exit 1 in the CLI). Python runs them
  anyway, with only a warning, and answers with unknown quality. This is a
  deliberate choice.
- **A literal `[MASK]` in the input is an error.** It returns `MaskMismatch` instead
  of silently adding a bogus option slot.
- **The legacy `pos_criteria`/`neg_criteria` spellings on Noul** are folded into
  `criteria` with a deprecation warning. This matches the upstream Python fix; older
  Python releases silently dropped them.
- **Two known formatting gaps** remain, both rare and both only in dict or list
  `state` values: unassigned Unicode code points inside nested strings are not
  escaped as Python's `repr` would, and integers outside the 64-bit range become
  floats (Python keeps them exact).
- **The helpers take an explicit `Decider`.** Python's `decide`/`judge`/`rate` and
  patterns use a hidden process-wide client, and `VonClient(local=True)` wraps the
  engine; in Rust the local engine is `Von` itself. The patterns' `backend=`
  argument is dropped, since Von serves one model.
- **Answers are typed.** If a response lacks the requested answer, or has the wrong
  type (only possible with a misbehaving remote server), the helpers return
  `UnexpectedResponse`. Python's `judge` returns `0.0` in that case.
- **`route` returns `Routed::Handled(value)` or `Routed::Unhandled(answer)`**, and
  takes handlers as a slice of `(option id, &dyn Fn)` pairs.
- **`two_stage_choice`'s option template** supports `{category}`, `{{` and `}}`
  only; Python's full `str.format` syntax is not reproduced.
- **`VonClient`'s timeout covers the whole request.** httpx applies 30 s to each
  phase (connect, read, write) separately.
- **Server:** `VON_API_KEY` and `VON_CORS_ORIGINS` are read once at startup (Python
  reads the key on every request). Question-level validation errors are still 422
  with a string `detail`, but the wording is serde's, not pydantic's; for malformed
  JSON only `ctx.error` (the decoder's message) differs. At most one inference runs
  at a time on Metal (the CPU count on CPU); further requests queue.
- **CLI:** stdout carries only the JSON result, and logs go to stderr (Python prints
  its load line to stdout). Unexpected failures print `Error: …` and exit 1 instead
  of a traceback. `--version` prints `1.1.0`; Python's CLI still says `1.0.0`.
  `--device` accepts `auto`, `metal`/`mps` and `cpu`. `serve --reload` is accepted
  and ignored with a warning.

## Unsafe code audit

von-rs contains **exactly one** `unsafe` block. Everything else is safe Rust.

### The site

`src/weights.rs`, function `with_mapped_weights`: it calls
`candle_nn::VarBuilder::from_mmaped_safetensors` to memory-map the 1.5 GB
`option_marker.safetensors` while the model is built.

### Why it exists

candle marks memory-mapped loading `unsafe` because a memory map is only sound
while no other process truncates or rewrites the file; if one does, reads can
return torn data or fault with SIGBUS. The safe alternative,
`VarBuilder::from_buffered_safetensors`, first reads the whole file into a heap
buffer and then copies every tensor again. That means reading 1.5 GB of extra
bytes and roughly **2× peak memory** (about 3 GB) during startup, in a runtime
whose goals include fast cold start. The project owner approved this
single block on 2026-09-22, on the condition that it stays documented, tracked
and audited.

### Why it is sound here

1. **The files are write-once.** Hub cache blobs are read-only (0444), and
   `tools/convert_weights.py` writes a checkpoint once and never touches it again.
2. **The mapping is scoped to loading.** candle copies every tensor it loads into
   owned storage: `Tensor::from_slice` goes through `Device::storage_from_slice`,
   which makes a heap `Vec` on CPU and a new buffer on Metal. The `VarBuilder`
   that owns the mapping is dropped before `with_mapped_weights` returns, so the
   file is never read after `Von::load`.
3. **Bounds are validated.** safetensors checks the header and every tensor's
   byte range against the file length before any slice is formed.

### How it stays tracked

- `Cargo.toml` sets `unsafe_code = "deny"` crate-wide, and clippy's
  `undocumented_unsafe_blocks = "deny"`. The site carries the only
  `#[allow(unsafe_code)]` and a `// SAFETY:` comment.
- `tests/unsafe_audit.rs::static_inventory` runs in every `cargo test`. It fails
  if any other `unsafe` block or `allow(unsafe_code)` appears anywhere in `src/`,
  `tests/`, `examples/` or `benches/`; if the SAFETY comment or either lint is
  removed; or if this section stops naming the site.
- `tests/unsafe_audit.rs::mapping_is_released_after_load` checks invariant 2 at
  runtime. It loads from a copy-on-write clone of the weights, truncates the
  clone to zero bytes, runs inference, and requires identical logits.

### Re-audit triggers

Re-audit, and add a row to the log below, when any of these happen:

- `src/weights.rs` changes, or anything else starts holding a `VarBuilder` beyond model construction.
- candle is upgraded. Invariant 2 depends on candle copying tensors out of the
  mapping, so re-read `candle-core/src/safetensors.rs` (`convert_slice`) and run
  `mapping_is_released_after_load` on CPU **and** Metal.
- Checkpoints start being written or updated in place while a process may be loading them.

### Audit log

| Date | candle | Change | Checked | Result |
|---|---|---|---|---|
| 2026-09-22 | 0.11.0 | Introduced | SAFETY invariants 1–3 read against candle source; `static_inventory` passes; `mapping_is_released_after_load` passes on CPU and Metal | Approved by project owner |
