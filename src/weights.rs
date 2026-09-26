//! Checkpoint resolution and weight loading.
//!
//! A checkpoint directory holds `option_marker.safetensors` (the full fine-tuned
//! encoder + scorer, converted from `option_marker.pt` by
//! `tools/convert_weights.py`), `config.json`, `tokenizer.json`, and optionally
//! `tokenizer_config.json` and `marker_calibration.json`.
//!
//! This module contains the crate's only `unsafe` block (`with_mapped_weights`).
//! See the "Unsafe code audit" section of README.md before changing it.

use std::path::{Path, PathBuf};

use candle_core::{DType, Device};
use candle_nn::VarBuilder;

use crate::error::{Result, VonError};

pub const HF_REPO: &str = "wfzyx/von";
pub const WEIGHTS_FILE: &str = "option_marker.safetensors";
pub const CONFIG_FILE: &str = "config.json";
pub const TOKENIZER_FILE: &str = "tokenizer.json";
pub const TOKENIZER_CONFIG_FILE: &str = "tokenizer_config.json";
pub const CALIBRATION_FILE: &str = "marker_calibration.json";

/// Searched in order, relative to the working directory. The first three match the
/// Python runtime's defaults.
pub const DEFAULT_CHECKPOINT_DIRS: [&str; 4] = [
    "checkpoints/von-1.2",
    "checkpoints/von-option-marker-universal",
    "checkpoints/von-option-marker",
    "checkpoints/von-1.1",
];

#[derive(Debug, Clone)]
pub struct Checkpoint {
    pub weights: PathBuf,
    pub config: PathBuf,
    pub tokenizer: PathBuf,
    pub tokenizer_config: Option<PathBuf>,
    pub calibration: Option<PathBuf>,
    /// Human-readable origin, for the load log line.
    pub source: String,
}

/// Finds a checkpoint: `explicit`, then `VON_CHECKPOINT_DIR`, then
/// `DEFAULT_CHECKPOINT_DIRS`, then the Hugging Face Hub. An explicit or env
/// directory is authoritative: if it is incomplete, loading fails rather than
/// silently falling through to another source.
pub fn resolve_checkpoint(explicit: Option<&Path>) -> Result<Checkpoint> {
    let pinned = explicit
        .map(|p| (p.to_path_buf(), "checkpoint dir"))
        .or_else(|| {
            std::env::var_os("VON_CHECKPOINT_DIR").map(|p| (PathBuf::from(p), "VON_CHECKPOINT_DIR"))
        });
    if let Some((dir, origin)) = pinned {
        return from_dir(&dir).map_err(|why| VonError::CheckpointNotFound {
            tried: vec![format!("{origin} {}: {why}", dir.display())],
        });
    }

    let mut tried = Vec::new();
    for dir in DEFAULT_CHECKPOINT_DIRS.map(Path::new) {
        if dir.join(WEIGHTS_FILE).exists() {
            return from_dir(dir).map_err(|why| VonError::CheckpointNotFound {
                tried: vec![format!("{}: {why}", dir.display())],
            });
        }
        tried.push(format!("{} (no {WEIGHTS_FILE})", dir.display()));
    }

    #[cfg(feature = "hub")]
    match from_hub() {
        Ok(ckpt) => return Ok(ckpt),
        Err(why) => tried.push(format!("Hugging Face Hub '{HF_REPO}': {why}")),
    }
    #[cfg(not(feature = "hub"))]
    tried.push("Hugging Face Hub (disabled: built without the `hub` feature)".into());

    Err(VonError::CheckpointNotFound { tried })
}

fn from_dir(dir: &Path) -> Result<Checkpoint, String> {
    let required = |name: &str| {
        let p = dir.join(name);
        if p.is_file() {
            Ok(p)
        } else {
            Err(format!("missing {name}"))
        }
    };
    let optional = |name: &str| Some(dir.join(name)).filter(|p| p.is_file());
    Ok(Checkpoint {
        weights: required(WEIGHTS_FILE)?,
        config: required(CONFIG_FILE)?,
        tokenizer: required(TOKENIZER_FILE)?,
        tokenizer_config: optional(TOKENIZER_CONFIG_FILE),
        calibration: optional(CALIBRATION_FILE),
        source: format!("local dir '{}'", dir.display()),
    })
}

/// Downloads into the standard HF cache (`HF_HUB_CACHE` / `HF_HOME`). Uses a
/// private tokio runtime, so it must not be called from inside an async context.
#[cfg(feature = "hub")]
fn from_hub() -> Result<Checkpoint, String> {
    let (owner, name) = HF_REPO.split_once('/').expect("HF_REPO is owner/name");
    let client = hf_hub::HFClientSync::new().map_err(|e| e.to_string())?;
    let repo = client.model(owner, name);
    let fetch = |file: &str| {
        repo.download_file()
            .filename(file)
            .send()
            .map_err(|e| format!("{file}: {e}"))
    };
    // Weights first: it is the file most likely to be missing (not yet published).
    let weights = fetch(WEIGHTS_FILE)?;
    Ok(Checkpoint {
        weights,
        config: fetch(CONFIG_FILE)?,
        tokenizer: fetch(TOKENIZER_FILE)?,
        tokenizer_config: fetch(TOKENIZER_CONFIG_FILE).ok(),
        calibration: fetch(CALIBRATION_FILE).ok(),
        source: format!("Hugging Face Hub '{HF_REPO}'"),
    })
}

/// Memory-maps the safetensors file and hands a `VarBuilder` over it to `build`.
///
/// The mapping lives only for the duration of `build`: candle copies every
/// tensor it loads into owned storage (heap on CPU, a new buffer on Metal), and
/// the mapping is released when the `VarBuilder` is dropped at the end of this
/// function. `build` must not stash the `VarBuilder` (or a clone of it) in its result.
///
/// This is the only `unsafe` code in von-rs. See README.md, "Unsafe code audit".
#[allow(unsafe_code)]
pub(crate) fn with_mapped_weights<T>(
    path: &Path,
    device: &Device,
    build: impl FnOnce(VarBuilder<'_>) -> Result<T>,
) -> Result<T> {
    // SAFETY: `from_mmaped_safetensors` is unsafe because a memory map is only
    // sound while no other process truncates or rewrites the underlying file;
    // if that happens the process can read torn data or fault (SIGBUS).
    // Invariants von-rs relies on:
    //   1. Checkpoint files are write-once artifacts: HF cache blobs are
    //      read-only (0444), and convert_weights.py writes a file once and does
    //      not touch it again.
    //   2. The mapping is read-only and scoped to this call: every tensor is
    //      copied out during `build` (candle `Tensor::from_slice` →
    //      `storage_from_slice`), and the VarBuilder (the only owner of the
    //      mapping) is dropped before this function returns.
    //   3. safetensors validates the header and every tensor's byte range
    //      against the file length before any slice is formed.
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[path], DType::F32, device) }?;
    build(vb)
}
