use std::path::PathBuf;

/// Every failure the Von runtime can report.
#[derive(Debug, thiserror::Error)]
pub enum VonError {
    /// No usable checkpoint was found. Lists every location that was tried.
    #[error(
        "Failed to load Von decision weights; tried:\n{}\nRun `uv run python von-rs/tools/convert_weights.py --out <dir>` and set VON_CHECKPOINT_DIR=<dir>.",
        .tried.iter().map(|t| format!("  - {t}")).collect::<Vec<_>>().join("\n")
    )]
    CheckpointNotFound { tried: Vec<String> },

    #[error("failed to read {path}: {source}")]
    Io {
        path: PathBuf,
        source: std::io::Error,
    },

    #[error("invalid checkpoint file {path}: {reason}")]
    InvalidCheckpoint { path: PathBuf, reason: String },

    #[error("tokenizer error: {0}")]
    Tokenize(String),

    #[error("inference error: {0}")]
    Inference(#[from] candle_core::Error),

    #[error("input is {tokens} tokens; Von accepts at most {max}")]
    InputTooLong { tokens: usize, max: usize },

    /// A literal mask token in user text would add bogus option slots.
    #[error(
        "packed input has {found} {token} markers for {expected} options; the state, instructions and options must not contain the literal {token} token"
    )]
    MaskMismatch {
        expected: usize,
        found: usize,
        token: String,
    },

    #[error(
        "Unknown model '{name}'. Von {version} is the only model; accepted aliases: {aliases}."
    )]
    UnknownModel {
        name: String,
        version: &'static str,
        aliases: String,
    },

    #[error("device '{0}' is not supported on macOS; use auto, metal, mps or cpu")]
    UnsupportedDevice(String),

    #[error("device '{requested}' is not available: {reason}")]
    DeviceUnavailable { requested: String, reason: String },

    #[error("{0}")]
    InvalidQuestion(String),

    /// A response did not carry the answer a helper asked for, or carried the
    /// wrong answer type (e.g. from a misbehaving remote server).
    #[error("unexpected response: {0}")]
    UnexpectedResponse(String),

    /// The remote server answered with a non-2xx status.
    #[error("HTTP {status} from {url}: {body}")]
    Http {
        status: u16,
        url: String,
        body: String,
    },

    /// The HTTP request could not be sent or its response could not be read.
    #[error("request to {url} failed: {reason}")]
    Request { url: String, reason: String },
}

pub type Result<T, E = VonError> = std::result::Result<T, E>;
