//! The `Von` handle: model version aliases, loading, and the process-wide
//! instance (port of `engine.py`).

use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use candle_core::Device;
use indexmap::IndexMap;
use serde_json::Value;

use crate::backend::Backend;
use crate::calibration::Calibration;
use crate::device::{describe, resolve_device};
use crate::error::{Result, VonError};
use crate::model::OptionMarkerModel;
use crate::types::{Question, SystemOneResponse};
use crate::weights::resolve_checkpoint;

/// Von ships exactly one model, identified by version number only.
pub const VON_VERSION: &str = "1.2";
/// Every response is stamped with this id, whatever alias the caller asked for.
pub const VON_MODEL_ID: &str = "von-1.2.0";
/// Names that resolve to the current model. `von-1.1` stays accepted: 1.2 is the
/// same model family retrained.
pub const MODEL_ALIASES: [&str; 8] = [
    "von-1.2",
    "1.2",
    "von-1.1",
    "1.1",
    "von",
    "default",
    "latest",
    "von-latest",
];

#[derive(Debug, Clone, Default)]
pub struct LoadOptions {
    /// Model alias; defaults to `VON_BACKEND`, then `von-1.2`.
    pub model: Option<String>,
    /// Checkpoint dir; defaults to `VON_CHECKPOINT_DIR`, the local defaults, then the Hub.
    pub checkpoint_dir: Option<PathBuf>,
    /// `auto` | `metal` | `mps` | `cpu`; defaults to `VON_DEVICE`, then `auto`.
    pub device: Option<String>,
}

pub struct Von {
    backend: Backend,
    device: Device,
}

static GLOBAL: Mutex<Option<Arc<Von>>> = Mutex::new(None);

impl Von {
    /// Loads the model eagerly: weights, tokenizer and calibration.
    pub fn load(opts: LoadOptions) -> Result<Self> {
        let model_name = opts
            .model
            .or_else(|| std::env::var("VON_BACKEND").ok())
            .unwrap_or_else(|| format!("von-{VON_VERSION}"));
        check_model_alias(&model_name)?;

        let device = resolve_device(opts.device.as_deref())?;
        let checkpoint = resolve_checkpoint(opts.checkpoint_dir.as_deref())?;
        let model = OptionMarkerModel::load(&checkpoint, &device)?;
        let calibration = Calibration::from_file(checkpoint.calibration.as_deref());
        tracing::info!(
            "Loaded {VON_MODEL_ID} weights from {} on {} ({})",
            checkpoint.source,
            describe(&device),
            calibration.describe()
        );
        Ok(Self {
            backend: Backend::new(model, calibration),
            device,
        })
    }

    /// The process-wide instance, loaded with default options on first use.
    /// A failed load is not cached; the next call tries again.
    pub fn global() -> Result<Arc<Von>> {
        let mut slot = GLOBAL
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if let Some(von) = slot.as_ref() {
            return Ok(von.clone());
        }
        let von = Arc::new(Von::load(LoadOptions::default())?);
        *slot = Some(von.clone());
        Ok(von)
    }

    pub fn evaluate(
        &self,
        state: &Value,
        questions: &IndexMap<String, Question>,
    ) -> Result<SystemOneResponse> {
        self.backend.evaluate(state, questions, VON_MODEL_ID)
    }

    pub fn backend(&self) -> &Backend {
        &self.backend
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn device_description(&self) -> &'static str {
        describe(&self.device)
    }
}

/// Accepts any alias of the current model (case- and whitespace-insensitive).
pub fn check_model_alias(name: &str) -> Result<()> {
    let normalized = name.trim().to_lowercase();
    if MODEL_ALIASES.contains(&normalized.as_str()) {
        return Ok(());
    }
    let mut aliases = MODEL_ALIASES.to_vec();
    aliases.sort_unstable();
    Err(VonError::UnknownModel {
        name: normalized,
        version: VON_VERSION,
        aliases: aliases.join(", "),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn aliases() {
        for a in ["von-1.2", "1.2", "von-1.1", " LATEST ", "von"] {
            assert!(check_model_alias(a).is_ok(), "{a}");
        }
        let err = check_model_alias("von-1.0").unwrap_err().to_string();
        assert!(
            err.contains("Unknown model 'von-1.0'") && err.contains("1.1, 1.2, default, latest"),
            "{err}"
        );
    }
}
