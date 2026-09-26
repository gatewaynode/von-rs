//! The option-marker decision model: ModernBERT encoder + MLP scorer
//! (port of `models/option_marker.py`).

pub mod masks;
pub mod modernbert;
pub mod packing;
pub mod scorer;

use std::path::Path;

use candle_core::{Device, IndexOp, Tensor};
use serde_json::Value;
use tokenizers::Tokenizer;

use crate::error::{Result, VonError};
use crate::weights::{Checkpoint, with_mapped_weights};
use masks::AttentionMode;
use modernbert::{Config, ModernBert};
use scorer::Scorer;

/// `OptionMarkerModel.__init__` overrides the checkpoint's 2048 positions with 8192.
pub const MAX_TOKENS: usize = 8192;
/// PyTorch `nn.LayerNorm` default, used by the scorer.
const SCORER_LAYER_NORM_EPS: f64 = 1e-5;

pub struct OptionMarkerModel {
    encoder: ModernBert,
    config: Config,
    scorer: Scorer,
    tokenizer: Tokenizer,
    mask_token: String,
    sep_token: String,
    mask_id: u32,
    device: Device,
}

impl OptionMarkerModel {
    pub fn load(ckpt: &Checkpoint, device: &Device) -> Result<Self> {
        let config = encoder_config(&read_json(&ckpt.config)?).map_err(|reason| {
            VonError::InvalidCheckpoint {
                path: ckpt.config.clone(),
                reason,
            }
        })?;

        let (encoder, scorer) = with_mapped_weights(&ckpt.weights, device, |vb| {
            // The encoder names the backbone `model.*`; the checkpoint calls it `encoder.*`.
            let encoder_vb = vb
                .clone()
                .rename_f(|name: &str| match name.strip_prefix("model.") {
                    Some(rest) => format!("encoder.{rest}"),
                    None => name.to_string(),
                });
            let encoder = ModernBert::load(encoder_vb, &config)?;
            let scorer = Scorer::load(vb.pp("scorer"), config.hidden_size, SCORER_LAYER_NORM_EPS)?;
            Ok((encoder, scorer))
        })?;

        let mut tokenizer =
            Tokenizer::from_file(&ckpt.tokenizer).map_err(|e| VonError::InvalidCheckpoint {
                path: ckpt.tokenizer.clone(),
                reason: e.to_string(),
            })?;
        // Python calls the tokenizer without truncation or padding.
        tokenizer
            .with_truncation(None)
            .map_err(|e| VonError::Tokenize(e.to_string()))?
            .with_padding(None);

        let tok_cfg = match &ckpt.tokenizer_config {
            Some(p) => read_json(p)?,
            None => Value::Null,
        };
        let special =
            |key: &str, default: &str| tok_cfg[key].as_str().unwrap_or(default).to_string();
        let mask_token = special("mask_token", "[MASK]");
        let sep_token = special("sep_token", "[SEP]");
        let mask_id =
            tokenizer
                .token_to_id(&mask_token)
                .ok_or_else(|| VonError::InvalidCheckpoint {
                    path: ckpt.tokenizer.clone(),
                    reason: format!("tokenizer has no {mask_token} token"),
                })?;

        Ok(Self {
            encoder,
            config,
            scorer,
            tokenizer,
            mask_token,
            sep_token,
            mask_id,
            device: device.clone(),
        })
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Packs state, question and option descriptions into one marker sequence.
    pub fn pack(&self, state: &str, question: &str, options: &[String]) -> String {
        packing::pack_sequence(state, question, options, &self.mask_token, &self.sep_token)
    }

    /// Token ids with special tokens, as the model sees them.
    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        let enc = self
            .tokenizer
            .encode(text, true)
            .map_err(|e| VonError::Tokenize(e.to_string()))?;
        Ok(enc.get_ids().to_vec())
    }

    /// Token count without special tokens (the calibration length feature).
    pub fn count_tokens(&self, text: &str) -> Result<usize> {
        let enc = self
            .tokenizer
            .encode(text, false)
            .map_err(|e| VonError::Tokenize(e.to_string()))?;
        Ok(enc.len())
    }

    /// One forward pass over a packed sequence → one raw logit per option.
    pub fn option_logits(
        &self,
        packed: &str,
        n_options: usize,
        mode: AttentionMode,
    ) -> Result<Vec<f32>> {
        let ids = self.encode(packed)?;
        if ids.len() > MAX_TOKENS {
            return Err(VonError::InputTooLong {
                tokens: ids.len(),
                max: MAX_TOKENS,
            });
        }
        let positions: Vec<u32> = ids
            .iter()
            .enumerate()
            .filter(|(_, t)| **t == self.mask_id)
            .map(|(i, _)| i as u32)
            .collect();
        if positions.len() != n_options {
            return Err(VonError::MaskMismatch {
                expected: n_options,
                found: positions.len(),
                token: self.mask_token.clone(),
            });
        }
        // Metal returns some objects (command buffers, encoders) autoreleased. They are
        // freed only when the calling thread's autorelease pool drains, and worker
        // threads (e.g. tokio's blocking pool) never drain one, so drain per pass.
        #[cfg(feature = "metal")]
        if self.device.is_metal() {
            return objc2::rc::autoreleasepool(|_| self.forward(&ids, &positions, mode));
        }
        self.forward(&ids, &positions, mode)
    }

    fn forward(&self, ids: &[u32], positions: &[u32], mode: AttentionMode) -> Result<Vec<f32>> {
        let input = Tensor::new(ids, &self.device)?.unsqueeze(0)?;
        let mask_positions: Vec<usize> = positions.iter().map(|&p| p as usize).collect();
        let (position_ids, masks) =
            masks::build(mode, &self.config, &mask_positions, ids.len(), &self.device)?;
        let hidden = self.encoder.forward(&input, &position_ids, &masks)?.i(0)?;
        let reps = hidden.index_select(&Tensor::new(positions, &self.device)?, 0)?;
        Ok(self.scorer.forward(&reps)?.to_vec1()?)
    }
}

/// Von's `config.json` is in transformers-5 form (`rope_parameters`, `norm_eps`);
/// the encoder wants the flat ModernBERT fields.
fn encoder_config(raw: &Value) -> Result<Config, String> {
    let usize_field = |k: &str| {
        raw[k]
            .as_u64()
            .map(|v| v as usize)
            .ok_or(format!("missing integer '{k}'"))
    };
    let rope_theta = |kind: &str| {
        raw["rope_parameters"][kind]["rope_theta"]
            .as_f64()
            .ok_or(format!("missing rope_parameters.{kind}.rope_theta"))
    };
    Ok(Config {
        vocab_size: usize_field("vocab_size")?,
        hidden_size: usize_field("hidden_size")?,
        num_hidden_layers: usize_field("num_hidden_layers")?,
        num_attention_heads: usize_field("num_attention_heads")?,
        intermediate_size: usize_field("intermediate_size")?,
        max_position_embeddings: MAX_TOKENS,
        layer_norm_eps: raw["norm_eps"].as_f64().ok_or("missing 'norm_eps'")?,
        global_attn_every_n_layers: usize_field("global_attn_every_n_layers")?,
        global_rope_theta: rope_theta("full_attention")?,
        local_attention: usize_field("local_attention")?,
        local_rope_theta: rope_theta("sliding_attention")?,
    })
}

fn read_json(path: &Path) -> Result<Value> {
    let text = std::fs::read_to_string(path).map_err(|source| VonError::Io {
        path: path.into(),
        source,
    })?;
    serde_json::from_str(&text).map_err(|e| VonError::InvalidCheckpoint {
        path: path.into(),
        reason: e.to_string(),
    })
}
