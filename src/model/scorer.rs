//! Calibrated MLP scoring head (port of `OptionMarkerScorer`).

use candle_core::{D, Module, Result, Tensor};
use candle_nn::{LayerNorm, Linear, VarBuilder, layer_norm, linear};

/// LayerNorm → Linear(H, H/2) → GELU (erf) → LayerNorm → Linear(H/2, 1).
/// Dropout is a no-op at inference.
pub struct Scorer {
    input_norm: LayerNorm,
    dense: Linear,
    norm: LayerNorm,
    out_proj: Linear,
}

impl Scorer {
    pub fn load(vb: VarBuilder, hidden_size: usize, eps: f64) -> Result<Self> {
        Ok(Self {
            input_norm: layer_norm(hidden_size, eps, vb.pp("input_norm"))?,
            dense: linear(hidden_size, hidden_size / 2, vb.pp("dense"))?,
            norm: layer_norm(hidden_size / 2, eps, vb.pp("norm"))?,
            out_proj: linear(hidden_size / 2, 1, vb.pp("out_proj"))?,
        })
    }

    /// `(K, H)` option representations → `(K,)` logits.
    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let h = self
            .dense
            .forward(&self.input_norm.forward(xs)?)?
            .gelu_erf()?;
        self.out_proj
            .forward(&self.norm.forward(&h)?)?
            .squeeze(D::Minus1)
    }
}
