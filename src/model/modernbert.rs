//! ModernBERT encoder backbone.
//!
//! Vendored from candle-transformers 0.11.0 `src/models/modernbert.rs`
//! (<https://github.com/huggingface/candle>, MIT OR Apache-2.0, copyright the candle
//! authors). Changes from the original:
//!
//! - `forward` takes the attention masks and the position ids from the caller instead of
//!   building a padding mask and positions `0..n` itself, so a sequence can use
//!   per-option masks and restarted positions.
//! - RoPE looks up its cos/sin rows at the given position ids.
//! - Only the backbone is kept (no masked-LM or classification heads), and `Config` is
//!   built by the caller rather than deserialized.

use candle_core::{D, DType, Device, Result, Tensor};
use candle_nn::{
    Embedding, LayerNorm, Linear, Module, VarBuilder, embedding, layer_norm_no_bias,
    linear_no_bias, ops::softmax,
};

use std::sync::Arc;

#[derive(Debug, Clone, PartialEq)]
pub struct Config {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub intermediate_size: usize,
    pub max_position_embeddings: usize,
    pub layer_norm_eps: f64,
    pub global_attn_every_n_layers: usize,
    pub global_rope_theta: f64,
    pub local_attention: usize,
    pub local_rope_theta: f64,
}

/// Additive attention masks (`0` = attend, `-inf` = blocked), each of shape
/// `(seq_len, seq_len)`: row = query token, column = key token.
pub struct AttentionMasks {
    /// For global-attention layers; `None` lets every token attend to every token.
    pub global: Option<Tensor>,
    /// For sliding-window layers.
    pub sliding: Tensor,
}

impl AttentionMasks {
    /// The plain encoder's masks for one unpadded sequence: global layers see every
    /// token, sliding layers only tokens within `local_attention / 2` positions.
    pub fn full(config: &Config, seq_len: usize, device: &Device) -> Result<Self> {
        Ok(Self {
            global: None,
            sliding: sliding_window_mask(seq_len, config.local_attention / 2, device)?,
        })
    }
}

/// `-inf` where `|i - j| > max_distance`, else `0`.
pub fn sliding_window_mask(seq_len: usize, max_distance: usize, device: &Device) -> Result<Tensor> {
    let mask: Vec<_> = (0..seq_len)
        .flat_map(|i| {
            (0..seq_len).map(move |j| {
                if i.abs_diff(j) > max_distance {
                    f32::NEG_INFINITY
                } else {
                    0.
                }
            })
        })
        .collect();
    Tensor::from_slice(&mask, (seq_len, seq_len), device)
}

#[derive(Debug, Clone)]
struct RotaryEmbedding {
    sin: Tensor,
    cos: Tensor,
}

impl RotaryEmbedding {
    fn new(dtype: DType, config: &Config, rope_theta: f64, dev: &Device) -> Result<Self> {
        let dim = config.hidden_size / config.num_attention_heads;
        let inv_freq: Vec<_> = (0..dim)
            .step_by(2)
            .map(|i| 1f32 / rope_theta.powf(i as f64 / dim as f64) as f32)
            .collect();
        let inv_freq_len = inv_freq.len();
        let inv_freq = Tensor::from_vec(inv_freq, (1, inv_freq_len), dev)?.to_dtype(dtype)?;
        let max_seq_len = config.max_position_embeddings;
        let t = Tensor::arange(0u32, max_seq_len as u32, dev)?
            .to_dtype(dtype)?
            .reshape((max_seq_len, 1))?;
        let freqs = t.matmul(&inv_freq)?;
        Ok(Self {
            sin: freqs.sin()?,
            cos: freqs.cos()?,
        })
    }

    /// The cos/sin rows for `position_ids`, one row per token.
    fn at(&self, position_ids: &Tensor) -> Result<Rope> {
        Ok(Rope {
            cos: self.cos.index_select(position_ids, 0)?,
            sin: self.sin.index_select(position_ids, 0)?,
        })
    }
}

/// Cos/sin tables gathered for one sequence's positions.
struct Rope {
    cos: Tensor,
    sin: Tensor,
}

impl Rope {
    fn apply(&self, q: &Tensor, k: &Tensor) -> Result<(Tensor, Tensor)> {
        let q_embed = candle_nn::rotary_emb::rope(&q.contiguous()?, &self.cos, &self.sin)?;
        let k_embed = candle_nn::rotary_emb::rope(&k.contiguous()?, &self.cos, &self.sin)?;
        Ok((q_embed, k_embed))
    }
}

#[derive(Clone)]
struct ModernBertAttention {
    qkv: Linear,
    proj: Linear,
    num_attention_heads: usize,
    attention_head_size: usize,
}

impl ModernBertAttention {
    fn load(vb: VarBuilder, config: &Config) -> Result<Self> {
        let num_attention_heads = config.num_attention_heads;
        let attention_head_size = config.hidden_size / config.num_attention_heads;

        let qkv = linear_no_bias(config.hidden_size, config.hidden_size * 3, vb.pp("Wqkv"))?;
        let proj = linear_no_bias(config.hidden_size, config.hidden_size, vb.pp("Wo"))?;

        Ok(Self {
            qkv,
            proj,
            num_attention_heads,
            attention_head_size,
        })
    }

    fn forward(
        &self,
        hidden_states: &Tensor,
        attention_mask: Option<&Tensor>,
        rope: &Rope,
    ) -> Result<Tensor> {
        let xs = hidden_states.clone();
        let (b, seq_len, d) = xs.dims3()?;
        let qkv = xs
            .apply(&self.qkv)?
            .reshape((
                b,
                seq_len,
                3,
                self.num_attention_heads,
                self.attention_head_size,
            ))?
            .permute((2, 0, 3, 1, 4))?;

        let q = qkv.get(0)?;
        let k = qkv.get(1)?;
        let v = qkv.get(2)?;

        let (q, k) = rope.apply(&q, &k)?;

        let scale = (self.attention_head_size as f64).powf(-0.5);
        let q = (q * scale)?;

        let att = q.matmul(&k.transpose(D::Minus2, D::Minus1)?)?;

        let att = match attention_mask {
            Some(mask) => att.broadcast_add(mask)?,
            None => att,
        };
        let att = softmax(&att, D::Minus1)?;

        let xs = att.matmul(&v)?;

        let xs = xs.transpose(1, 2)?.reshape((b, seq_len, d))?;
        let xs = xs.apply(&self.proj)?;
        let xs = xs.reshape((b, seq_len, d))?;

        Ok(xs)
    }
}

#[derive(Clone)]
struct ModernBertMLP {
    wi: Linear,
    wo: Linear,
}

impl ModernBertMLP {
    fn load(vb: VarBuilder, config: &Config) -> Result<Self> {
        let wi = linear_no_bias(
            config.hidden_size,
            config.intermediate_size * 2,
            vb.pp("Wi"),
        )?;
        let wo = linear_no_bias(config.intermediate_size, config.hidden_size, vb.pp("Wo"))?;
        Ok(Self { wi, wo })
    }
}

impl Module for ModernBertMLP {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let xs = xs.apply(&self.wi)?;
        let xs = xs.chunk(2, D::Minus1)?;
        let xs = (&xs[0].gelu_erf()? * &xs[1])?.apply(&self.wo)?; // GeGLU
        Ok(xs)
    }
}

#[derive(Clone)]
struct ModernBertLayer {
    attn: ModernBertAttention,
    mlp: ModernBertMLP,
    attn_norm: Option<LayerNorm>,
    mlp_norm: LayerNorm,
    uses_local_attention: bool,
}

impl ModernBertLayer {
    fn load(vb: VarBuilder, config: &Config, uses_local_attention: bool) -> Result<Self> {
        let attn = ModernBertAttention::load(vb.pp("attn"), config)?;
        let mlp = ModernBertMLP::load(vb.pp("mlp"), config)?;
        let attn_norm = layer_norm_no_bias(
            config.hidden_size,
            config.layer_norm_eps,
            vb.pp("attn_norm"),
        )
        .ok();
        let mlp_norm =
            layer_norm_no_bias(config.hidden_size, config.layer_norm_eps, vb.pp("mlp_norm"))?;
        Ok(Self {
            attn,
            mlp,
            attn_norm,
            mlp_norm,
            uses_local_attention,
        })
    }

    fn forward(&self, xs: &Tensor, attention_mask: Option<&Tensor>, rope: &Rope) -> Result<Tensor> {
        let residual = xs.clone();
        let mut xs = xs.clone();
        if let Some(norm) = &self.attn_norm {
            xs = xs.apply(norm)?;
        }

        let xs = self.attn.forward(&xs, attention_mask, rope)?;
        let xs = (xs + residual)?;
        let mlp_out = xs.apply(&self.mlp_norm)?.apply(&self.mlp)?;
        let xs = (xs + mlp_out)?;
        Ok(xs)
    }
}

// ModernBERT backbone
#[derive(Clone)]
pub struct ModernBert {
    word_embeddings: Embedding,
    norm: LayerNorm,
    layers: Vec<ModernBertLayer>,
    final_norm: LayerNorm,
    global_rotary_emb: Arc<RotaryEmbedding>,
    local_rotary_emb: Arc<RotaryEmbedding>,
}

impl ModernBert {
    pub fn load(vb: VarBuilder, config: &Config) -> Result<Self> {
        let word_embeddings = embedding(
            config.vocab_size,
            config.hidden_size,
            vb.pp("model.embeddings.tok_embeddings"),
        )?;
        let norm = layer_norm_no_bias(
            config.hidden_size,
            config.layer_norm_eps,
            vb.pp("model.embeddings.norm"),
        )?;
        let global_rotary_emb = Arc::new(RotaryEmbedding::new(
            vb.dtype(),
            config,
            config.global_rope_theta,
            vb.device(),
        )?);
        let local_rotary_emb = Arc::new(RotaryEmbedding::new(
            vb.dtype(),
            config,
            config.local_rope_theta,
            vb.device(),
        )?);

        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for layer_id in 0..config.num_hidden_layers {
            let layer_uses_local_attention = layer_id % config.global_attn_every_n_layers != 0;
            layers.push(ModernBertLayer::load(
                vb.pp(format!("model.layers.{layer_id}")),
                config,
                layer_uses_local_attention,
            )?);
        }

        let final_norm = layer_norm_no_bias(
            config.hidden_size,
            config.layer_norm_eps,
            vb.pp("model.final_norm"),
        )?;

        Ok(Self {
            word_embeddings,
            norm,
            layers,
            final_norm,
            global_rotary_emb,
            local_rotary_emb,
        })
    }

    /// Hidden states for `xs` (token ids, shape `(1, seq_len)`), with `position_ids`
    /// (`u32`, shape `(seq_len,)`) giving each token's RoPE position.
    pub fn forward(
        &self,
        xs: &Tensor,
        position_ids: &Tensor,
        masks: &AttentionMasks,
    ) -> Result<Tensor> {
        let global_rope = self.global_rotary_emb.at(position_ids)?;
        let local_rope = self.local_rotary_emb.at(position_ids)?;
        let mut xs = xs.apply(&self.word_embeddings)?.apply(&self.norm)?;
        for layer in self.layers.iter() {
            xs = if layer.uses_local_attention {
                layer.forward(&xs, Some(&masks.sliding), &local_rope)?
            } else {
                layer.forward(&xs, masks.global.as_ref(), &global_rope)?
            };
        }
        let xs = xs.apply(&self.final_norm)?;
        Ok(xs)
    }
}
