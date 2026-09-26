//! Attention masks and position ids for the two attention modes (port of
//! `build_independent_option_masks` and `build_option_invariant_position_ids`).
//!
//! A packed sequence is `[prefix][MASK opt0 text][MASK opt1 text]...[SEP]`. In the
//! independent-options mode each option sees only the prefix and itself, and every
//! option's positions restart right after the prefix, so an option's logit does not
//! depend on the other options or on their order.

use candle_core::{Device, Result, Tensor};

use super::modernbert::{AttentionMasks, Config};

/// How options attend to each other in the encoder.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum AttentionMode {
    /// Full bidirectional attention (Von 1.1 and earlier).
    #[default]
    Full,
    /// Each option attends only to the prefix and to itself (Von 1.2+).
    IndependentOptions,
}

/// Option index of every token, or `None` for the prefix. Option `k` spans from its
/// [MASK] to the next [MASK]; the last option ends before the trailing [SEP], which
/// stays prefix so the last slot is not special.
pub fn option_ids(mask_positions: &[usize], seq_len: usize) -> Vec<Option<usize>> {
    let mut ids = vec![None; seq_len];
    let last_content = seq_len.saturating_sub(1);
    for (k, &start) in mask_positions.iter().enumerate() {
        let end = mask_positions.get(k + 1).copied().unwrap_or(last_content);
        for id in ids.iter_mut().take(end).skip(start) {
            *id = Some(k);
        }
    }
    ids
}

/// Position ids with every option's span restarted at the prefix length.
pub fn invariant_position_ids(mask_positions: &[usize], seq_len: usize) -> Vec<u32> {
    let mut pos: Vec<u32> = (0..seq_len as u32).collect();
    let Some(&prefix_len) = mask_positions.first() else {
        return pos;
    };
    let last_content = seq_len.saturating_sub(1);
    for (k, &start) in mask_positions.iter().enumerate() {
        let end = mask_positions.get(k + 1).copied().unwrap_or(last_content);
        for (offset, p) in pos.iter_mut().take(end).skip(start).enumerate() {
            *p = (prefix_len + offset) as u32;
        }
    }
    pos
}

/// Row-major `(seq_len, seq_len)` allow-lists (row = query, column = key) for the
/// global and the sliding-window layers. The sliding window, when given, is measured
/// on `position_ids`, not on the raw index. The diagonal is always allowed.
pub fn independent_allowed(
    option_ids: &[Option<usize>],
    position_ids: &[u32],
    sliding_window: Option<usize>,
) -> (Vec<bool>, Vec<bool>) {
    let n = option_ids.len();
    let mut global = Vec::with_capacity(n * n);
    let mut sliding = Vec::with_capacity(n * n);
    for i in 0..n {
        for j in 0..n {
            let allowed = i == j
                || match (option_ids[i], option_ids[j]) {
                    (None, key) => key.is_none(),
                    (Some(_), None) => true,
                    (Some(a), Some(b)) => a == b,
                };
            let near = sliding_window
                .is_none_or(|w| position_ids[i].abs_diff(position_ids[j]) as usize <= w);
            global.push(allowed);
            sliding.push(i == j || (allowed && near));
        }
    }
    (global, sliding)
}

/// Position ids and encoder masks for one unpadded sequence.
pub fn build(
    mode: AttentionMode,
    config: &Config,
    mask_positions: &[usize],
    seq_len: usize,
    device: &Device,
) -> Result<(Tensor, AttentionMasks)> {
    match mode {
        AttentionMode::Full => Ok((
            Tensor::arange(0u32, seq_len as u32, device)?,
            AttentionMasks::full(config, seq_len, device)?,
        )),
        AttentionMode::IndependentOptions => {
            let positions = invariant_position_ids(mask_positions, seq_len);
            // transformers' ModernBERT `sliding_window` is `local_attention // 2`.
            let (global, sliding) = independent_allowed(
                &option_ids(mask_positions, seq_len),
                &positions,
                Some(config.local_attention / 2),
            );
            Ok((
                Tensor::new(positions, device)?,
                AttentionMasks {
                    global: Some(additive(&global, seq_len, device)?),
                    sliding: additive(&sliding, seq_len, device)?,
                },
            ))
        }
    }
}

fn additive(allowed: &[bool], seq_len: usize, device: &Device) -> Result<Tensor> {
    let values: Vec<f32> = allowed
        .iter()
        .map(|&a| if a { 0. } else { f32::NEG_INFINITY })
        .collect();
    Tensor::from_vec(values, (seq_len, seq_len), device)
}

#[cfg(test)]
mod tests {
    use super::*;

    // [CLS] a b [MASK] x [MASK] y z [SEP]
    const MASKS: [usize; 2] = [3, 5];
    const LEN: usize = 9;

    #[test]
    fn spans_stop_before_the_trailing_sep() {
        let ids = option_ids(&MASKS, LEN);
        let want = [
            None,
            None,
            None,
            Some(0),
            Some(0),
            Some(1),
            Some(1),
            Some(1),
            None,
        ];
        assert_eq!(ids, want);
    }

    #[test]
    fn options_restart_after_the_prefix() {
        assert_eq!(
            invariant_position_ids(&MASKS, LEN),
            [0, 1, 2, 3, 4, 3, 4, 5, 8]
        );
        assert_eq!(invariant_position_ids(&[], 4), [0, 1, 2, 3]);
    }

    #[test]
    fn options_never_see_each_other() {
        let ids = option_ids(&MASKS, LEN);
        let (global, _) = independent_allowed(&ids, &invariant_position_ids(&MASKS, LEN), None);
        let at = |i: usize, j: usize| global[i * LEN + j];
        assert!(at(0, 8) && at(8, 0), "prefix and [SEP] see each other");
        assert!(!at(0, 3), "prefix never sees an option");
        assert!(at(3, 0) && at(4, 3), "an option sees the prefix and itself");
        assert!(!at(3, 5) && !at(5, 4), "options never see each other");
    }
}
