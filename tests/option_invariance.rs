//! Independent-options mode is order invariant: permuting the options permutes the
//! logits and changes nothing else (port of the Python `tests/test_option_invariance.py`).
//! The guarantee is structural, so any checkpoint works; the mode is forced here.
//!
//!     VON_WEIGHTS=~/.local/models/von-1.2 cargo test --release --test option_invariance -- --ignored

use std::sync::OnceLock;

use von::model::masks::AttentionMode;
use von::{LoadOptions, Von};

fn von() -> &'static Von {
    static VON: OnceLock<Von> = OnceLock::new();
    VON.get_or_init(|| {
        let dir = std::env::var("VON_WEIGHTS")
            .expect("set VON_WEIGHTS to a converted checkpoint dir (see tools/convert_weights.py)");
        Von::load(LoadOptions {
            checkpoint_dir: Some(dir.into()),
            ..Default::default()
        })
        .unwrap()
    })
}

fn score(state: &str, question: &str, options: &[&str], mode: AttentionMode) -> Vec<f32> {
    let model = von().backend().model();
    let options: Vec<String> = options.iter().map(|s| s.to_string()).collect();
    let packed = model.pack(state, question, &options);
    model.option_logits(&packed, options.len(), mode).unwrap()
}

const INVOICE: &str = "The invoice was submitted three days after the policy deadline of 30 days.";
const INVOICE_Q: &str = "Is this invoice compliant?";
const INVOICE_OPTS: [&str; 2] = ["Yes, it is compliant.", "No, it is not compliant."];

fn max_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max)
}

#[test]
#[ignore = "needs model weights (VON_WEIGHTS)"]
fn two_option_reversal_is_invariant() {
    let fwd = score(
        INVOICE,
        INVOICE_Q,
        &INVOICE_OPTS,
        AttentionMode::IndependentOptions,
    );
    let rev_opts: Vec<&str> = INVOICE_OPTS.iter().rev().copied().collect();
    let mut rev = score(
        INVOICE,
        INVOICE_Q,
        &rev_opts,
        AttentionMode::IndependentOptions,
    );
    rev.reverse();
    assert!(max_diff(&fwd, &rev) <= 1e-4, "{fwd:?} vs reversed {rev:?}");
}

#[test]
#[ignore = "needs model weights (VON_WEIGHTS)"]
fn three_option_all_permutations_are_invariant() {
    let state = "The shipment weighed 42kg against a 40kg limit for the express tier.";
    let question = "What is the verdict?";
    let options = [
        "Compliant, within policy.",
        "Non-compliant, exceeds threshold.",
        "Unclear, needs review.",
    ];
    let perms = [
        [0, 1, 2],
        [0, 2, 1],
        [1, 0, 2],
        [1, 2, 0],
        [2, 0, 1],
        [2, 1, 0],
    ];
    let restored: Vec<Vec<f32>> = perms
        .iter()
        .map(|perm| {
            let opts: Vec<&str> = perm.iter().map(|&i| options[i]).collect();
            let logits = score(state, question, &opts, AttentionMode::IndependentOptions);
            let mut out = vec![0.0; 3];
            for (slot, &orig) in perm.iter().enumerate() {
                out[orig] = logits[slot];
            }
            out
        })
        .collect();
    for r in &restored[1..] {
        assert!(
            max_diff(&restored[0], r) < 1e-3,
            "{:?} vs {r:?}",
            restored[0]
        );
    }
}

/// Checks the test itself: full attention is not order invariant, so the
/// invariance above comes from the independent-options masks.
#[test]
#[ignore = "needs model weights (VON_WEIGHTS)"]
fn full_attention_is_not_invariant() {
    let fwd = score(INVOICE, INVOICE_Q, &INVOICE_OPTS, AttentionMode::Full);
    let rev_opts: Vec<&str> = INVOICE_OPTS.iter().rev().copied().collect();
    let mut rev = score(INVOICE, INVOICE_Q, &rev_opts, AttentionMode::Full);
    rev.reverse();
    assert!(max_diff(&fwd, &rev) > 1e-4, "{fwd:?} vs reversed {rev:?}");
}
