//! Golden parity against the Python reference runtime.
//!
//! Needs model weights, so the tests are `#[ignore]`d by default:
//!
//!     VON_WEIGHTS=checkpoints/von-1.2 cargo test --release --test parity -- --ignored --nocapture
//!
//! Two golden sets, both from `tools/export_golden.py` (torch CPU fp32):
//! `tests/fixtures/golden/v1_2.json` (Von 1.2, weights from `VON_WEIGHTS`, required) and
//! `tests/fixtures/golden/v1.json` (Von 1.1, default attention; weights from
//! `VON_WEIGHTS_V11`, skipped when that is unset or missing). `VON_DEVICE` picks the
//! device (default `auto`: Metal if visible, else CPU). Each stage is checked on its
//! own so a failure points at packing, tokenization, the encoder/scorer, or answer
//! assembly.

use std::path::Path;
use std::sync::OnceLock;

use indexmap::IndexMap;
use serde_json::Value;
use von::model::masks::AttentionMode;
use von::state::format_state;
use von::{Answer, LoadOptions, Question, Von};

/// Parity tolerances.
const LOGIT_TOL: f64 = 1e-3;
const PROB_TOL: f64 = 2e-3;
/// A score is a probability-weighted sum over up to 10 levels.
const SCORE_TOL: f64 = 1e-2;

/// A checkpoint and the golden set recorded from it.
struct Suite {
    von: Von,
    golden: Value,
}

impl Suite {
    fn load(dir: &str, golden: &str) -> Self {
        let von = Von::load(LoadOptions {
            checkpoint_dir: Some(dir.into()),
            ..Default::default()
        })
        .expect("load");
        let path = format!(
            "{}/tests/fixtures/golden/{golden}",
            env!("CARGO_MANIFEST_DIR")
        );
        let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{path}: {e}"));
        let golden: Value = serde_json::from_str(&text).unwrap();
        let independent =
            von.backend().calibration().attention_mode() == AttentionMode::IndependentOptions;
        assert_eq!(
            golden["independent_options"].as_bool().unwrap_or(false),
            independent,
            "{dir} is not the checkpoint {path} was recorded from"
        );
        eprintln!("parity {path}: device {}", von.device_description());
        Suite { von, golden }
    }

    fn cases(&self) -> impl Iterator<Item = &Value> {
        self.golden["cases"].as_array().unwrap().iter()
    }
}

/// Von 1.2 (independent options), from `VON_WEIGHTS`.
fn v1_2() -> &'static Suite {
    static SUITE: OnceLock<Suite> = OnceLock::new();
    SUITE.get_or_init(|| {
        let dir = std::env::var("VON_WEIGHTS")
            .expect("set VON_WEIGHTS to a converted checkpoint dir (see tools/convert_weights.py)");
        Suite::load(&dir, "v1_2.json")
    })
}

/// Von 1.1 (default attention), from `VON_WEIGHTS_V11` when it holds a checkpoint.
fn v1_1() -> Option<&'static Suite> {
    static SUITE: OnceLock<Option<Suite>> = OnceLock::new();
    SUITE
        .get_or_init(|| match std::env::var("VON_WEIGHTS_V11") {
            Ok(dir) if Path::new(&dir).is_dir() => Some(Suite::load(&dir, "v1.json")),
            _ => {
                eprintln!("skipping the Von 1.1 golden set: VON_WEIGHTS_V11 holds no checkpoint");
                None
            }
        })
        .as_ref()
}

fn questions(case: &Value) -> IndexMap<String, Question> {
    serde_json::from_value(case["questions"].clone())
        .unwrap_or_else(|e| panic!("{}: {e}", case["id"]))
}

fn passes(case: &Value) -> &Vec<Value> {
    case["passes"].as_array().unwrap()
}

fn check_packing_and_tokenization(suite: &Suite) {
    let von = &suite.von;
    let mut failures = Vec::new();
    for case in suite.cases() {
        let state_text = format_state(&case["state"]);
        let ours: Vec<String> = questions(case)
            .values()
            .flat_map(|q| von.backend().packed_inputs(&state_text, q).unwrap())
            .collect();
        let theirs: Vec<&str> = passes(case)
            .iter()
            .map(|p| p["packed_input"].as_str().unwrap())
            .collect();
        if ours != theirs {
            failures.push(format!(
                "{}: packed inputs differ\n  ours:   {ours:?}\n  theirs: {theirs:?}",
                case["id"]
            ));
            continue;
        }
        for (i, pass) in passes(case).iter().enumerate() {
            let want: Vec<u32> = pass["token_ids"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_u64().unwrap() as u32)
                .collect();
            if von.backend().model().encode(&ours[i]).unwrap() != want {
                failures.push(format!("{} pass {i}: token ids differ", case["id"]));
            }
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

fn check_logits(suite: &Suite) {
    let von = &suite.von;
    let (mut worst, mut worst_case, mut n) = (0.0f64, String::new(), 0);
    let mut failures = Vec::new();
    for case in suite.cases() {
        for (i, pass) in passes(case).iter().enumerate() {
            let want: Vec<f64> = pass["logits"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_f64().unwrap())
                .collect();
            let got = von
                .backend()
                .model()
                .option_logits(
                    pass["packed_input"].as_str().unwrap(),
                    want.len(),
                    von.backend().calibration().attention_mode(),
                )
                .unwrap();
            let delta = got
                .iter()
                .zip(&want)
                .map(|(g, w)| (*g as f64 - w).abs())
                .fold(0.0, f64::max);
            n += 1;
            if delta > worst {
                (worst, worst_case) = (delta, format!("{} pass {i}", case["id"]));
            }
            if delta > LOGIT_TOL {
                failures.push(format!(
                    "{} pass {i}: max logit delta {delta:.2e}",
                    case["id"]
                ));
            }
        }
    }
    eprintln!("logits: {n} passes, worst delta {worst:.3e} ({worst_case})");
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

/// Records a failure when `a` and `b` differ by more than `tol`; tracks the worst probability delta.
fn check(errs: &mut Vec<String>, worst_prob: &mut f64, label: String, a: f64, b: f64, tol: f64) {
    if label.contains(".p") {
        *worst_prob = worst_prob.max((a - b).abs());
    }
    if (a - b).abs() > tol {
        errs.push(format!("{label}: {a} vs {b}"));
    }
}

fn check_responses(suite: &Suite) {
    let von = &suite.von;
    let mut errs = Vec::new();
    let mut worst = 0.0f64;
    for case in suite.cases() {
        let id = case["id"].as_str().unwrap();
        let got = von.evaluate(&case["state"], &questions(case)).unwrap();
        let want: von::SystemOneResponse =
            serde_json::from_value(case["response"].clone()).unwrap();

        if got.model != want.model {
            errs.push(format!("{id}: model {} != {}", got.model, want.model));
        }
        if got.usage != want.usage {
            errs.push(format!("{id}: usage {:?} != {:?}", got.usage, want.usage));
        }
        if !got.answers.keys().eq(want.answers.keys()) {
            errs.push(format!("{id}: answer ids or order differ"));
            continue;
        }
        for ((qid, g), w) in got.answers.iter().zip(want.answers.values()) {
            let at = format!("{id}.{qid}");
            match (g, w) {
                (Answer::Choice(g), Answer::Choice(w)) => {
                    if g.choice != w.choice {
                        errs.push(format!("{at}.choice: {} vs {}", g.choice, w.choice));
                    }
                    if !g.probabilities.keys().eq(w.probabilities.keys()) {
                        errs.push(format!("{at}.probabilities keys differ"));
                    }
                    for (k, p) in &g.probabilities {
                        check(
                            &mut errs,
                            &mut worst,
                            format!("{at}.p[{k}]"),
                            *p,
                            w.probabilities[k],
                            PROB_TOL,
                        );
                    }
                    check(
                        &mut errs,
                        &mut worst,
                        format!("{at}.confidence"),
                        g.confidence,
                        w.confidence,
                        PROB_TOL,
                    );
                }
                (Answer::Noul(g), Answer::Noul(w)) => {
                    check(
                        &mut errs,
                        &mut worst,
                        format!("{at}.p(noul)"),
                        g.noul,
                        w.noul,
                        PROB_TOL,
                    );
                }
                (Answer::Score(g), Answer::Score(w)) => {
                    if g.legend != w.legend {
                        errs.push(format!("{at}.legend: {:?} vs {:?}", g.legend, w.legend));
                    }
                    for (k, p) in &g.probabilities {
                        check(
                            &mut errs,
                            &mut worst,
                            format!("{at}.p[{k}]"),
                            *p,
                            w.probabilities[k],
                            PROB_TOL,
                        );
                    }
                    check(
                        &mut errs,
                        &mut worst,
                        format!("{at}.score"),
                        g.score,
                        w.score,
                        SCORE_TOL,
                    );
                    check(
                        &mut errs,
                        &mut worst,
                        format!("{at}.confidence"),
                        g.confidence,
                        w.confidence,
                        PROB_TOL,
                    );
                }
                _ => errs.push(format!("{at}: answer type differs")),
            }
        }
    }
    eprintln!(
        "responses: {} cases, worst probability delta {worst:.2e}",
        suite.cases().count()
    );
    assert!(errs.is_empty(), "{}", errs.join("\n"));
}

#[test]
#[ignore = "needs model weights (VON_WEIGHTS)"]
fn packing_and_tokenization() {
    check_packing_and_tokenization(v1_2());
}

#[test]
#[ignore = "needs model weights (VON_WEIGHTS_V11)"]
fn packing_and_tokenization_v1_1() {
    if let Some(suite) = v1_1() {
        check_packing_and_tokenization(suite);
    }
}

#[test]
#[ignore = "needs model weights (VON_WEIGHTS)"]
fn logits() {
    check_logits(v1_2());
}

#[test]
#[ignore = "needs model weights (VON_WEIGHTS_V11)"]
fn logits_v1_1() {
    if let Some(suite) = v1_1() {
        check_logits(suite);
    }
}

#[test]
#[ignore = "needs model weights (VON_WEIGHTS)"]
fn responses() {
    check_responses(v1_2());
}

#[test]
#[ignore = "needs model weights (VON_WEIGHTS_V11)"]
fn responses_v1_1() {
    if let Some(suite) = v1_1() {
        check_responses(suite);
    }
}
