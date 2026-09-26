//! Criterion latency benchmarks for whole `Von::evaluate` calls: a 4-option
//! choice, a zero-shot Noul (two forward passes) and a 10-level score, each
//! with the main pass padded to 64, 512 and 4096 tokens. Measurement only.
//!
//!     VON_DEVICE=metal cargo bench --bench latency
//!     VON_DEVICE=cpu cargo bench --bench latency -- /512     # one length
//!
//! Weights come from `VON_WEIGHTS` (default `checkpoints/von-1.1`); without
//! them the benchmark prints a note and exits successfully.

use std::path::PathBuf;
use std::time::{Duration, Instant};

use criterion::{BenchmarkId, Criterion, SamplingMode};
use indexmap::IndexMap;
use serde_json::{Value, json};
use von::{LoadOptions, Question, Von};

const TOKENS: [usize; 3] = [64, 512, 4096];
const FILLER: [&str; 4] = [
    "The customer wrote in again about the order that arrived late.",
    "Support replied with the tracking number and an apology.",
    "Logs show the payment service retried twice before succeeding.",
    "The account has been active for three years without incidents.",
];

fn questions(kind: &str) -> IndexMap<String, Question> {
    let q = match kind {
        "choice" => json!({"type": "choice", "instructions": "What does the customer need?",
            "criteria": {"refund": "A refund for the order", "tracking": "Where the package is",
                         "cancel": "Cancel the account", "other": "None of these"}}),
        "noul-zero-shot" => json!({"type": "noul", "instructions": "Is the customer upset?"}),
        "score" => json!({"type": "score", "instructions": "How urgent is this?",
            "criteria": ["None", "Trace", "Minimal", "Low", "Mild",
                         "Moderate", "Notable", "High", "Severe", "Critical"]}),
        _ => unreachable!(),
    };
    IndexMap::from([(
        "q".to_string(),
        serde_json::from_value(q).expect("valid question"),
    )])
}

fn state_of(sentences: usize) -> Value {
    let text: Vec<&str> = (0..sentences).map(|i| FILLER[i % FILLER.len()]).collect();
    Value::String(text.join(" "))
}

/// Model tokens (with special tokens) of the first forward pass.
fn tokens(von: &Von, state: &Value, question: &Question) -> usize {
    let backend = von.backend();
    let packed = backend
        .packed_inputs(state.as_str().unwrap(), question)
        .expect("packs");
    backend.model().encode(&packed[0]).expect("encodes").len()
}

/// The longest filler state whose first pass fits in `target` tokens.
fn state_for(von: &Von, question: &Question, target: usize) -> (Value, usize) {
    let (mut lo, mut hi) = (0, target);
    while lo < hi {
        let mid = (lo + hi).div_ceil(2);
        if tokens(von, &state_of(mid), question) <= target {
            lo = mid;
        } else {
            hi = mid - 1;
        }
    }
    let state = state_of(lo);
    let n = tokens(von, &state, question);
    assert!(
        n <= target,
        "question alone needs {n} tokens, more than {target}"
    );
    (state, n)
}

fn main() {
    let dir = PathBuf::from(std::env::var("VON_WEIGHTS").unwrap_or("checkpoints/von-1.1".into()));
    if !dir.exists() {
        eprintln!("latency bench: no weights at {}, skipping", dir.display());
        return;
    }
    let von = Von::load(LoadOptions {
        checkpoint_dir: Some(dir),
        ..Default::default()
    })
    .expect("model loads");
    eprintln!("# {}", von.device_description());

    // Criterion's positional filter, so skipped benchmarks don't pay for the
    // sizing call below. Plain substrings only (criterion also takes regexes).
    let filter = std::env::args().skip(1).find(|a| !a.starts_with('-'));
    let mut c = Criterion::default().configure_from_args();
    for kind in ["choice", "noul-zero-shot", "score"] {
        let qs = questions(kind);
        let mut group = c.benchmark_group(kind);
        for target in TOKENS {
            let id = format!("{kind}/{target}");
            if filter.as_ref().is_some_and(|f| !id.contains(f.as_str())) {
                continue;
            }
            let (state, n) = state_for(&von, &qs["q"], target);
            // Size each run from one timed call, so long inputs get enough time
            // for their samples instead of a criterion warning.
            let t = Instant::now();
            von.evaluate(&state, &qs).expect("evaluates");
            let once = t.elapsed();
            let samples = if target >= 4096 {
                10
            } else if target >= 512 {
                20
            } else {
                100
            };
            // Flat sampling for long inputs: linear sampling would need ~5x the calls.
            let mode = if target >= 512 {
                SamplingMode::Flat
            } else {
                SamplingMode::Linear
            };
            group
                .sampling_mode(mode)
                .sample_size(samples)
                .warm_up_time(Duration::from_secs(1).max(once * 2))
                .measurement_time(Duration::from_secs(5).max(once * samples as u32 * 3 / 2));
            eprintln!("{id}: {n} tokens");
            group.bench_with_input(BenchmarkId::from_parameter(target), &state, |b, s| {
                b.iter(|| von.evaluate(s, &qs).expect("evaluates"))
            });
        }
        group.finish();
    }
    c.final_summary();
}
