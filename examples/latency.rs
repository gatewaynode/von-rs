//! End-to-end `Von::evaluate` latency on a few golden requests. This only
//! measures; tuning comes later. Comparable with `tools/bench_python.py`,
//! which times the same requests through the Python backend.
//!
//!     VON_DEVICE=metal cargo run --release --example latency -- [checkpoint_dir]

use std::time::Instant;

use indexmap::IndexMap;
use serde_json::Value;
use von::{LoadOptions, Question, Von};

const CASES: [(&str, usize); 5] = [
    ("route-account-access", 100),
    ("many-options", 100),
    ("noul-zero-shot", 100),
    ("score-detailed", 100),
    ("long-600-tokens", 30),
];
const WARMUP: usize = 10;

fn main() -> anyhow::Result<()> {
    let dir = std::env::args()
        .nth(1)
        .or_else(|| std::env::var("VON_WEIGHTS").ok())
        .unwrap_or_else(|| "checkpoints/von-1.2".into());
    let started = Instant::now();
    let von = Von::load(LoadOptions {
        checkpoint_dir: Some(dir.into()),
        ..Default::default()
    })?;
    println!(
        "# {} · load {:.2}s",
        von.device_description(),
        started.elapsed().as_secs_f64()
    );

    let golden: Value = serde_json::from_str(&std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/golden/v1_2.json"
    ))?)?;
    println!("| case | p50 ms | p95 ms |\n|---|---|---|");
    for (id, iters) in CASES {
        let case = golden["cases"]
            .as_array()
            .and_then(|cs| cs.iter().find(|c| c["id"] == id))
            .ok_or_else(|| anyhow::anyhow!("golden case {id} not found"))?;
        let questions: IndexMap<String, Question> =
            serde_json::from_value(case["questions"].clone())?;
        for _ in 0..WARMUP {
            von.evaluate(&case["state"], &questions)?;
        }
        let mut ms: Vec<f64> = (0..iters)
            .map(|_| {
                let t = Instant::now();
                von.evaluate(&case["state"], &questions)
                    .map(|_| t.elapsed().as_secs_f64() * 1e3)
            })
            .collect::<Result<_, _>>()?;
        ms.sort_by(f64::total_cmp);
        println!(
            "| {id} | {:.2} | {:.2} |",
            ms[iters / 2],
            ms[(iters - 1) * 95 / 100]
        );
    }
    Ok(())
}
