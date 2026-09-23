//! Checks the Python-compatibility layer against real Python output
//! (`tests/fixtures/python_oracle.json`, from `tools/export_pyfixtures.py`).
//! Needs no model weights.

use serde_json::Value;
use von::calibration::{Calibration, CalibrationMap, NoulPrior};
use von::pyfmt::{float_repr, py_strip, round, str_repr};
use von::state::format_state;
use von::types::{Answer, Question, SystemOneResponse};

fn oracle() -> Value {
    let text = std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/python_oracle.json"
    ))
    .expect("python_oracle.json");
    serde_json::from_str(&text).unwrap()
}

fn f64_bits(v: &Value) -> f64 {
    f64::from_bits(u64::from_str_radix(v.as_str().unwrap(), 16).unwrap())
}

fn cases<'a>(o: &'a Value, key: &str) -> &'a Vec<Value> {
    o[key]
        .as_array()
        .unwrap_or_else(|| panic!("fixture section {key}"))
}

#[test]
fn float_repr_matches_python() {
    let o = oracle();
    let failures: Vec<String> = cases(&o, "float_repr")
        .iter()
        .filter_map(|c| {
            let x = f64_bits(&c["x"]);
            let (got, want) = (float_repr(x), c["repr"].as_str().unwrap());
            (got != want).then(|| format!("{x:e}: got {got}, want {want}"))
        })
        .collect();
    assert!(failures.is_empty(), "{failures:#?}");
}

#[test]
fn round_matches_python() {
    let o = oracle();
    let failures: Vec<String> = cases(&o, "round")
        .iter()
        .filter_map(|c| {
            let (x, n) = (f64_bits(&c["x"]), c["n"].as_u64().unwrap() as usize);
            let (got, want) = (round(x, n), f64_bits(&c["out"]));
            (got.to_bits() != want.to_bits())
                .then(|| format!("round({x:?}, {n}): got {got:?}, want {want:?}"))
        })
        .collect();
    assert!(failures.is_empty(), "{failures:#?}");
}

#[test]
fn strip_and_repr_match_python() {
    let o = oracle();
    for c in cases(&o, "strip") {
        assert_eq!(
            py_strip(c["s"].as_str().unwrap()),
            c["out"].as_str().unwrap(),
            "strip {:?}",
            c["s"]
        );
    }
    for c in cases(&o, "str_repr") {
        assert_eq!(
            str_repr(c["s"].as_str().unwrap()),
            c["out"].as_str().unwrap(),
            "repr {:?}",
            c["s"]
        );
    }
}

#[test]
fn format_state_matches_python() {
    let o = oracle();
    for c in cases(&o, "format_state") {
        let input: Value = serde_json::from_str(c["json"].as_str().unwrap()).unwrap();
        assert_eq!(
            format_state(&input),
            c["out"].as_str().unwrap(),
            "state {}",
            c["json"]
        );
    }
}

#[test]
fn wire_types_round_trip_pydantic_dumps() {
    let o = oracle();
    for c in cases(&o, "models") {
        let name = c["name"].as_str().unwrap();
        let json = c["json"].as_str().unwrap();
        let want: Value = serde_json::from_str(json).unwrap();
        let got = match name {
            n if n.ends_with("_answer") => {
                serde_json::to_value(serde_json::from_str::<Answer>(json).unwrap())
            }
            "response" => {
                serde_json::to_value(serde_json::from_str::<SystemOneResponse>(json).unwrap())
            }
            _ => serde_json::to_value(serde_json::from_str::<Question>(json).unwrap()),
        }
        .unwrap();
        // Same keys, same order, same values.
        assert_eq!(
            serde_json::to_string(&got).unwrap(),
            serde_json::to_string(&want).unwrap(),
            "{name}"
        );
    }
}

#[test]
fn calibration_maps_validate_like_python() {
    let o = oracle();
    for c in cases(&o, "calibration_maps") {
        let got = CalibrationMap::from_json(&c["raw"]);
        match &c["valid"] {
            Value::Null => assert!(got.is_none(), "expected rejection of {}", c["raw"]),
            want => {
                let m = got.unwrap_or_else(|| panic!("expected acceptance of {}", c["raw"]));
                for (key, field) in [
                    ("lo", m.lo),
                    ("hi", m.hi),
                    ("bias", m.bias),
                    ("entropy", m.entropy),
                    ("log_tokens", m.log_tokens),
                    ("n_options", m.n_options),
                ] {
                    let expected = want.get(key).map(f64_bits).unwrap_or(0.0);
                    assert!(
                        field.to_bits() == expected.to_bits()
                            || (field.is_nan() && expected.is_nan()),
                        "{key} of {}: got {field}, want {expected}",
                        c["raw"]
                    );
                }
                assert_eq!(
                    m.bounds_look_wrong(),
                    c["warned"].as_bool().unwrap(),
                    "warning for {}",
                    c["raw"]
                );
            }
        }
    }
}

#[test]
fn effective_temperature_matches_python() {
    let o = oracle();
    let fitted = CalibrationMap {
        bias: 0.2056,
        entropy: -3.255,
        log_tokens: 20.2391,
        n_options: -5.2263,
        lo: 0.3,
        hi: 12.0,
    };
    for (i, c) in cases(&o, "temperatures").iter().enumerate() {
        let logits: Vec<f32> = c["logits"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| f64_bits(v) as f32)
            .collect();
        let calibration = Calibration {
            temperature: 2.2,
            map: c["calibrated"].as_bool().unwrap().then(|| fitted.clone()),
            noul_prior: None,
        };
        let tokens = c["state_tokens"].as_u64().unwrap() as usize;
        let got = calibration.effective_temperature(
            &logits,
            c["n_options"].as_u64().unwrap() as usize,
            || tokens,
            c["override"].as_f64(),
        );
        let want = f64_bits(&c["temperature"]);
        // Entropy is summed in f32 in a different order than torch's reduction.
        assert!(
            (got - want).abs() <= 1e-5 * want.abs().max(1.0),
            "case {i}: got {got}, want {want}"
        );
    }
}

#[test]
fn noul_priors_validate_like_python() {
    let o = oracle();
    for c in cases(&o, "noul_priors") {
        let got = NoulPrior::from_json(&c["raw"]);
        match &c["valid"] {
            Value::Null => assert!(got.is_none(), "expected rejection of {}", c["raw"]),
            want => {
                let p = got.unwrap_or_else(|| panic!("expected acceptance of {}", c["raw"]));
                for (field, key) in [(p.a, "a"), (p.b, "b")] {
                    let expected = f64_bits(&want[key]);
                    assert!(
                        field.to_bits() == expected.to_bits()
                            || (field.is_nan() && expected.is_nan()),
                        "{key} of {}: got {field}, want {expected}",
                        c["raw"]
                    );
                }
            }
        }
    }
}

#[test]
fn noul_corrections_match_torch_bit_for_bit() {
    let o = oracle();
    for c in cases(&o, "noul_corrections") {
        let null: Vec<f32> = c["null"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| f64_bits(v) as f32)
            .collect();
        let bias = null[0] - null[1];
        let fitted = Calibration {
            noul_prior: Some(NoulPrior {
                a: f64_bits(&c["a"]),
                b: f64_bits(&c["b"]),
            }),
            ..Calibration::default()
        };
        for (calibration, key) in [(fitted, "fitted"), (Calibration::default(), "default")] {
            let got = calibration.noul_correction(bias);
            let want = f64_bits(&c[key]) as f32;
            assert_eq!(
                got.to_bits(),
                want.to_bits(),
                "{key}: got {got}, want {want}"
            );
        }
    }
}

#[test]
fn presets_match_python_model_dump() {
    use indexmap::IndexMap;
    use von::presets::{email_preset, moderation_preset, security_preset, triage_preset};

    let o = oracle();
    let custom = IndexMap::from([
        ("ops".to_string(), "Operations".to_string()),
        ("hr".to_string(), "People team".to_string()),
    ]);
    for (name, preset) in [
        ("triage", triage_preset()),
        ("email", email_preset(None)),
        ("email_empty_custom", email_preset(Some(IndexMap::new()))),
        ("email_custom", email_preset(Some(custom))),
        ("moderation", moderation_preset()),
        ("security", security_preset()),
    ] {
        // Same keys, same order, same values.
        assert_eq!(
            serde_json::to_string(&preset).unwrap(),
            serde_json::to_string(&o["presets"][name]).unwrap(),
            "{name}"
        );
    }
}
