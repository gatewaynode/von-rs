//! Confidence calibration (port of `_validate_calibration_map` and
//! `_effective_temperature`).
//!
//! When `marker_calibration.json` carries a fitted map, the softmax temperature
//! is a bounded linear function of the request's own features, so confidence
//! tracks difficulty. Temperature never changes the argmax.

use std::path::Path;

use serde_json::Value;

/// Share of the context-free polarity prior removed in zero-shot Noul when the
/// checkpoint ships no fitted `noul_zero_shot_prior`.
pub const NOUL_DEBIAS: f32 = 0.7;
/// Bounds outside `(0, TEMP_SANITY_MAX]` indicate a broken calibration file.
pub const TEMP_SANITY_MAX: f64 = 50.0;
const FEATURE_KEYS: [&str; 4] = ["bias", "entropy", "log_tokens", "n_options"];

#[derive(Debug, Clone, PartialEq)]
pub struct CalibrationMap {
    pub bias: f64,
    pub entropy: f64,
    pub log_tokens: f64,
    pub n_options: f64,
    pub lo: f64,
    pub hi: f64,
}

impl CalibrationMap {
    /// Validates a raw `calibration_map` value once, at load time. Anything
    /// unusable yields `None`, which falls back to the scalar temperature.
    pub fn from_json(raw: &Value) -> Option<Self> {
        let obj = raw.as_object()?;
        let mut values = std::collections::HashMap::new();
        for (key, value) in obj {
            values.insert(key.as_str(), py_float(value)?);
        }
        if !FEATURE_KEYS.iter().any(|k| values.contains_key(k)) {
            return None;
        }
        let get = |k: &str| values.get(k).copied();
        let map = CalibrationMap {
            bias: get("bias").unwrap_or(0.0),
            entropy: get("entropy").unwrap_or(0.0),
            log_tokens: get("log_tokens").unwrap_or(0.0),
            n_options: get("n_options").unwrap_or(0.0),
            lo: get("lo").unwrap_or(0.5),
            hi: get("hi").unwrap_or(12.0),
        };
        if map.lo > map.hi {
            return None;
        }
        if map.bounds_look_wrong() {
            tracing::warn!(
                "calibration map bounds [{}, {}] are outside the sane range (0, {TEMP_SANITY_MAX}]; confidences may be distorted.",
                map.lo,
                map.hi
            );
        }
        Some(map)
    }

    pub fn bounds_look_wrong(&self) -> bool {
        self.lo <= 0.0 || self.hi > TEMP_SANITY_MAX
    }
}

/// Fitted zero-shot Noul debiasing (`noul_zero_shot_prior` in the calibration
/// file; port of `_validate_noul_prior`). The correction subtracted from the
/// "true" logit is `a * bias + b` instead of the default `0.7 * bias`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NoulPrior {
    pub a: f64,
    pub b: f64,
}

impl NoulPrior {
    /// Both keys must be present and `float()`-coercible; anything else means
    /// no prior, which keeps the default correction.
    pub fn from_json(raw: &Value) -> Option<Self> {
        let obj = raw.as_object()?;
        Some(NoulPrior {
            a: py_float(obj.get("a")?)?,
            b: py_float(obj.get("b")?)?,
        })
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Calibration {
    /// Scalar temperature, used when there is no valid map.
    pub temperature: f64,
    pub map: Option<CalibrationMap>,
    pub noul_prior: Option<NoulPrior>,
}

impl Default for Calibration {
    fn default() -> Self {
        Calibration {
            temperature: 1.0,
            map: None,
            noul_prior: None,
        }
    }
}

impl Calibration {
    /// Loads `marker_calibration.json`. A missing or malformed file means
    /// uncalibrated (T = 1.0), exactly as in Python.
    pub fn from_file(path: Option<&Path>) -> Self {
        path.and_then(|p| std::fs::read_to_string(p).ok())
            .and_then(|text| serde_json::from_str::<Value>(&text).ok())
            .and_then(|v| Self::from_json(&v))
            .unwrap_or_default()
    }

    pub fn from_json(doc: &Value) -> Option<Self> {
        let obj = doc.as_object()?;
        let temperature = match obj.get("temperature") {
            None => 1.0,
            Some(v) => py_float(v)?,
        };
        let map = obj
            .get("calibration_map")
            .and_then(CalibrationMap::from_json);
        let noul_prior = obj
            .get("noul_zero_shot_prior")
            .and_then(NoulPrior::from_json);
        Some(Calibration {
            temperature,
            map,
            noul_prior,
        })
    }

    pub fn describe(&self) -> String {
        if self.map.is_some() {
            "input-conditioned calibration map active".into()
        } else if self.temperature != 1.0 {
            format!("temperature {}", self.temperature)
        } else {
            "uncalibrated, T=1.0".into()
        }
    }

    /// Amount subtracted from the "true" logit in zero-shot Noul, given the
    /// context-free bias `null_true - null_false`. Computed in f32, as torch
    /// does for a Python float times an f32 tensor.
    pub fn noul_correction(&self, bias: f32) -> f32 {
        match self.noul_prior {
            Some(p) => p.a as f32 * bias + p.b as f32,
            None => NOUL_DEBIAS * bias,
        }
    }

    /// Softmax temperature for one request. `state_tokens` is evaluated only
    /// when a map is present, because counting needs a tokenizer pass.
    pub fn effective_temperature(
        &self,
        logits: &[f32],
        n_options: usize,
        state_tokens: impl FnOnce() -> usize,
        override_temperature: Option<f64>,
    ) -> f64 {
        if let Some(t) = override_temperature {
            return t;
        }
        let Some(m) = &self.map else {
            return self.temperature;
        };
        // Entropy of the unscaled distribution, in f32 like the torch reference.
        let probs = softmax_f32(logits, 1.0);
        let n = probs.len().max(1);
        let entropy = if n > 1 {
            let h: f32 = probs.iter().map(|p| -(p * p.max(1e-12).ln())).sum();
            h as f64 / (n as f64).ln()
        } else {
            0.0
        };
        let tokens = state_tokens().max(1);
        let raw = m.bias
            + m.entropy * entropy
            + m.log_tokens * ((tokens as f64).log10() / 4.0)
            + m.n_options * (n_options as f64 / 8.0);
        // Python `min(hi, max(lo, raw))`, including its NaN behaviour.
        let floored = if raw > m.lo { raw } else { m.lo };
        if floored < m.hi { floored } else { m.hi }
    }
}

/// `softmax(logits / temperature)` in f32, matching the torch reference dtype.
pub fn softmax_f32(logits: &[f32], temperature: f64) -> Vec<f32> {
    let t = temperature.max(1e-4) as f32;
    let scaled: Vec<f32> = logits.iter().map(|x| x / t).collect();
    let max = scaled.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = scaled.iter().map(|x| (x - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    exps.iter().map(|e| e / sum).collect()
}

/// Python `float(value)` for JSON-decoded values: numbers, bools, and numeric
/// strings (surrounding whitespace, `inf`/`nan`, and digit-group underscores allowed).
fn py_float(v: &Value) -> Option<f64> {
    match v {
        Value::Number(n) => n.as_f64(),
        Value::Bool(b) => Some(if *b { 1.0 } else { 0.0 }),
        Value::String(s) => {
            let s = s.trim();
            let chars: Vec<char> = s.chars().collect();
            let underscores_ok = chars.iter().enumerate().all(|(i, c)| {
                *c != '_'
                    || (i > 0
                        && i + 1 < chars.len()
                        && chars[i - 1].is_ascii_digit()
                        && chars[i + 1].is_ascii_digit())
            });
            if !underscores_ok {
                return None;
            }
            s.replace('_', "").parse::<f64>().ok()
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn noul_prior_is_read_from_the_calibration_file() {
        let c = Calibration::from_json(&json!({
            "temperature": 2.0,
            "noul_zero_shot_prior": {"a": 0.5, "b": -1.0}
        }))
        .unwrap();
        assert_eq!(c.noul_prior, Some(NoulPrior { a: 0.5, b: -1.0 }));
        assert_eq!(c.noul_correction(2.0), 0.0);

        // A malformed prior keeps the default correction; the rest still loads.
        let c = Calibration::from_json(&json!({
            "temperature": 2.0,
            "noul_zero_shot_prior": {"a": 0.5}
        }))
        .unwrap();
        assert_eq!((c.temperature, c.noul_prior), (2.0, None));
        assert_eq!(c.noul_correction(2.0), NOUL_DEBIAS * 2.0);

        // A bad temperature discards the whole file, prior included (Python's `except`).
        assert!(
            Calibration::from_json(&json!({
                "temperature": "hot",
                "noul_zero_shot_prior": {"a": 1, "b": 0}
            }))
            .is_none()
        );
    }
}
