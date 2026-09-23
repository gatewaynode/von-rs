//! Rendering of the request `state` into model input text.
//!
//! Port of `_format_state` in `option_marker_backend.py`: strings pass through,
//! objects become `key: value` lines, and everything else — including nested
//! values — is rendered the way Python's `str()` renders the object that
//! `json.loads` produced (`True`, `None`, `1.0`, `['a', 1]`, `{'k': 'v'}`).

use serde_json::{Number, Value};

use crate::pyfmt::{float_repr, str_repr};

pub fn format_state(state: &Value) -> String {
    match state {
        Value::String(s) => s.clone(),
        Value::Object(map) => map
            .iter()
            .map(|(k, v)| format!("{k}: {}", py_str(v)))
            .collect::<Vec<_>>()
            .join("\n"),
        other => py_str(other),
    }
}

/// Python `str(obj)` of a JSON-decoded value.
pub fn py_str(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        other => py_repr(other),
    }
}

/// Python `repr(obj)` of a JSON-decoded value.
///
/// Known gap: integers beyond the u64/i64 range are decoded as floats by
/// serde_json, where Python keeps an exact int. Listed in README, "Differences from the Python runtime".
pub fn py_repr(v: &Value) -> String {
    match v {
        Value::Null => "None".into(),
        Value::Bool(true) => "True".into(),
        Value::Bool(false) => "False".into(),
        Value::Number(n) => number_repr(n),
        Value::String(s) => str_repr(s),
        Value::Array(items) => format!(
            "[{}]",
            items.iter().map(py_repr).collect::<Vec<_>>().join(", ")
        ),
        Value::Object(map) => format!(
            "{{{}}}",
            map.iter()
                .map(|(k, v)| format!("{}: {}", str_repr(k), py_repr(v)))
                .collect::<Vec<_>>()
                .join(", ")
        ),
    }
}

fn number_repr(n: &Number) -> String {
    if let Some(i) = n.as_i64() {
        i.to_string()
    } else if let Some(u) = n.as_u64() {
        u.to_string()
    } else {
        float_repr(n.as_f64().unwrap_or(f64::NAN))
    }
}
