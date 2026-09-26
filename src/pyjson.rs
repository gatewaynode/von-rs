//! JSON text byte-identical to Python's `json.dumps`, for the outputs where von-rs
//! promises the same bytes: the HTTP server (FastAPI's compact form), the CLI
//! (`indent=2`, ASCII-only) and structured question instructions (the defaults).
//!
//! serde_json differs from Python in two ways this fixes: floats are written with
//! Python `repr` (`1e-05`, `1e+16`, not `1e-5`, `1e16`), and `ensure_ascii`
//! escapes every non-ASCII character and DEL as `\uXXXX` (surrogate pairs above U+FFFF).
//! Control characters already match: both use the short escapes and lowercase `\u00XX`.

use std::io::{self, Write};

use serde::Serialize;
use serde_json::Value;
use serde_json::ser::{Formatter, PrettyFormatter, Serializer};

use crate::pyfmt::float_repr;

/// `json.dumps(v, ensure_ascii=False, separators=(",", ":"))`, as FastAPI's `JSONResponse` renders.
pub fn compact<T: Serialize + ?Sized>(value: &T) -> String {
    render(value, PyFormatter::compact(false))
}

/// `json.dumps(v, indent=2)`, as the Python CLI prints (ASCII-only).
pub fn pretty<T: Serialize + ?Sized>(value: &T) -> String {
    render(value, PyFormatter::pretty(true))
}

/// `json.dumps(v, sort_keys=sort_keys)`: `", "` and `": "` separators, ASCII-only.
/// `sort_keys` sorts every object at any depth by code point, as Python does.
pub fn dumps(value: &Value, sort_keys: bool) -> String {
    let formatter = PyFormatter {
        layout: Layout::Spaced,
        ensure_ascii: true,
    };
    if sort_keys {
        render(&sorted(value), formatter)
    } else {
        render(value, formatter)
    }
}

/// A copy with every object's keys in code point order (UTF-8 byte order is the same).
fn sorted(value: &Value) -> Value {
    match value {
        Value::Object(map) => {
            let mut entries: Vec<_> = map.iter().collect();
            entries.sort_by(|a, b| a.0.cmp(b.0));
            Value::Object(
                entries
                    .into_iter()
                    .map(|(k, v)| (k.clone(), sorted(v)))
                    .collect(),
            )
        }
        Value::Array(items) => Value::Array(items.iter().map(sorted).collect()),
        other => other.clone(),
    }
}

fn render<T: Serialize + ?Sized>(value: &T, formatter: PyFormatter) -> String {
    let mut out = Vec::new();
    let mut ser = Serializer::with_formatter(&mut out, formatter);
    value
        .serialize(&mut ser)
        .expect("serializing to memory cannot fail for JSON-compatible values");
    String::from_utf8(out).expect("the formatter writes UTF-8")
}

struct PyFormatter {
    layout: Layout,
    ensure_ascii: bool,
}

enum Layout {
    /// `separators=(",", ":")`.
    Compact,
    /// The `json.dumps` default, `separators=(", ", ": ")`.
    Spaced,
    Pretty(PrettyFormatter<'static>),
}

impl PyFormatter {
    fn compact(ensure_ascii: bool) -> Self {
        Self {
            layout: Layout::Compact,
            ensure_ascii,
        }
    }

    fn pretty(ensure_ascii: bool) -> Self {
        Self {
            layout: Layout::Pretty(PrettyFormatter::with_indent(b"  ")),
            ensure_ascii,
        }
    }
}

/// Delegates layout to `PrettyFormatter` when indenting; the compact defaults
/// already match `separators=(",", ":")`, and `Spaced` differs only in separators.
macro_rules! layout {
    ($name:ident $(, $arg:ident : $ty:ty)*) => {
        fn $name<W: ?Sized + Write>(&mut self, writer: &mut W $(, $arg: $ty)*) -> io::Result<()> {
            match &mut self.layout {
                Layout::Pretty(p) => p.$name(writer $(, $arg)*),
                _ => serde_json::ser::CompactFormatter.$name(writer $(, $arg)*),
            }
        }
    };
}

impl PyFormatter {
    fn separator<W: ?Sized + Write>(&self, writer: &mut W, first: bool) -> io::Result<()> {
        match (first, &self.layout) {
            (true, _) => Ok(()),
            (false, Layout::Spaced) => writer.write_all(b", "),
            (false, _) => writer.write_all(b","),
        }
    }
}

impl Formatter for PyFormatter {
    layout!(begin_array);
    layout!(end_array);
    layout!(end_array_value);
    layout!(begin_object);
    layout!(end_object);
    layout!(end_object_key);
    layout!(end_object_value);

    fn begin_array_value<W: ?Sized + Write>(
        &mut self,
        writer: &mut W,
        first: bool,
    ) -> io::Result<()> {
        match &mut self.layout {
            Layout::Pretty(p) => p.begin_array_value(writer, first),
            _ => self.separator(writer, first),
        }
    }

    fn begin_object_key<W: ?Sized + Write>(
        &mut self,
        writer: &mut W,
        first: bool,
    ) -> io::Result<()> {
        match &mut self.layout {
            Layout::Pretty(p) => p.begin_object_key(writer, first),
            _ => self.separator(writer, first),
        }
    }

    fn begin_object_value<W: ?Sized + Write>(&mut self, writer: &mut W) -> io::Result<()> {
        match &mut self.layout {
            Layout::Pretty(p) => p.begin_object_value(writer),
            Layout::Spaced => writer.write_all(b": "),
            Layout::Compact => writer.write_all(b":"),
        }
    }

    fn write_f64<W: ?Sized + Write>(&mut self, writer: &mut W, value: f64) -> io::Result<()> {
        writer.write_all(float_repr(value).as_bytes())
    }

    fn write_f32<W: ?Sized + Write>(&mut self, writer: &mut W, value: f32) -> io::Result<()> {
        self.write_f64(writer, f64::from(value))
    }

    fn write_string_fragment<W: ?Sized + Write>(
        &mut self,
        writer: &mut W,
        fragment: &str,
    ) -> io::Result<()> {
        // `ensure_ascii` escapes everything outside ' '..='~', so DEL too.
        let printable = |c: char| c.is_ascii() && c as u32 != 0x7f;
        if !self.ensure_ascii || fragment.chars().all(printable) {
            return writer.write_all(fragment.as_bytes());
        }
        for c in fragment.chars() {
            if printable(c) {
                writer.write_all(&[c as u8])?;
            } else {
                let mut units = [0u16; 2];
                for unit in c.encode_utf16(&mut units) {
                    write!(writer, "\\u{unit:04x}")?;
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn floats_use_python_repr() {
        assert_eq!(
            compact(&json!([1e-5, 1e16, 0.1, 3.0, -0.0, 1.5e300, 12])),
            "[1e-05,1e+16,0.1,3.0,-0.0,1.5e+300,12]"
        );
    }

    #[test]
    fn ensure_ascii_only_in_pretty() {
        let v = json!({"k": "café ☃ 𝄞 \"q\" \n\u{1}"});
        assert_eq!(compact(&v), "{\"k\":\"café ☃ 𝄞 \\\"q\\\" \\n\\u0001\"}");
        assert_eq!(
            pretty(&v),
            "{\n  \"k\": \"caf\\u00e9 \\u2603 \\ud834\\udd1e \\\"q\\\" \\n\\u0001\"\n}"
        );
    }

    #[test]
    fn dumps_defaults_and_sort_keys() {
        let v = json!({"b": [1, {"z": 0.5, "a": "é"}], "a": {}, "c": []});
        assert_eq!(
            dumps(&v, false),
            "{\"b\": [1, {\"z\": 0.5, \"a\": \"\\u00e9\"}], \"a\": {}, \"c\": []}"
        );
        assert_eq!(
            dumps(&v, true),
            "{\"a\": {}, \"b\": [1, {\"a\": \"\\u00e9\", \"z\": 0.5}], \"c\": []}"
        );
    }

    #[test]
    fn pretty_layout_matches_indent_2() {
        assert_eq!(
            pretty(&json!({"a": [], "b": {}, "c": [1, {"d": null}]})),
            "{\n  \"a\": [],\n  \"b\": {},\n  \"c\": [\n    1,\n    {\n      \"d\": null\n    }\n  ]\n}"
        );
    }
}
