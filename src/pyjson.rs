//! JSON text byte-identical to Python's `json.dumps`, for the outputs where von-rs
//! promises the same bytes: the HTTP server (FastAPI's compact form) and the CLI
//! (`indent=2`, ASCII-only).
//!
//! serde_json differs from Python in two ways this fixes: floats are written with
//! Python `repr` (`1e-05`, `1e+16`, not `1e-5`, `1e16`), and `ensure_ascii`
//! escapes every non-ASCII character as `\uXXXX` (surrogate pairs above U+FFFF).
//! Control characters already match: both use the short escapes and lowercase `\u00XX`.

use std::io::{self, Write};

use serde::Serialize;
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

fn render<T: Serialize + ?Sized>(value: &T, formatter: PyFormatter) -> String {
    let mut out = Vec::new();
    let mut ser = Serializer::with_formatter(&mut out, formatter);
    value
        .serialize(&mut ser)
        .expect("serializing to memory cannot fail for JSON-compatible values");
    String::from_utf8(out).expect("the formatter writes UTF-8")
}

struct PyFormatter {
    pretty: Option<PrettyFormatter<'static>>,
    ensure_ascii: bool,
}

impl PyFormatter {
    fn compact(ensure_ascii: bool) -> Self {
        Self {
            pretty: None,
            ensure_ascii,
        }
    }

    fn pretty(ensure_ascii: bool) -> Self {
        Self {
            pretty: Some(PrettyFormatter::with_indent(b"  ")),
            ensure_ascii,
        }
    }
}

/// Delegates layout to `PrettyFormatter` when indenting; the compact defaults
/// already match `separators=(",", ":")`.
macro_rules! layout {
    ($name:ident $(, $arg:ident : $ty:ty)*) => {
        fn $name<W: ?Sized + Write>(&mut self, writer: &mut W $(, $arg: $ty)*) -> io::Result<()> {
            match &mut self.pretty {
                Some(p) => p.$name(writer $(, $arg)*),
                None => serde_json::ser::CompactFormatter.$name(writer $(, $arg)*),
            }
        }
    };
}

impl Formatter for PyFormatter {
    layout!(begin_array);
    layout!(end_array);
    layout!(begin_array_value, first: bool);
    layout!(end_array_value);
    layout!(begin_object);
    layout!(end_object);
    layout!(begin_object_key, first: bool);
    layout!(end_object_key);
    layout!(begin_object_value);
    layout!(end_object_value);

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
        if !self.ensure_ascii || fragment.is_ascii() {
            return writer.write_all(fragment.as_bytes());
        }
        for c in fragment.chars() {
            if c.is_ascii() {
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
    fn pretty_layout_matches_indent_2() {
        assert_eq!(
            pretty(&json!({"a": [], "b": {}, "c": [1, {"d": null}]})),
            "{\n  \"a\": [],\n  \"b\": {},\n  \"c\": [\n    1,\n    {\n      \"d\": null\n    }\n  ]\n}"
        );
    }
}
