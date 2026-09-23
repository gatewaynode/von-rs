//! Python-compatible text and number formatting.
//!
//! The model's input text and the published numbers are produced by Python
//! builtins in the reference runtime (`repr(float)`, `round()`, `str.strip()`,
//! `repr(str)`), so von-rs reproduces them exactly. Checked against real Python
//! output in `tests/python_oracle.rs`.

/// `repr(float)`: the shortest round-tripping digits, in fixed notation for
/// decimal exponents in `-4..16` and scientific (`1e-05`, `1.5e+16`) otherwise.
pub fn float_repr(x: f64) -> String {
    if x.is_nan() {
        return "nan".into();
    }
    if x.is_infinite() {
        return if x > 0.0 { "inf" } else { "-inf" }.into();
    }
    if x == 0.0 {
        return if x.is_sign_negative() { "-0.0" } else { "0.0" }.into();
    }
    // Rust's `{:e}` is also shortest-round-trip, e.g. "-1.2345e-5" or "1e16".
    let sci = format!("{x:e}");
    let (mantissa, exp) = sci
        .split_once('e')
        .expect("LowerExp always has an exponent");
    let exp: i32 = exp.parse().expect("LowerExp exponent is an integer");
    let (negative, mantissa) = match mantissa.strip_prefix('-') {
        Some(m) => (true, m),
        None => (false, mantissa),
    };
    let digits: String = mantissa.chars().filter(|c| *c != '.').collect();

    let mut out = String::with_capacity(digits.len() + 8);
    if negative {
        out.push('-');
    }
    if (-4..16).contains(&exp) {
        if exp >= 0 {
            let int_len = exp as usize + 1;
            if digits.len() <= int_len {
                out.push_str(&digits);
                out.push_str(&"0".repeat(int_len - digits.len()));
                out.push_str(".0");
            } else {
                out.push_str(&digits[..int_len]);
                out.push('.');
                out.push_str(&digits[int_len..]);
            }
        } else {
            out.push_str("0.");
            out.push_str(&"0".repeat((-exp - 1) as usize));
            out.push_str(&digits);
        }
    } else {
        out.push_str(&digits[..1]);
        if digits.len() > 1 {
            out.push('.');
            out.push_str(&digits[1..]);
        }
        out.push('e');
        out.push(if exp < 0 { '-' } else { '+' });
        out.push_str(&format!("{:02}", exp.unsigned_abs()));
    }
    out
}

/// `round(x, n)` for floats: correctly rounded to `n` decimals (ties to even on
/// the exact binary value), returned as the nearest double.
pub fn round(x: f64, n: usize) -> f64 {
    if !x.is_finite() {
        return x;
    }
    format!("{x:.n$}").parse().expect("formatted float parses")
}

/// Python `str.isspace()` for a single character: Unicode `White_Space` plus the
/// ASCII information separators U+001C..U+001F.
pub fn is_py_space(c: char) -> bool {
    c.is_whitespace() || matches!(c as u32, 0x1c..=0x1f)
}

/// `str.strip()` with no arguments.
pub fn py_strip(s: &str) -> &str {
    s.trim_matches(is_py_space)
}

/// `repr(str)`: picks the quote character the way CPython does, escapes
/// backslashes, the chosen quote, control characters and non-printable code points.
pub fn str_repr(s: &str) -> String {
    let quote = if s.contains('\'') && !s.contains('"') {
        '"'
    } else {
        '\''
    };
    let mut out = String::with_capacity(s.len() + 2);
    out.push(quote);
    for c in s.chars() {
        let u = c as u32;
        match c {
            '\\' => out.push_str("\\\\"),
            '\t' => out.push_str("\\t"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            _ if c == quote => {
                out.push('\\');
                out.push(c);
            }
            _ if u < 0x20 || u == 0x7f => out.push_str(&format!("\\x{u:02x}")),
            _ if u < 0x7f || is_printable(u) => out.push(c),
            _ if u <= 0xff => out.push_str(&format!("\\x{u:02x}")),
            _ if u <= 0xffff => out.push_str(&format!("\\u{u:04x}")),
            _ => out.push_str(&format!("\\U{u:08x}")),
        }
    }
    out.push(quote);
    out
}

/// CPython `str.isprintable()` for non-ASCII code points: everything except the
/// categories Cc, Cf, Co, Zl, Zp and Zs (other than the ASCII space).
///
/// Known gap: unassigned code points (Cn) are treated as printable, because
/// that needs the full Unicode database. Listed in README, "Differences from the Python runtime".
fn is_printable(u: u32) -> bool {
    !matches!(u,
        // Cc
        0x80..=0x9f
        // Zs, Zl, Zp
        | 0xa0 | 0x1680 | 0x2000..=0x200a | 0x2028 | 0x2029 | 0x202f | 0x205f | 0x3000
        // Cf
        | 0xad | 0x600..=0x605 | 0x61c | 0x6dd | 0x70f | 0x890..=0x891 | 0x8e2 | 0x180e
        | 0x200b..=0x200f | 0x202a..=0x202e | 0x2060..=0x2064 | 0x2066..=0x206f | 0xfeff
        | 0xfff9..=0xfffb | 0x110bd | 0x110cd | 0x13430..=0x1343f | 0x1bca0..=0x1bca3
        | 0x1d173..=0x1d17a | 0xe0001 | 0xe0020..=0xe007f
        // Co
        | 0xe000..=0xf8ff | 0xf0000..=0xffffd | 0x100000..=0x10fffd
        // Noncharacters (Cn) that are cheap to name
        | 0xfffe | 0xffff
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn float_repr_basics() {
        assert_eq!(float_repr(1.0), "1.0");
        assert_eq!(float_repr(1e-5), "1e-05");
        assert_eq!(float_repr(1e16), "1e+16");
        assert_eq!(float_repr(0.37), "0.37");
        assert_eq!(float_repr(-0.0), "-0.0");
    }

    #[test]
    fn strip_includes_information_separators() {
        let s: String = [0x1c, 0x20, 0x78, 0x1f]
            .iter()
            .map(|u| char::from_u32(*u).unwrap())
            .collect();
        assert_eq!(py_strip(&s), "x");
    }

    #[test]
    fn repr_quote_choice() {
        assert_eq!(str_repr("it's"), "\"it's\"");
        assert_eq!(str_repr("both ' and \""), "'both \\' and \"'");
    }
}
