//! Option-marker sequence packing (port of `OptionMarkerModel.pack_sequence`).

use std::sync::LazyLock;

use regex::Regex;

use crate::pyfmt::py_strip;

/// Python's `re` `\d` on `str`: any Unicode decimal digit (category Nd).
static DIGIT_RUN: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"\d+").unwrap());

/// `"{question} {state}".strip() {sep} {mask} opt1 {mask} opt2 …`
///
/// With an empty question the prefix is just the stripped state. Every option
/// is stripped, and the trailing layout is kept even when the prefix is empty.
pub fn pack_sequence(
    state: &str,
    question: &str,
    options: &[String],
    mask: &str,
    sep: &str,
) -> String {
    let prefix = if question.is_empty() {
        py_strip(state).to_string()
    } else {
        py_strip(&format!("{question} {state}")).to_string()
    };
    let packed_options = options
        .iter()
        .map(|opt| format!("{mask} {}", py_strip(opt)))
        .collect::<Vec<_>>()
        .join(" ");
    format!("{prefix} {sep} {packed_options}")
}

/// Spaces out every digit of every digit run, `"2026"` → `"2 0 2 6"` (port of
/// `split_digits`). Used when the checkpoint was trained with `digit_split`, so
/// each digit becomes its own token.
///
/// `\d` follows the regex crate's Unicode tables, which can be newer than the
/// running Python's: digits added in Unicode 16 split here but not on Python 3.12.
pub fn split_digits(text: &str) -> String {
    DIGIT_RUN
        .replace_all(text, |caps: &regex::Captures| {
            let run = &caps[0];
            let mut out = String::with_capacity(run.len() * 2);
            for (i, c) in run.chars().enumerate() {
                if i > 0 {
                    out.push(' ');
                }
                out.push(c);
            }
            out
        })
        .into_owned()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layout() {
        let opts = vec![" a ".to_string(), "b".to_string()];
        assert_eq!(
            pack_sequence(" s ", "Q?", &opts, "[MASK]", "[SEP]"),
            "Q?  s [SEP] [MASK] a [MASK] b"
        );
        assert_eq!(
            pack_sequence(" s ", "", &opts, "[MASK]", "[SEP]"),
            "s [SEP] [MASK] a [MASK] b"
        );
        assert_eq!(
            pack_sequence("", "Q?", &opts, "[MASK]", "[SEP]"),
            "Q? [SEP] [MASK] a [MASK] b"
        );
    }

    #[test]
    fn digits_are_spaced_out() {
        assert_eq!(split_digits("fee 2026, v1.2"), "fee 2 0 2 6, v1.2");
        assert_eq!(split_digits(""), "");
    }
}
