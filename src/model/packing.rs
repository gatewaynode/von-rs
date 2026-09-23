//! Option-marker sequence packing (port of `OptionMarkerModel.pack_sequence`).

use crate::pyfmt::py_strip;

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
}
