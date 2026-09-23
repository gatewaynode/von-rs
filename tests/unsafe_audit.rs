//! Keeps the crate's single `unsafe` block tracked. See README.md, "Unsafe code audit".
//!
//! `static_inventory` runs in every `cargo test` and fails if an `unsafe` block
//! or `allow(unsafe_code)` appears anywhere other than the audited site, if the
//! site loses its SAFETY comment, or if the lints or the README entry are removed.
//! `mapping_is_released_after_load` checks the safety argument's key claim at
//! runtime and needs model weights.

use std::fs;
use std::path::{Path, PathBuf};

const AUDITED_FILE: &str = "src/weights.rs";
const AUDITED_FN: &str = "with_mapped_weights";

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn rust_files(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            rust_files(&path, out);
        } else if path.extension().is_some_and(|e| e == "rs") {
            out.push(path);
        }
    }
}

/// Code portion of a line: drops `//` comments. Good enough for this crate,
/// which has no `//` inside string literals on lines that mention `unsafe`.
fn code(line: &str) -> &str {
    line.split("//").next().unwrap_or("")
}

#[test]
fn static_inventory() {
    let root = root();
    let this_file = root.join("tests/unsafe_audit.rs");
    let mut files = Vec::new();
    for dir in ["src", "tests", "examples", "benches"] {
        rust_files(&root.join(dir), &mut files);
    }

    let mut unsafe_sites = Vec::new();
    let mut allow_sites = Vec::new();
    for file in files.iter().filter(|f| **f != this_file) {
        let rel = file.strip_prefix(&root).unwrap().display().to_string();
        let text = fs::read_to_string(file).unwrap();
        for (i, line) in text.lines().enumerate() {
            let c = code(line);
            if [
                "unsafe {",
                "unsafe fn",
                "unsafe impl",
                "unsafe trait",
                "unsafe extern",
            ]
            .iter()
            .any(|p| c.contains(p))
            {
                unsafe_sites.push((rel.clone(), i));
            }
            if c.contains("allow(unsafe_code)") {
                allow_sites.push((rel.clone(), i));
            }
        }
    }

    assert_eq!(
        unsafe_sites.len(),
        1,
        "expected exactly one unsafe site ({AUDITED_FILE}); found {unsafe_sites:?}. \
         New unsafe code needs a README audit entry and an update to this test."
    );
    assert_eq!(
        allow_sites.len(),
        1,
        "expected exactly one allow(unsafe_code); found {allow_sites:?}"
    );
    let (file, line) = &unsafe_sites[0];
    assert_eq!(file, AUDITED_FILE);
    assert_eq!(&allow_sites[0].0, AUDITED_FILE);

    // The block sits inside the audited function and is preceded by a SAFETY comment.
    let text = fs::read_to_string(root.join(AUDITED_FILE)).unwrap();
    let lines: Vec<&str> = text.lines().collect();
    let fn_line = lines
        .iter()
        .position(|l| l.contains(&format!("fn {AUDITED_FN}")))
        .expect("audited fn");
    assert!(
        allow_sites[0].1 < fn_line && fn_line < *line,
        "unsafe block must be inside {AUDITED_FN}"
    );
    assert!(
        lines[fn_line..*line]
            .iter()
            .any(|l| l.trim_start().starts_with("// SAFETY:")),
        "the unsafe block in {AUDITED_FN} lost its SAFETY comment"
    );

    let cargo = fs::read_to_string(root.join("Cargo.toml")).unwrap();
    assert!(
        cargo.contains(r#"unsafe_code = "deny""#),
        "Cargo.toml must deny unsafe_code"
    );
    assert!(
        cargo.contains(r#"undocumented_unsafe_blocks = "deny""#),
        "Cargo.toml must deny undocumented unsafe"
    );

    let readme = fs::read_to_string(root.join("README.md")).unwrap();
    let audit = readme
        .split("## Unsafe code audit")
        .nth(1)
        .expect("README.md needs an 'Unsafe code audit' section");
    assert!(
        audit.contains(AUDITED_FILE) && audit.contains(AUDITED_FN),
        "README audit must name the site"
    );
}

/// Invariant 2 of the SAFETY comment: after loading, nothing reads the mapped
/// file. Loads from a copy-on-write clone of the weights, truncates the clone to
/// zero bytes, then runs inference. If any tensor still pointed into the mapping,
/// the process would fault (SIGBUS) or the logits would change.
///
///     VON_WEIGHTS=checkpoints/von-1.1 cargo test --release --test unsafe_audit -- --ignored
#[test]
#[ignore = "needs model weights (VON_WEIGHTS)"]
fn mapping_is_released_after_load() {
    let src = PathBuf::from(std::env::var("VON_WEIGHTS").expect("set VON_WEIGHTS"));
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("unsafe-audit-ckpt");
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    for name in [
        "option_marker.safetensors",
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
    ] {
        // On APFS this is a clonefile: instant, and truncating the clone leaves the original intact.
        fs::copy(src.join(name), dir.join(name)).unwrap();
    }

    let von = von::Von::load(von::LoadOptions {
        checkpoint_dir: Some(dir.clone()),
        ..Default::default()
    })
    .unwrap();
    let packed = "Is the disk full? df reports 100% on /var [SEP] [MASK] Yes [MASK] No";
    let before = von.backend().model().option_logits(packed, 2).unwrap();

    let weights = fs::OpenOptions::new()
        .write(true)
        .open(dir.join("option_marker.safetensors"))
        .unwrap();
    weights.set_len(0).unwrap();
    drop(weights);

    let after = von.backend().model().option_logits(packed, 2).unwrap();
    assert_eq!(
        before, after,
        "logits changed after the weights file was truncated"
    );
    fs::remove_dir_all(&dir).unwrap();
}
