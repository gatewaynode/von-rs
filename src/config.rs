//! The local settings file the `von` binary reads at startup:
//! `$XDG_CONFIG_HOME/von/von.env`, or `~/.config/von/von.env` when
//! `XDG_CONFIG_HOME` is unset.
//!
//! One `KEY=value` per line. Blank lines and lines starting with `#` are
//! skipped, an `export ` prefix is allowed, and one pair of matching quotes
//! around the value is removed. There are no inline comments. A value of `~`
//! or starting with `~/` is expanded against `HOME`. Unknown keys are kept
//! but unused, so the same file can hold settings for other tools.
//!
//! Environment variables always win over the file. The library never reads
//! the file by itself: [`crate::Von::load`], [`crate::server::ServerConfig::from_env`]
//! and the client read only the environment, and the binary passes file values
//! to them explicitly.

use std::collections::HashMap;
use std::ffi::OsString;
use std::path::{Path, PathBuf};

/// Settings read from a `von.env` file.
#[derive(Debug, Clone, Default)]
pub struct Config {
    path: Option<PathBuf>,
    values: HashMap<String, String>,
}

impl Config {
    /// Where the settings file lives, or `None` when neither an absolute
    /// `XDG_CONFIG_HOME` nor `HOME` is set.
    pub fn default_path() -> Option<PathBuf> {
        default_path_from(|k| std::env::var_os(k))
    }

    /// Reads the file at [`Config::default_path`]. See [`Config::from_file`].
    pub fn load() -> Self {
        Self::default_path()
            .map(|p| Self::from_file(&p))
            .unwrap_or_default()
    }

    /// Reads `path`. A missing file gives empty settings. An unreadable file
    /// or a malformed line is logged as a warning and skipped, so a bad
    /// settings file never stops the binary.
    pub fn from_file(path: &Path) -> Self {
        let text = match std::fs::read_to_string(path) {
            Ok(text) => text,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Self::default(),
            Err(e) => {
                tracing::warn!("Ignoring settings file {}: {e}", path.display());
                return Self::default();
            }
        };
        let home = std::env::var_os("HOME").map(PathBuf::from);
        let (values, bad_lines) = parse(&text, home.as_deref());
        for line in bad_lines {
            tracing::warn!(
                "Ignoring {} line {line}: expected KEY=value",
                path.display()
            );
        }
        tracing::debug!("Read {} settings from {}", values.len(), path.display());
        Self {
            path: Some(path.to_path_buf()),
            values,
        }
    }

    /// Settings parsed from `text` as if read from a file (for tests).
    #[cfg(all(test, feature = "cli"))]
    pub(crate) fn from_text(text: &str) -> Self {
        Self {
            path: None,
            values: parse(text, None).0,
        }
    }

    /// The file that was read, if it existed.
    pub fn path(&self) -> Option<&Path> {
        self.path.as_deref()
    }

    /// The file's value for `key`.
    pub fn get(&self, key: &str) -> Option<&str> {
        self.values.get(key).map(String::as_str)
    }

    /// The environment's value for `key`, else the file's.
    pub fn var(&self, key: &str) -> Option<String> {
        std::env::var(key)
            .ok()
            .or_else(|| self.get(key).map(String::from))
    }

    /// The file's value for `key`, only when the environment does not set it.
    /// Use this to fill options whose default the library already takes from
    /// the environment.
    pub fn fallback(&self, key: &str) -> Option<String> {
        if std::env::var_os(key).is_some() {
            return None;
        }
        self.get(key).map(String::from)
    }
}

fn default_path_from(env: impl Fn(&str) -> Option<OsString>) -> Option<PathBuf> {
    let set = |k: &str| env(k).filter(|v| !v.is_empty()).map(PathBuf::from);
    // The XDG spec says a relative XDG_CONFIG_HOME is invalid and must be ignored.
    let base = set("XDG_CONFIG_HOME")
        .filter(|p| p.is_absolute())
        .or_else(|| set("HOME").map(|h| h.join(".config")))?;
    Some(base.join("von").join("von.env"))
}

/// Parses `KEY=value` lines. Returns the values (the last one wins) and the
/// 1-based numbers of malformed lines.
fn parse(text: &str, home: Option<&Path>) -> (HashMap<String, String>, Vec<usize>) {
    let mut values = HashMap::new();
    let mut bad_lines = Vec::new();
    for (i, line) in text.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let line = line.strip_prefix("export ").unwrap_or(line);
        match line.split_once('=') {
            Some((key, value)) if is_key(key.trim()) => {
                values.insert(
                    key.trim().to_string(),
                    expand_home(unquote(value.trim()), home),
                );
            }
            _ => bad_lines.push(i + 1),
        }
    }
    (values, bad_lines)
}

fn is_key(key: &str) -> bool {
    let mut chars = key.chars();
    chars
        .next()
        .is_some_and(|c| c.is_ascii_alphabetic() || c == '_')
        && chars.all(|c| c.is_ascii_alphanumeric() || c == '_')
}

fn unquote(value: &str) -> &str {
    for quote in ['"', '\''] {
        if let Some(inner) = value
            .strip_prefix(quote)
            .and_then(|v| v.strip_suffix(quote))
        {
            return inner;
        }
    }
    value
}

fn expand_home(value: &str, home: Option<&Path>) -> String {
    match (home, value) {
        (Some(home), "~") => home.display().to_string(),
        (Some(home), v) if v.starts_with("~/") => home.join(&v[2..]).display().to_string(),
        _ => value.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_lines() {
        let text = "\
# models
VON_DEVICE=cpu
export VON_BACKEND = \"von-1.1\"
VON_API_KEY='a=b'

VON_CHECKPOINT_DIR=~/models/von-1.1
VON_CORS_ORIGINS=
not a pair
1BAD=x
VON_DEVICE=metal
";
        let (values, bad) = parse(text, Some(Path::new("/home/u")));
        assert_eq!(values["VON_DEVICE"], "metal", "the last value wins");
        assert_eq!(values["VON_BACKEND"], "von-1.1");
        assert_eq!(values["VON_API_KEY"], "a=b");
        assert_eq!(values["VON_CHECKPOINT_DIR"], "/home/u/models/von-1.1");
        assert_eq!(values["VON_CORS_ORIGINS"], "");
        assert_eq!(bad, vec![8, 9]);
    }

    #[test]
    fn home_expansion() {
        let home = Some(Path::new("/h"));
        assert_eq!(expand_home("~", home), "/h");
        assert_eq!(expand_home("~/x", home), "/h/x");
        assert_eq!(expand_home("~x", home), "~x");
        assert_eq!(expand_home("a/~/b", home), "a/~/b");
        assert_eq!(expand_home("~/x", None), "~/x");
    }

    #[test]
    fn default_path_prefers_absolute_xdg() {
        let env = |pairs: &'static [(&'static str, &'static str)]| {
            move |k: &str| {
                pairs
                    .iter()
                    .find(|(key, _)| *key == k)
                    .map(|(_, v)| OsString::from(v))
            }
        };
        assert_eq!(
            default_path_from(env(&[("XDG_CONFIG_HOME", "/x"), ("HOME", "/h")])),
            Some(PathBuf::from("/x/von/von.env"))
        );
        assert_eq!(
            default_path_from(env(&[("XDG_CONFIG_HOME", "rel"), ("HOME", "/h")])),
            Some(PathBuf::from("/h/.config/von/von.env"))
        );
        assert_eq!(
            default_path_from(env(&[("XDG_CONFIG_HOME", ""), ("HOME", "/h")])),
            Some(PathBuf::from("/h/.config/von/von.env"))
        );
        assert_eq!(default_path_from(env(&[])), None);
    }

    #[test]
    fn missing_file_is_empty() {
        let config = Config::from_file(Path::new("/nonexistent/von/von.env"));
        assert!(config.path().is_none());
        assert!(config.get("VON_DEVICE").is_none());
    }
}
