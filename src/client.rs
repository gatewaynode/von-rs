//! Blocking HTTP client for a remote `/v1/systemone` server (port of the remote
//! half of `client.py`; the in-process half is [`crate::Von`] itself).
//!
//! Uses reqwest's blocking client, which runs its own runtime: do not call it
//! from inside an async context. (Async support is planned.)

use std::time::Duration;

use indexmap::IndexMap;
use serde_json::{Value, json};

use crate::api::{DEFAULT_MODEL, Decider};
use crate::error::{Result, VonError};
use crate::types::{Question, SystemOneResponse};

pub const DEFAULT_BASE_URL: &str = "http://localhost:8000";
pub const DEFAULT_TIMEOUT: Duration = Duration::from_secs(30);

/// Settings for [`VonClient`]. Unset fields fall back to the environment and
/// then to the defaults, as in Python.
#[derive(Debug, Clone, Default)]
pub struct ClientOptions {
    /// Bearer token; defaults to `VON_API_KEY`, then `TYPESAFE_API_KEY`, then none.
    pub api_key: Option<String>,
    /// Server root; defaults to `VON_BASE_URL`, then [`DEFAULT_BASE_URL`].
    pub base_url: Option<String>,
    /// Whole-request timeout; defaults to [`DEFAULT_TIMEOUT`].
    pub timeout: Option<Duration>,
}

impl ClientOptions {
    /// Fills unset fields from `env`. Empty values count as unset, as with
    /// Python's `x or os.environ.get(...)`.
    fn resolve(self, env: impl Fn(&str) -> Option<String>) -> (Option<String>, String, Duration) {
        let set = |v: Option<String>| v.filter(|s| !s.is_empty());
        let api_key = set(self.api_key)
            .or_else(|| set(env("VON_API_KEY")))
            .or_else(|| set(env("TYPESAFE_API_KEY")));
        let base_url = set(self.base_url)
            .or_else(|| set(env("VON_BASE_URL")))
            .unwrap_or_else(|| DEFAULT_BASE_URL.to_string());
        (api_key, base_url, self.timeout.unwrap_or(DEFAULT_TIMEOUT))
    }
}

/// Sends System One requests to a remote Von (or compatible) server.
#[derive(Debug, Clone)]
pub struct VonClient {
    http: reqwest::blocking::Client,
    endpoint: String,
    api_key: Option<String>,
}

impl VonClient {
    pub fn new(opts: ClientOptions) -> Result<Self> {
        let (api_key, base_url, timeout) = opts.resolve(|k| std::env::var(k).ok());
        let endpoint = format!("{}/v1/systemone", base_url.trim_end_matches('/'));
        let http = reqwest::blocking::Client::builder()
            .timeout(timeout)
            // httpx does not follow redirects, and raise_for_status rejects a 3xx.
            .redirect(reqwest::redirect::Policy::none())
            .build()
            .map_err(|e| VonError::Request {
                url: endpoint.clone(),
                reason: e.to_string(),
            })?;
        Ok(Self {
            http,
            endpoint,
            api_key,
        })
    }

    /// A client configured entirely from the environment.
    pub fn from_env() -> Result<Self> {
        Self::new(ClientOptions::default())
    }

    /// The full URL requests are sent to.
    pub fn endpoint(&self) -> &str {
        &self.endpoint
    }

    /// POSTs `{model, state, questions}` and parses the response. A non-2xx
    /// status is an error. `model` defaults to [`DEFAULT_MODEL`].
    pub fn system_one(
        &self,
        state: &Value,
        questions: &IndexMap<String, Question>,
        model: Option<&str>,
    ) -> Result<SystemOneResponse> {
        let payload = json!({
            "model": model.unwrap_or(DEFAULT_MODEL),
            "state": state,
            "questions": questions,
        });
        let mut request = self.http.post(&self.endpoint).json(&payload);
        if let Some(key) = &self.api_key {
            request = request.bearer_auth(key);
        }
        let failed = |e: &dyn std::fmt::Display| VonError::Request {
            url: self.endpoint.clone(),
            reason: e.to_string(),
        };
        let response = request.send().map_err(|e| failed(&e))?;
        let status = response.status();
        let body = response.text().map_err(|e| failed(&e))?;
        if !status.is_success() {
            return Err(VonError::Http {
                status: status.as_u16(),
                url: self.endpoint.clone(),
                body,
            });
        }
        serde_json::from_str(&body)
            .map_err(|e| VonError::UnexpectedResponse(format!("{e}; body: {body}")))
    }
}

impl Decider for VonClient {
    fn system_one(
        &self,
        state: &Value,
        questions: &IndexMap<String, Question>,
        model: &str,
    ) -> Result<SystemOneResponse> {
        VonClient::system_one(self, state, questions, Some(model))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn env(vars: &'static [(&'static str, &'static str)]) -> impl Fn(&str) -> Option<String> {
        move |k| {
            vars.iter()
                .find(|(name, _)| *name == k)
                .map(|(_, v)| v.to_string())
        }
    }

    #[test]
    fn defaults_without_environment() {
        let (key, url, timeout) = ClientOptions::default().resolve(env(&[]));
        assert_eq!(key, None);
        assert_eq!(url, DEFAULT_BASE_URL);
        assert_eq!(timeout, Duration::from_secs(30));
    }

    #[test]
    fn environment_fallbacks_in_python_order() {
        let (key, url, _) = ClientOptions::default().resolve(env(&[
            ("TYPESAFE_API_KEY", "ts"),
            ("VON_BASE_URL", "http://remote:9000/"),
        ]));
        assert_eq!(key.as_deref(), Some("ts"));
        assert_eq!(url, "http://remote:9000/");

        let (key, _, _) = ClientOptions::default()
            .resolve(env(&[("VON_API_KEY", "von"), ("TYPESAFE_API_KEY", "ts")]));
        assert_eq!(key.as_deref(), Some("von"));

        // Empty values are skipped, like Python's `or`.
        let (key, url, _) = ClientOptions::default().resolve(env(&[
            ("VON_API_KEY", ""),
            ("TYPESAFE_API_KEY", "ts"),
            ("VON_BASE_URL", ""),
        ]));
        assert_eq!(key.as_deref(), Some("ts"));
        assert_eq!(url, DEFAULT_BASE_URL);
    }

    #[test]
    fn explicit_options_win() {
        let opts = ClientOptions {
            api_key: Some("mine".into()),
            base_url: Some("http://x".into()),
            timeout: Some(Duration::from_secs(5)),
        };
        let (key, url, timeout) =
            opts.resolve(env(&[("VON_API_KEY", "von"), ("VON_BASE_URL", "http://y")]));
        assert_eq!(
            (key.as_deref(), url.as_str(), timeout),
            (Some("mine"), "http://x", Duration::from_secs(5))
        );
    }
}
