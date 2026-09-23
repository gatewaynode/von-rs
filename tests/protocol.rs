//! Replays the Python server's and CLI's recorded behaviour
//! (`tests/fixtures/protocol.json`, from `tools/export_protocol.py`) against the
//! Rust server and CLI, with a fake `Decider` standing in for the model on both
//! sides. No weights needed.
#![cfg(all(feature = "server", feature = "cli"))]

use std::sync::{Arc, Mutex};

use axum::body::Body;
use axum::http::Request;
use http_body_util::BodyExt;
use indexmap::IndexMap;
use serde_json::Value;
use tower::ServiceExt;
use von::server::{ServerConfig, parse_origins, router};
use von::{Answer, Decider, Question, SystemOneResponse, Usage, VonError};

const FIXTURES: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures");

fn fixture() -> Value {
    let text = std::fs::read_to_string(format!("{FIXTURES}/protocol.json")).unwrap();
    serde_json::from_str(&text).unwrap()
}

/// The Python fake engine's twin: answers every question with the canned answer
/// for its type (or fails), and records the request.
struct Fake {
    canned: Value,
    error: Option<String>,
    saw: Mutex<Option<Value>>,
}

impl Fake {
    fn new(canned: &Value, error: Option<&str>) -> Arc<Self> {
        Arc::new(Fake {
            canned: canned.clone(),
            error: error.map(String::from),
            saw: Mutex::new(None),
        })
    }

    fn saw(&self) -> Value {
        self.saw.lock().unwrap().clone().unwrap_or(Value::Null)
    }
}

impl Decider for Fake {
    fn system_one(
        &self,
        state: &Value,
        questions: &IndexMap<String, Question>,
        model: &str,
    ) -> von::Result<SystemOneResponse> {
        *self.saw.lock().unwrap() = Some(serde_json::json!({
            "state": state, "questions": questions, "model": model,
        }));
        if let Some(e) = &self.error {
            return Err(VonError::InvalidQuestion(e.clone()));
        }
        let answers = questions
            .iter()
            .map(|(id, q)| {
                let kind = match q {
                    Question::Noul(_) => "noul",
                    Question::Choice(_) => "choice",
                    Question::Score(_) => "score",
                };
                let answer: Answer = serde_json::from_value(self.canned[kind].clone()).unwrap();
                (id.clone(), answer)
            })
            .collect();
        Ok(SystemOneResponse {
            model: "von-1.1.0".into(),
            answers,
            usage: Usage {
                input_tokens: 12,
                output_tokens: 0,
            },
        })
    }
}

/// What Python's fake engine recorded, with questions normalized through the
/// Rust types (Python passes raw dicts for `eval`, e.g. without `type`).
fn normalized_saw(saw: &Value) -> Value {
    if saw.is_null() {
        return Value::Null;
    }
    let questions: IndexMap<String, Question> =
        serde_json::from_value(saw["questions"].clone()).unwrap();
    serde_json::json!({
        "state": saw["state"], "questions": questions, "model": saw["model"],
    })
}

/// FastAPI's `json_invalid` carries Python's decoder message in `ctx.error`;
/// serde's wording differs, so that one field is not compared.
fn without_decoder_message(body: &str) -> Value {
    let mut v: Value = serde_json::from_str(body).unwrap_or(Value::String(body.into()));
    if let Some(errors) = v.get_mut("detail").and_then(Value::as_array_mut) {
        for e in errors {
            if e["type"] == "json_invalid" {
                e["ctx"]["error"] = Value::Null;
            }
        }
    }
    v
}

const KEEP_HEADERS: [&str; 5] = [
    "access-control-",
    "vary",
    "content-type",
    "location",
    "allow",
];

#[tokio::test]
async fn server_matches_python() {
    let fx = fixture();
    let mut failures = Vec::new();
    for case in fx["server"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let config = &case["config"];
        let fake = Fake::new(&fx["canned"], case["engine_error"].as_str());
        let app = router(
            fake.clone(),
            ServerConfig {
                api_key: config["key"].as_str().map(String::from),
                cors_origins: parse_origins(config["cors"].as_str().unwrap_or("*")),
                max_in_flight: 1,
            },
        );

        let req = &case["request"];
        let mut builder = Request::builder()
            .method(req["method"].as_str().unwrap())
            .uri(req["path"].as_str().unwrap())
            .header("host", "testserver");
        for (k, v) in req["headers"].as_object().unwrap() {
            builder = builder.header(k, v.as_str().unwrap());
        }
        let body = req["body"].as_str().unwrap_or("").to_string();
        let response = app
            .oneshot(builder.body(Body::from(body)).unwrap())
            .await
            .unwrap();

        let want = &case["response"];
        let status = response.status().as_u16();
        let mut headers: Vec<(String, String)> = response
            .headers()
            .iter()
            .filter(|(k, _)| KEEP_HEADERS.iter().any(|p| k.as_str().starts_with(p)))
            .map(|(k, v)| (k.to_string(), v.to_str().unwrap().to_string()))
            .collect();
        headers.sort();
        let mut want_headers: Vec<(String, String)> = want["headers"]
            .as_object()
            .unwrap()
            .iter()
            .map(|(k, v)| (k.clone(), v.as_str().unwrap().to_string()))
            .collect();
        want_headers.sort();
        let bytes = response.into_body().collect().await.unwrap().to_bytes();
        let got_body = String::from_utf8(bytes.to_vec()).unwrap();
        let want_body = want["body"].as_str().unwrap();

        let body_ok = if want_body.contains("\"json_invalid\"") {
            without_decoder_message(&got_body) == without_decoder_message(want_body)
        } else {
            got_body == want_body
        };
        if status != want["status"] || headers != want_headers || !body_ok {
            failures.push(format!(
                "{name}:\n  status {status} vs {}\n  headers {headers:?}\n     vs   {want_headers:?}\n  body {got_body}\n    vs {want_body}",
                want["status"]
            ));
        }
        if fake.saw() != normalized_saw(&case["engine_saw"]) {
            failures.push(format!(
                "{name}: engine saw {}\n  vs {}",
                fake.saw(),
                case["engine_saw"]
            ));
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

#[test]
fn cli_matches_python() {
    let fx = fixture();
    let cli_dir = format!("{FIXTURES}/cli/");
    let mut failures = Vec::new();
    for case in fx["cli"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let mut args = vec!["von".to_string()];
        let raw: Vec<&str> = case["args"]
            .as_array()
            .unwrap()
            .iter()
            .map(|a| a.as_str().unwrap())
            .collect();
        for (i, a) in raw.iter().enumerate() {
            // The Python run used the fixture dir as its working directory.
            let is_file = i > 0 && raw[i - 1] == "eval";
            args.push(if is_file {
                format!("{cli_dir}{a}")
            } else {
                a.to_string()
            });
        }

        let fake = Fake::new(&fx["canned"], None);
        let mut connect =
            |_opts: von::LoadOptions| -> von::Result<Arc<dyn Decider>> { Ok(fake.clone()) };
        let (mut out, mut err) = (Vec::new(), Vec::new());
        let code = von::cli::run(&args, &mut connect, &mut out, &mut err);
        let out = String::from_utf8(out).unwrap();
        let err = String::from_utf8(err).unwrap().replace(&cli_dir, "");

        let want_out = case["stdout"].as_str().unwrap().replace(
            "von, version 1.0.0",
            &format!("von, version {}", von::cli::CLI_VERSION),
        );
        let want_err = case["stderr"].as_str().unwrap();
        let err_ok = match (name, case["exception"].as_str()) {
            // Python ends with an uncaught exception (a traceback, exit 1); Rust prints it.
            (_, Some(exc)) => err == format!("Error: {}\n", exc.split_once(": ").unwrap().1),
            // click and clap word usage errors differently; only the exit code is compared.
            ("decide missing choices", _) => !err.is_empty(),
            // Python's JSON decoder message differs from serde's.
            ("eval invalid json", _) => {
                err.starts_with(want_err.split("is not valid JSON: ").next().unwrap())
            }
            _ => err == want_err,
        };
        if i64::from(code) != case["exit"].as_i64().unwrap() || out != want_out || !err_ok {
            failures.push(format!(
                "{name}: exit {code} vs {}\n  stdout {out:?}\n      vs {want_out:?}\n  stderr {err:?}\n      vs {want_err:?}",
                case["exit"]
            ));
        }
        if fake.saw() != normalized_saw(&case["engine_saw"]) {
            failures.push(format!(
                "{name}: engine saw {}\n  vs {}",
                fake.saw(),
                case["engine_saw"]
            ));
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}
