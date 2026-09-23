//! `VonClient` against a one-shot fake HTTP server on localhost (no weights).
#![cfg(feature = "client")]

use std::io::{BufRead, BufReader, Read, Write};
use std::net::TcpListener;
use std::thread::JoinHandle;
use std::time::Duration;

use indexmap::IndexMap;
use serde_json::{Value, json};
use von::{Answer, ClientOptions, Decider, Noul, Question, VonClient, VonError, judge};

struct Captured {
    request_line: String,
    headers: Vec<(String, String)>,
    body: Value,
}

impl Captured {
    fn header(&self, name: &str) -> Option<&str> {
        self.headers
            .iter()
            .find(|(k, _)| k.eq_ignore_ascii_case(name))
            .map(|(_, v)| v.as_str())
    }
}

/// Serves exactly one request with `status` and `body`, and returns what it received.
fn serve_once(status: &'static str, body: &'static str) -> (String, JoinHandle<Captured>) {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let base = format!("http://{}", listener.local_addr().unwrap());
    let handle = std::thread::spawn(move || {
        let (stream, _) = listener.accept().unwrap();
        let mut reader = BufReader::new(stream.try_clone().unwrap());
        let mut request_line = String::new();
        reader.read_line(&mut request_line).unwrap();
        let mut headers = Vec::new();
        loop {
            let mut line = String::new();
            reader.read_line(&mut line).unwrap();
            let line = line.trim_end();
            if line.is_empty() {
                break;
            }
            let (k, v) = line.split_once(':').unwrap();
            headers.push((k.trim().to_string(), v.trim().to_string()));
        }
        let length: usize = headers
            .iter()
            .find(|(k, _)| k.eq_ignore_ascii_case("content-length"))
            .map(|(_, v)| v.parse().unwrap())
            .unwrap_or(0);
        let mut raw = vec![0; length];
        reader.read_exact(&mut raw).unwrap();
        let mut stream = stream;
        write!(
            stream,
            "HTTP/1.1 {status}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
            body.len()
        )
        .unwrap();
        Captured {
            request_line: request_line.trim_end().to_string(),
            headers,
            body: serde_json::from_slice(&raw).unwrap(),
        }
    });
    (base, handle)
}

fn client(base: String, api_key: Option<&str>) -> VonClient {
    VonClient::new(ClientOptions {
        api_key: api_key.map(String::from),
        base_url: Some(base),
        timeout: Some(Duration::from_secs(5)),
    })
    .unwrap()
}

fn one_noul() -> IndexMap<String, Question> {
    IndexMap::from([(
        "is_down".to_string(),
        Question::from(Noul {
            instructions: "Is it down?".into(),
            criteria: None,
        }),
    )])
}

const OK_BODY: &str = r#"{"model":"von-1.1.0","answers":{"is_down":{"type":"noul","noul":0.9}},"usage":{"input_tokens":7,"output_tokens":0},"extra":"ignored"}"#;

#[test]
fn posts_python_payload_with_bearer_auth() {
    let (base, server) = serve_once("200 OK", OK_BODY);
    // A trailing slash is stripped, as with Python's rstrip('/').
    let c = client(format!("{base}/"), Some("secret"));
    assert_eq!(c.endpoint(), format!("{base}/v1/systemone"));

    let resp = c
        .system_one(&json!({"disk": "100%"}), &one_noul(), None)
        .unwrap();
    assert_eq!(resp.model, "von-1.1.0");
    assert!(matches!(resp.answers["is_down"], Answer::Noul(ref a) if a.noul == 0.9));
    assert_eq!(resp.usage.input_tokens, 7);

    let got = server.join().unwrap();
    assert_eq!(got.request_line, "POST /v1/systemone HTTP/1.1");
    assert_eq!(got.header("authorization"), Some("Bearer secret"));
    assert_eq!(got.header("content-type"), Some("application/json"));
    assert_eq!(
        got.body,
        json!({
            "model": "von-latest",
            "state": {"disk": "100%"},
            "questions": {"is_down": {"type": "noul", "instructions": "Is it down?", "criteria": null}}
        })
    );
}

#[test]
fn no_key_means_no_authorization_header() {
    let (base, server) = serve_once("200 OK", OK_BODY);
    // Through the Decider trait and a helper, with an explicit model.
    let p = judge(
        &client(base, None),
        &json!("x"),
        "Is it down?",
        None,
        Some("von-1.1"),
    );
    // The helper asks under "judgment", which the canned body lacks.
    assert!(matches!(p, Err(VonError::UnexpectedResponse(_))), "{p:?}");
    let got = server.join().unwrap();
    assert_eq!(got.header("authorization"), None);
    assert_eq!(got.body["model"], "von-1.1");
    assert!(got.body["questions"].get("judgment").is_some());
}

#[test]
fn non_2xx_is_an_http_error_with_the_body() {
    let (base, server) = serve_once("401 Unauthorized", r#"{"detail":"bad key"}"#);
    let err =
        Decider::system_one(&client(base, Some("k")), &json!("x"), &one_noul(), "von").unwrap_err();
    match err {
        VonError::Http { status, body, .. } => {
            assert_eq!(status, 401);
            assert!(body.contains("bad key"));
        }
        other => panic!("expected Http, got {other:?}"),
    }
    server.join().unwrap();
}

#[test]
fn redirects_are_not_followed() {
    let (base, server) = serve_once("307 Temporary Redirect", "{}");
    let err = client(base, None)
        .system_one(&json!("x"), &one_noul(), None)
        .unwrap_err();
    assert!(matches!(err, VonError::Http { status: 307, .. }), "{err:?}");
    server.join().unwrap();
}

#[test]
fn malformed_body_is_an_unexpected_response() {
    let (base, server) = serve_once("200 OK", r#"{"model":"x"}"#);
    let err = client(base, None)
        .system_one(&json!("x"), &one_noul(), None)
        .unwrap_err();
    assert!(matches!(err, VonError::UnexpectedResponse(_)), "{err:?}");
    server.join().unwrap();
}

#[test]
fn connection_failure_is_a_request_error() {
    // Bind then drop, so the port is very likely closed.
    let port = TcpListener::bind("127.0.0.1:0")
        .unwrap()
        .local_addr()
        .unwrap()
        .port();
    let err = client(format!("http://127.0.0.1:{port}"), None)
        .system_one(&json!("x"), &one_noul(), None)
        .unwrap_err();
    assert!(matches!(err, VonError::Request { .. }), "{err:?}");
}
