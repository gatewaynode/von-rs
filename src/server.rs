//! HTTP server (port of `server.py`): a drop-in replacement for the FastAPI app.
//!
//! Status codes, headers and bodies match FastAPI/Starlette byte for byte where
//! clients can observe them (checked against `tests/fixtures/protocol.json`):
//! request validation errors are 422 with pydantic-style `detail` lists and run
//! *before* the auth check, engine errors are 422 with a string `detail`, unknown
//! paths are 404, wrong methods (including `HEAD`) are 405 with `Allow`, and a
//! trailing slash redirects with 307. CORS follows Starlette's `CORSMiddleware`.
//!
//! Inference runs on the blocking pool behind a semaphore, so the async runtime
//! is never blocked and memory stays bounded under load.

use std::sync::Arc;

use axum::Router;
use axum::body::{Body, Bytes};
use axum::extract::{Request, State};
use axum::http::{HeaderMap, HeaderValue, Method, StatusCode, header};
use axum::middleware::{self, Next};
use axum::response::Response;
use axum::routing::any;
use indexmap::IndexMap;
use serde_json::{Map, Value, json};
use tokio::sync::Semaphore;

use crate::api::Decider;
use crate::pyfmt::py_strip;
use crate::pyjson;
use crate::types::Question;

/// Methods Starlette lists for `allow_methods=["*"]`.
const ALL_METHODS: [&str; 7] = ["DELETE", "GET", "HEAD", "OPTIONS", "PATCH", "POST", "PUT"];
/// `/health` reports the Python package version this server mirrors.
const PYTHON_SERVER_VERSION: &str = "1.2.3";
/// Every routed path; used for the trailing-slash redirect.
const ROUTES: [&str; 4] = ["/", "/health", "/v1/models", "/v1/systemone"];

/// Server settings. [`ServerConfig::from_env`] reads them as Python does.
#[derive(Debug, Clone)]
pub struct ServerConfig {
    /// Bearer token required on `/v1/systemone`; `None` disables auth.
    pub api_key: Option<String>,
    /// Allowed CORS origins, parsed like `VON_CORS_ORIGINS`.
    pub cors_origins: Vec<String>,
    /// Maximum concurrent inferences.
    pub max_in_flight: usize,
}

impl ServerConfig {
    /// `VON_API_KEY` (empty means unset) and `VON_CORS_ORIGINS` (default `*`).
    /// Python reads the key on every request; von-rs reads it once at startup.
    pub fn from_env(max_in_flight: usize) -> Self {
        Self::from_lookup(max_in_flight, |k| std::env::var(k).ok())
    }

    /// Like [`ServerConfig::from_env`], reading variables through `lookup`.
    pub fn from_lookup(max_in_flight: usize, lookup: impl Fn(&str) -> Option<String>) -> Self {
        Self {
            api_key: lookup("VON_API_KEY").filter(|k| !k.is_empty()),
            cors_origins: parse_origins(lookup("VON_CORS_ORIGINS").as_deref().unwrap_or("*")),
            max_in_flight,
        }
    }
}

/// `[o.strip() for o in raw.split(",") if o.strip()]`.
pub fn parse_origins(raw: &str) -> Vec<String> {
    raw.split(',')
        .map(py_strip)
        .filter(|o| !o.is_empty())
        .map(String::from)
        .collect()
}

#[derive(Clone)]
struct AppState {
    decider: Arc<dyn Decider>,
    api_key: Option<Arc<str>>,
    permits: Arc<Semaphore>,
}

/// The full application: routes, fallbacks and CORS.
pub fn router(decider: Arc<dyn Decider>, config: ServerConfig) -> Router {
    let state = AppState {
        decider,
        api_key: config.api_key.map(Arc::from),
        permits: Arc::new(Semaphore::new(config.max_in_flight.max(1))),
    };
    let cors = Arc::new(Cors::new(config.cors_origins));
    Router::new()
        .route("/", any(health))
        .route("/health", any(health))
        .route("/v1/models", any(models))
        .route("/v1/systemone", any(system_one))
        .fallback(fallback)
        .with_state(state)
        .layer(middleware::from_fn_with_state(cors, cors_layer))
}

/// Serves `app` on `listener` until Ctrl-C.
pub async fn serve(listener: tokio::net::TcpListener, app: Router) -> std::io::Result<()> {
    axum::serve(listener, app)
        .with_graceful_shutdown(async {
            let _ = tokio::signal::ctrl_c().await;
        })
        .await
}

// --- Handlers ------------------------------------------------------------------

async fn health(method: Method) -> Response {
    if method != Method::GET {
        return method_not_allowed(&method, "GET");
    }
    json_response(
        StatusCode::OK,
        &json!({
            "status": "ok",
            "service": "von-decision-server",
            "version": PYTHON_SERVER_VERSION,
            "engine": "von-1.2",
            "homage": "John von Neumann & Ludwig von Mises",
        }),
    )
}

async fn models(method: Method) -> Response {
    if method != Method::GET {
        return method_not_allowed(&method, "GET");
    }
    let models = [
        (
            "von-latest",
            "Current Von System One decision model",
            "2026-09-23",
        ),
        (
            "von-1.2.0",
            "Von 1.2 stable release (order-invariant option scoring)",
            "2026-09-23",
        ),
        (
            "von-1.1.0",
            "Von 1.1 alias (resolves to current model)",
            "2026-09-21",
        ),
        (
            "jev-latest",
            "TypeSafe Jev compatibility alias",
            "2026-09-21",
        ),
    ];
    json_response(
        StatusCode::OK,
        &json!({
            "models": models.iter().map(|(name, description, date)| json!({
                "name": name, "description": description, "release_date": date,
            })).collect::<Vec<_>>(),
            "object": "list",
            "data": models.iter().map(|(name, _, _)| json!({
                "id": name, "object": "model", "owned_by": "von",
            })).collect::<Vec<_>>(),
        }),
    )
}

async fn system_one(
    State(state): State<AppState>,
    method: Method,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    if method != Method::POST {
        return method_not_allowed(&method, "POST");
    }
    // FastAPI validates the body before the endpoint (and its auth check) runs.
    let request = match validate_request(&headers, &body) {
        Ok(r) => r,
        Err(errors) => {
            return json_response(
                StatusCode::UNPROCESSABLE_ENTITY,
                &json!({ "detail": errors }),
            );
        }
    };
    if let Some(expected) = &state.api_key
        && let Err(detail) = check_bearer(&headers, expected)
    {
        return detail_response(StatusCode::UNAUTHORIZED, detail);
    }

    let mut questions = IndexMap::with_capacity(request.questions.len());
    for (id, raw) in request.questions {
        match serde_json::from_value::<Question>(raw) {
            Ok(q) => {
                questions.insert(id, q);
            }
            Err(e) => {
                return detail_response(StatusCode::UNPROCESSABLE_ENTITY, &format!("{id}: {e}"));
            }
        }
    }

    let Ok(_permit) = state.permits.clone().acquire_owned().await else {
        return internal_error();
    };
    let decider = state.decider.clone();
    let result = tokio::task::spawn_blocking(move || {
        decider.system_one(&request.state, &questions, &request.model)
    })
    .await;
    match result {
        Ok(Ok(response)) => json_response(StatusCode::OK, &response),
        Ok(Err(e)) => detail_response(StatusCode::UNPROCESSABLE_ENTITY, &e.to_string()),
        Err(join_error) => {
            tracing::error!("inference task failed: {join_error}");
            internal_error()
        }
    }
}

/// 404, or Starlette's `redirect_slashes`: a path that matches a route once a
/// trailing slash is added or removed redirects there with 307.
async fn fallback(request: Request) -> Response {
    let headers = request.headers();
    let uri = request.uri();
    let path = uri.path();
    if path != "/" {
        let alternative = match path.strip_suffix('/') {
            Some(p) => p.to_string(),
            None => format!("{path}/"),
        };
        if ROUTES.contains(&alternative.as_str()) {
            let host = headers
                .get(header::HOST)
                .and_then(|h| h.to_str().ok())
                .unwrap_or("localhost");
            let query = uri.query().map(|q| format!("?{q}")).unwrap_or_default();
            return Response::builder()
                .status(StatusCode::TEMPORARY_REDIRECT)
                .header(
                    header::LOCATION,
                    format!("http://{host}{alternative}{query}"),
                )
                .body(Body::empty())
                .expect("static response parts are valid");
        }
    }
    detail_response(StatusCode::NOT_FOUND, "Not Found")
}

// --- Request validation (pydantic's view of `SystemOneRequest`) ------------------

struct SystemOneRequest {
    model: String,
    state: Value,
    questions: Map<String, Value>,
}

/// Parses and validates the body as FastAPI does, returning pydantic-style
/// error objects on failure.
fn validate_request(headers: &HeaderMap, body: &[u8]) -> Result<SystemOneRequest, Vec<Value>> {
    if body.is_empty() {
        return Err(vec![field_error(
            "missing",
            &["body"],
            "Field required",
            Value::Null,
        )]);
    }
    let value = if is_json_content_type(headers) {
        match serde_json::from_slice::<Value>(body) {
            Ok(v) => v,
            Err(e) => return Err(vec![json_invalid(body, &e)]),
        }
    } else {
        Value::String(String::from_utf8_lossy(body).into_owned())
    };
    let Value::Object(obj) = &value else {
        return Err(vec![field_error(
            "model_attributes_type",
            &["body"],
            "Input should be a valid dictionary or object to extract fields from",
            value,
        )]);
    };

    let mut errors = Vec::new();
    let model = match obj.get("model") {
        None => Some("von-latest".to_string()),
        Some(Value::String(s)) => Some(s.clone()),
        Some(other) => {
            errors.push(field_error(
                "string_type",
                &["body", "model"],
                "Input should be a valid string",
                other.clone(),
            ));
            None
        }
    };
    let state = obj.get("state").cloned();
    if state.is_none() {
        errors.push(missing(&["body", "state"], &value));
    }
    let questions = match obj.get("questions") {
        None => {
            errors.push(missing(&["body", "questions"], &value));
            None
        }
        Some(Value::Object(qs)) => {
            // Python's `questions` validator: an empty dict is a malformed request.
            if qs.is_empty() {
                errors.push(json!({
                    "type": "value_error",
                    "loc": ["body", "questions"],
                    "msg": "Value error, questions must contain at least one entry",
                    "input": {},
                    "ctx": { "error": {} },
                }));
            }
            for (id, q) in qs {
                if !q.is_object() {
                    errors.push(field_error(
                        "dict_type",
                        &["body", "questions", id],
                        "Input should be a valid dictionary",
                        q.clone(),
                    ));
                }
            }
            Some(qs.clone())
        }
        Some(other) => {
            errors.push(field_error(
                "dict_type",
                &["body", "questions"],
                "Input should be a valid dictionary",
                other.clone(),
            ));
            None
        }
    };
    match (model, state, questions) {
        (Some(model), Some(state), Some(questions)) if errors.is_empty() => Ok(SystemOneRequest {
            model,
            state,
            questions,
        }),
        _ => Err(errors),
    }
}

/// FastAPI parses the body as JSON only for `application/json` or `application/*+json`.
fn is_json_content_type(headers: &HeaderMap) -> bool {
    let Some(ct) = headers
        .get(header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
    else {
        return false;
    };
    let mime = ct
        .split(';')
        .next()
        .unwrap_or("")
        .trim()
        .to_ascii_lowercase();
    match mime.split_once('/') {
        Some(("application", subtype)) => subtype == "json" || subtype.ends_with("+json"),
        _ => false,
    }
}

fn field_error(kind: &str, loc: &[&str], msg: &str, input: Value) -> Value {
    json!({ "type": kind, "loc": loc, "msg": msg, "input": input })
}

fn missing(loc: &[&str], body: &Value) -> Value {
    field_error("missing", loc, "Field required", body.clone())
}

/// `json_invalid` with the error position as a character offset, like Python's
/// `JSONDecodeError.pos`. The `ctx.error` text is serde's, not Python's.
fn json_invalid(body: &[u8], e: &serde_json::Error) -> Value {
    let text = String::from_utf8_lossy(body);
    let line_start: usize = text
        .split_inclusive('\n')
        .take(e.line().saturating_sub(1))
        .map(|l| l.chars().count())
        .sum();
    let pos = line_start + e.column().saturating_sub(1);
    let reason = e.to_string();
    let reason = reason.split(" at line ").next().unwrap_or(&reason);
    json!({
        "type": "json_invalid",
        "loc": ["body", pos],
        "msg": "JSON decode error",
        "input": {},
        "ctx": { "error": reason },
    })
}

/// The Python endpoint's check: `Bearer ` prefix (case-sensitive), then the
/// stripped token must equal the key (compared in constant time, like
/// `hmac.compare_digest`).
fn check_bearer(headers: &HeaderMap, expected: &str) -> Result<(), &'static str> {
    let auth = headers
        .get(header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok());
    let Some(token) = auth.and_then(|a| a.strip_prefix("Bearer ")) else {
        return Err("Missing or invalid Bearer token");
    };
    if constant_time_eq(py_strip(token).as_bytes(), expected.as_bytes()) {
        Ok(())
    } else {
        Err("Unauthorized: invalid API key")
    }
}

/// Compares without an early exit, so response time does not leak how much of
/// the key matched. (The length still leaks, as with most such checks.)
fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    a.len() == b.len() && a.iter().zip(b).fold(0u8, |acc, (x, y)| acc | (x ^ y)) == 0
}

// --- Responses -------------------------------------------------------------------

fn json_response<T: serde::Serialize + ?Sized>(status: StatusCode, body: &T) -> Response {
    Response::builder()
        .status(status)
        .header(header::CONTENT_TYPE, "application/json")
        .body(Body::from(pyjson::compact(body)))
        .expect("static response parts are valid")
}

fn detail_response(status: StatusCode, detail: &str) -> Response {
    json_response(status, &json!({ "detail": detail }))
}

fn method_not_allowed(method: &Method, allow: &'static str) -> Response {
    let mut response = detail_response(StatusCode::METHOD_NOT_ALLOWED, "Method Not Allowed");
    if method == Method::HEAD {
        *response.body_mut() = Body::empty();
    }
    response
        .headers_mut()
        .insert(header::ALLOW, HeaderValue::from_static(allow));
    response
}

fn internal_error() -> Response {
    Response::builder()
        .status(StatusCode::INTERNAL_SERVER_ERROR)
        .header(header::CONTENT_TYPE, "text/plain; charset=utf-8")
        .body(Body::from("Internal Server Error"))
        .expect("static response parts are valid")
}

// --- CORS (Starlette's CORSMiddleware with allow_methods/headers = ["*"]) -----------

struct Cors {
    origins: Vec<String>,
    allow_all_origins: bool,
    allow_credentials: bool,
}

impl Cors {
    /// Credentials are enabled unless the list is exactly `["*"]`, as in `server.py`.
    fn new(origins: Vec<String>) -> Self {
        let wildcard = origins.len() == 1 && origins[0] == "*";
        Self {
            allow_all_origins: origins.iter().any(|o| o == "*"),
            allow_credentials: !wildcard,
            origins,
        }
    }

    fn is_allowed(&self, origin: &str) -> bool {
        self.allow_all_origins || self.origins.iter().any(|o| o == origin)
    }

    fn preflight(&self, origin: &str, request: &HeaderMap) -> Response {
        let explicit_origin = !self.allow_all_origins || self.allow_credentials;
        let mut headers: Vec<(&str, String)> = Vec::new();
        if explicit_origin {
            headers.push(("vary", "Origin".into()));
        } else {
            headers.push(("access-control-allow-origin", "*".into()));
        }
        headers.push(("access-control-allow-methods", ALL_METHODS.join(", ")));
        headers.push(("access-control-max-age", "600".into()));
        if self.allow_credentials {
            headers.push(("access-control-allow-credentials", "true".into()));
        }

        let mut failures = Vec::new();
        if self.is_allowed(origin) {
            if explicit_origin {
                headers.push(("access-control-allow-origin", origin.to_string()));
            }
        } else {
            failures.push("origin");
        }
        let requested_method = header_str(request, "access-control-request-method").unwrap_or("");
        if !ALL_METHODS.contains(&requested_method) {
            failures.push("method");
        }
        if let Some(requested) = header_str(request, "access-control-request-headers") {
            headers.push(("access-control-allow-headers", requested.to_string()));
        }
        if request.contains_key("access-control-request-private-network") {
            failures.push("private-network");
        }

        let (status, text) = if failures.is_empty() {
            (StatusCode::OK, "OK".to_string())
        } else {
            (
                StatusCode::BAD_REQUEST,
                format!("Disallowed CORS {}", failures.join(", ")),
            )
        };
        let mut builder = Response::builder()
            .status(status)
            .header(header::CONTENT_TYPE, "text/plain; charset=utf-8");
        for (name, value) in headers {
            builder = builder.header(name, value);
        }
        builder
            .body(Body::from(text))
            .unwrap_or_else(|_| internal_error())
    }

    fn decorate(&self, origin: &str, headers: &mut HeaderMap) {
        if self.allow_all_origins {
            headers.insert(
                header::ACCESS_CONTROL_ALLOW_ORIGIN,
                HeaderValue::from_static("*"),
            );
        }
        if self.allow_credentials {
            headers.insert(
                header::ACCESS_CONTROL_ALLOW_CREDENTIALS,
                HeaderValue::from_static("true"),
            );
        }
        let explicit = (self.allow_all_origins && self.allow_credentials)
            || (!self.allow_all_origins && self.is_allowed(origin));
        if explicit && let Ok(value) = HeaderValue::from_str(origin) {
            headers.insert(header::ACCESS_CONTROL_ALLOW_ORIGIN, value);
            let vary = match headers.get(header::VARY).and_then(|v| v.to_str().ok()) {
                Some(existing) => format!("{existing}, Origin"),
                None => "Origin".to_string(),
            };
            if let Ok(v) = HeaderValue::from_str(&vary) {
                headers.insert(header::VARY, v);
            }
        }
    }
}

async fn cors_layer(State(cors): State<Arc<Cors>>, request: Request, next: Next) -> Response {
    let Some(origin) = header_str(request.headers(), "origin").map(String::from) else {
        return next.run(request).await;
    };
    if request.method() == Method::OPTIONS
        && request
            .headers()
            .contains_key("access-control-request-method")
    {
        return cors.preflight(&origin, request.headers());
    }
    let mut response = next.run(request).await;
    cors.decorate(&origin, response.headers_mut());
    response
}

fn header_str<'a>(headers: &'a HeaderMap, name: &str) -> Option<&'a str> {
    headers.get(name).and_then(|v| v.to_str().ok())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn origins_parse_like_python() {
        assert_eq!(parse_origins("*"), ["*"]);
        assert_eq!(parse_origins(" a , ,b,"), ["a", "b"]);
        assert!(parse_origins("").is_empty());
    }

    #[test]
    fn json_content_types() {
        let with = |ct: &str| {
            let mut h = HeaderMap::new();
            h.insert(header::CONTENT_TYPE, HeaderValue::from_str(ct).unwrap());
            is_json_content_type(&h)
        };
        assert!(with("application/json"));
        assert!(with("Application/JSON; charset=utf-8"));
        assert!(with("application/vnd.von+json"));
        assert!(!with("text/plain"));
        assert!(!with("application/jsonx"));
        assert!(!is_json_content_type(&HeaderMap::new()));
    }
}
