"""Protocol oracle for the von-rs server and CLI: the Python server's and CLI's exact behaviour.

Runs the real FastAPI app (via TestClient) and the real click CLI (via CliRunner)
with the engine replaced by a fake that returns canned answers and records the
request it received, then writes tests/fixtures/protocol.json. The Rust server
and CLI tests replay every case with an equivalent fake `Decider`, so no model
weights are needed on either side.

Usage (from the repo root):
    VON_PY_SRC=bug-fix-fork-von/src uv run python von-rs/tools/export_protocol.py
"""

import importlib
import json
import os
import sys
import warnings

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.environ.get("VON_PY_SRC") or os.path.join(HERE, "..", "..", "src"))
warnings.simplefilter("ignore")

from click.testing import CliRunner  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

import von.api  # noqa: E402
import von.engine  # noqa: E402
from von.types import Choice, Noul, Score, SystemOneResponse  # noqa: E402

OUT = os.path.join(HERE, "..", "tests", "fixtures", "protocol.json")
CLI_DIR = os.path.join(HERE, "..", "tests", "fixtures", "cli")

CHOICE = {"type": "choice", "choice": "café", "probabilities": {"café": 0.9999, "bar": 0.0001}, "confidence": 0.999}
NOUL = {"type": "noul", "noul": 0.8125}
SCORE = {"type": "score", "score": 1.0, "confidence": 0.25,
         "legend": {"0": "low", "1": "mid", "2": "high"},
         "probabilities": {"0": 0.1, "1": 0.8, "2": 0.1}}


def canned(qtype):
    return {"choice": CHOICE, "noul": NOUL, "score": SCORE}[qtype]


def as_model(q):
    """A raw question dict as the backend reads it (missing `type` = choice), so the
    recording shows what pydantic made of it, e.g. structured instructions as text."""
    if not isinstance(q, dict):
        return q
    return {"choice": Choice, "noul": Noul, "score": Score}[q.get("type", "choice")](**q)


class FakeEngine:
    """Stands in for VonEngine: records the request, then answers or raises."""

    def __init__(self, error=None):
        self.error = error
        self.seen = None

    def evaluate(self, state, questions, model=None):
        dumped = {k: as_model(q).model_dump() for k, q in questions.items()}
        self.seen = {"state": state, "questions": dumped, "model": model}
        if self.error:
            raise ValueError(self.error)
        answers = {k: canned(q.get("type", "choice")) for k, q in dumped.items()}
        return SystemOneResponse(model="von-1.1.0", answers=answers, usage={"input_tokens": 12, "output_tokens": 0})


# --- server ------------------------------------------------------------------

KEEP_HEADERS = ("access-control-", "vary", "content-type", "location", "allow")

VALID = {
    "model": "jev-latest",
    "state": {"ticket": "Überweisung fehlgeschlagen", "n": 3},
    "questions": {
        "kind": {"type": "choice", "instructions": "Which?", "criteria": {"café": "Coffee", "bar": None}},
        "is_payment": {"type": "noul", "instructions": "Payment?"},
        "severity": {"type": "score", "instructions": "How bad?", "criteria": ["low", "mid", "high"]},
    },
}

# TypeSafe allows object and array instructions; Python sends them to the model as JSON text.
STRUCTURED = {
    "model": "jev-latest",
    "state": "refund please",
    "questions": {
        "wants_refund": {"type": "noul", "instructions": {"task": "Does the customer want money back?",
                                                          "context": {"tier": "gold", "amount": 12.5}}},
        "route": {"type": "choice", "instructions": ["Pick a queue", {"z": 1, "a": "caf\u00e9"}],
                  "criteria": {"billing": None, "tech": None}},
        "urgency": {"type": "score", "instructions": {"scale": "1-3"}, "criteria": ["low", "mid", "high"]},
    },
}


def server_cases():
    json_hdr = {"content-type": "application/json"}
    body = lambda v: json.dumps(v)  # noqa: E731
    cases = [
        ("health root", {}, "GET", "/", {}, None, None),
        ("health", {}, "GET", "/health", {}, None, None),
        ("head health", {}, "HEAD", "/health", {}, None, None),
        ("models", {}, "GET", "/v1/models", {}, None, None),
        ("valid", {}, "POST", "/v1/systemone", json_hdr, body(VALID), None),
        ("default model and null state", {}, "POST", "/v1/systemone", json_hdr,
         body({"state": None, "questions": {"q": {"instructions": "Which?", "criteria": {"a": None}}}}), None),
        ("empty questions", {}, "POST", "/v1/systemone", json_hdr,
         body({"state": "x", "questions": {}}), None),
        ("empty questions and bad model", {}, "POST", "/v1/systemone", json_hdr,
         body({"model": 1, "state": "x", "questions": {}}), None),
        ("empty questions before auth", {"key": "secret"}, "POST", "/v1/systemone", json_hdr,
         body({"state": "x", "questions": {}}), None),
        ("structured instructions", {}, "POST", "/v1/systemone", json_hdr, body(STRUCTURED), None),
        ("engine error", {}, "POST", "/v1/systemone", json_hdr, body(VALID), "boom: bad question"),
        ("invalid json", {}, "POST", "/v1/systemone", json_hdr, "{bad", None),
        ("empty body", {}, "POST", "/v1/systemone", json_hdr, "", None),
        ("no content type", {}, "POST", "/v1/systemone", {}, body(VALID), None),
        ("text content type", {}, "POST", "/v1/systemone", {"content-type": "text/plain"}, "hello", None),
        ("json suffix content type", {}, "POST", "/v1/systemone",
         {"content-type": "application/vnd.von+json; charset=utf-8"}, body(VALID), None),
        ("body is a list", {}, "POST", "/v1/systemone", json_hdr, body([1]), None),
        ("several field errors", {}, "POST", "/v1/systemone", json_hdr,
         body({"model": None, "questions": {"a": 1, "b": {}}}), None),
        ("questions not a dict", {}, "POST", "/v1/systemone", json_hdr,
         body({"model": 5, "state": "x", "questions": [1]}), None),
        ("extra fields ignored", {}, "POST", "/v1/systemone", json_hdr,
         body({"state": "x", "questions": {"q": {"type": "noul", "instructions": "Q?"}}, "stream": True}), None),
        ("not found", {}, "GET", "/nope", {}, None, None),
        ("method not allowed", {}, "POST", "/health", {}, None, None),
        ("get on post route", {}, "GET", "/v1/systemone", {}, None, None),
        ("trailing slash redirect", {}, "GET", "/health/?x=1", {}, None, None),
        ("trailing slash redirect post", {}, "POST", "/v1/systemone/", json_hdr, body(VALID), None),
        # Auth (key read per request in Python).
        ("auth missing header", {"key": "secret"}, "POST", "/v1/systemone", json_hdr, body(VALID), None),
        ("auth wrong scheme", {"key": "secret"}, "POST", "/v1/systemone",
         {**json_hdr, "authorization": "Basic secret"}, body(VALID), None),
        ("auth wrong key", {"key": "secret"}, "POST", "/v1/systemone",
         {**json_hdr, "authorization": "Bearer nope"}, body(VALID), None),
        ("auth ok with padding", {"key": "secret"}, "POST", "/v1/systemone",
         {**json_hdr, "authorization": "Bearer  secret "}, body(VALID), None),
        ("auth lowercase scheme", {"key": "secret"}, "POST", "/v1/systemone",
         {**json_hdr, "authorization": "bearer secret"}, body(VALID), None),
        ("validation before auth", {"key": "secret"}, "POST", "/v1/systemone", json_hdr, "{bad", None),
        ("auth not needed for health", {"key": "secret"}, "GET", "/health", {}, None, None),
        # CORS, wildcard (the default).
        ("cors wildcard preflight", {}, "OPTIONS", "/v1/systemone",
         {"origin": "http://a.com", "access-control-request-method": "POST",
          "access-control-request-headers": "content-type,x-foo"}, None, None),
        ("cors wildcard preflight bad method", {}, "OPTIONS", "/v1/systemone",
         {"origin": "http://a.com", "access-control-request-method": "TRACE"}, None, None),
        ("cors wildcard simple", {}, "GET", "/health", {"origin": "http://a.com"}, None, None),
        ("cors wildcard simple with cookie", {}, "GET", "/health",
         {"origin": "http://a.com", "cookie": "a=b"}, None, None),
        ("options without preflight", {}, "OPTIONS", "/health", {"origin": "http://a.com"}, None, None),
        # CORS allowlist (credentials enabled).
        ("cors allowlist preflight", {"cors": " http://a.com, ,http://b.com"}, "OPTIONS", "/v1/systemone",
         {"origin": "http://b.com", "access-control-request-method": "POST",
          "access-control-request-headers": "authorization"}, None, None),
        ("cors allowlist preflight denied", {"cors": "http://a.com,http://b.com"}, "OPTIONS", "/v1/systemone",
         {"origin": "http://evil.com", "access-control-request-method": "POST"}, None, None),
        ("cors allowlist simple", {"cors": "http://a.com,http://b.com"}, "POST", "/v1/systemone",
         {**json_hdr, "origin": "http://a.com"}, body(VALID), None),
        ("cors allowlist simple denied", {"cors": "http://a.com"}, "GET", "/health",
         {"origin": "http://evil.com"}, None, None),
        ("cors star among others", {"cors": "*,http://a.com"}, "OPTIONS", "/health",
         {"origin": "http://z.com", "access-control-request-method": "GET"}, None, None),
        ("cors empty list", {"cors": ""}, "GET", "/health", {"origin": "http://a.com"}, None, None),
    ]
    out = []
    for name, config, method, path, headers, content, error in cases:
        if "cors" in config:
            os.environ["VON_CORS_ORIGINS"] = config["cors"]
        else:
            os.environ.pop("VON_CORS_ORIGINS", None)
        if "key" in config:
            os.environ["VON_API_KEY"] = config["key"]
        else:
            os.environ.pop("VON_API_KEY", None)
        import von.server as srv
        srv = importlib.reload(srv)  # CORS settings are read at import time
        fake = FakeEngine(error)
        srv.VonEngine.get_instance = staticmethod(lambda fake=fake: fake)
        client = TestClient(srv.app, follow_redirects=False)
        r = client.request(method, path, headers=headers,
                           content=None if content is None else content.encode())
        out.append({
            "name": name,
            "config": config,
            "request": {"method": method, "path": path, "headers": headers, "body": content},
            "engine_error": error,
            "engine_saw": fake.seen,
            "response": {
                "status": r.status_code,
                "headers": {k: v for k, v in r.headers.items() if k.startswith(KEEP_HEADERS)},
                "body": r.text,
            },
        })
    os.environ.pop("VON_CORS_ORIGINS", None)
    os.environ.pop("VON_API_KEY", None)
    return out


# --- CLI -----------------------------------------------------------------------

class FakeClient:
    def __init__(self):
        self.engine = FakeEngine()

    def system_one(self, state, questions, model="von-latest"):
        return self.engine.evaluate(state, questions, model)


def cli_cases():
    os.makedirs(CLI_DIR, exist_ok=True)
    files = {
        "request.json": json.dumps({"model": "von-1.1", "state": {"msg": "naïve ☃"}, "questions": VALID["questions"]}),
        "default_model.json": json.dumps({"state": "x", "questions": {"q": {"instructions": "Which?", "criteria": {"a": None}}}}),
        "no_questions.json": json.dumps({"state": "x"}),
        "null_state.json": json.dumps({"state": None, "questions": {}}),
        "bad.json": '{"state": "x",, }',
        "structured.json": json.dumps(STRUCTURED),
    }
    for name, text in files.items():
        with open(os.path.join(CLI_DIR, name), "w", encoding="utf-8") as f:
            f.write(text)

    import von.cli as cli
    cases = [
        ("decide", ["decide", "I was charged twice", "-c", " refund, bug ,,café"]),
        ("decide custom instructions", ["decide", "x", "--choices", "a,b", "-i", "Pick one", "--device", "cpu"]),
        ("decide no choices", ["decide", "x", "-c", " , ,"]),
        ("decide duplicate choices", ["decide", "x", "-c", "a,b,a"]),
        ("decide missing choices", ["decide", "x"]),
        ("judge", ["judge", "Prod is down", "-i", "Is it down?"]),
        ("judge pos only", ["judge", "Prod is down", "-i", "Is it down?", "--pos", "Down"]),
        ("judge pos and neg", ["judge", "x", "-i", "Q?", "--pos", "Down", "--neg", "Up"]),
        ("rate", ["rate", "Slow page", "-l", "low, mid ,high"]),
        ("rate custom instructions", ["rate", "x", "--levels", "a,b", "--instructions", "Scale:"]),
        ("rate one level", ["rate", "x", "-l", "only,"]),
        ("eval", ["eval", "request.json"]),
        ("eval default model", ["eval", "default_model.json"]),
        ("eval missing questions", ["eval", "no_questions.json"]),
        ("eval null state", ["eval", "null_state.json"]),
        ("eval structured instructions", ["eval", "structured.json"]),
        ("eval invalid json", ["eval", "bad.json"]),
        ("eval missing file", ["eval", "missing.json"]),
        ("version", ["--version"]),
    ]
    out = []
    cwd = os.getcwd()
    os.chdir(CLI_DIR)
    try:
        for name, args in cases:
            fake = FakeClient()
            von.api._default_client = fake
            runner = CliRunner()
            r = runner.invoke(cli.main, args, prog_name="von")
            out.append({
                "name": name,
                "args": args,
                "engine_saw": fake.engine.seen,
                "exit": r.exit_code,
                "stdout": r.stdout,
                "stderr": r.stderr,
                "exception": None if r.exception is None or isinstance(r.exception, SystemExit)
                else f"{type(r.exception).__name__}: {r.exception}",
            })
    finally:
        os.chdir(cwd)
        von.api._default_client = None
    return out


def main() -> None:
    fixture = {"canned": {"choice": CHOICE, "noul": NOUL, "score": SCORE},
               "server": server_cases(), "cli": cli_cases()}
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(fixture, f, ensure_ascii=False, indent=1)
    print(f"wrote {OUT}: server={len(fixture['server'])}, cli={len(fixture['cli'])}")


if __name__ == "__main__":
    main()
