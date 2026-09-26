"""Protocol-parity check, client side: drives a running `von serve` with the unmodified Python SDK.

1. Every golden request (tests/fixtures/golden/v1_2.json) goes through
   `VonClient(local=False)`; answers must match the Python engine's recorded
   responses within the parity tolerances (probabilities 2e-3, score 1e-2).
2. Ports of the Python server/client tests (test_server.py, test_client.py) run
   against the Rust server over real HTTP, including Bearer auth.

Usage: python cross_sdk_client.py BASE_URL API_KEY   (run by cross_sdk_check.sh)
"""

import json
import os
import sys

import httpx

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.environ.get("VON_PY_SRC") or os.path.join(HERE, "..", "..", "src"))

from von.client import VonClient  # noqa: E402
from von.types import choice, noul  # noqa: E402

PROB_TOL, SCORE_TOL = 2e-3, 1e-2


def compare(case_id, got, want, failures):
    if got["model"] != want["model"] or got["usage"] != want["usage"]:
        failures.append(f"{case_id}: model/usage {got['model']} {got['usage']} vs {want['model']} {want['usage']}")
    if list(got["answers"]) != list(want["answers"]):
        failures.append(f"{case_id}: answer ids {list(got['answers'])} vs {list(want['answers'])}")
        return 0.0
    worst = 0.0
    for qid, w in want["answers"].items():
        g = got["answers"][qid]
        if g["type"] != w["type"]:
            failures.append(f"{case_id}/{qid}: type {g['type']} vs {w['type']}")
            continue
        if w["type"] == "noul":
            d = abs(g["noul"] - w["noul"])
            worst = max(worst, d)
            if d > PROB_TOL:
                failures.append(f"{case_id}/{qid}: noul {g['noul']} vs {w['noul']}")
            continue
        if list(g["probabilities"]) != list(w["probabilities"]):
            failures.append(f"{case_id}/{qid}: probability keys differ")
            continue
        for k, p in w["probabilities"].items():
            d = abs(g["probabilities"][k] - p)
            worst = max(worst, d)
            if d > PROB_TOL:
                failures.append(f"{case_id}/{qid}: P({k}) {g['probabilities'][k]} vs {p}")
        if w["type"] == "choice" and g["choice"] != w["choice"]:
            failures.append(f"{case_id}/{qid}: choice {g['choice']} vs {w['choice']}")
        if w["type"] == "score":
            if abs(g["score"] - w["score"]) > SCORE_TOL or g["legend"] != w["legend"]:
                failures.append(f"{case_id}/{qid}: score {g['score']} vs {w['score']}")
    return worst


def main():
    base, key = sys.argv[1], sys.argv[2]
    failures = []

    # 1. Golden parity over HTTP through the Python SDK.
    golden = json.load(open(os.path.join(HERE, "..", "tests", "fixtures", "golden", "v1_2.json")))
    client = VonClient(base_url=base, api_key=key, local=False)
    worst = 0.0
    for case in golden["cases"]:
        resp = client.system_one(state=case["state"], questions=case["questions"])
        worst = max(worst, compare(case["id"], resp.model_dump(), case["response"], failures))
    print(f"golden over HTTP: {len(golden['cases'])} requests, worst probability delta {worst:.2e}")

    # 2. test_server.py, over real HTTP.
    h = {"Authorization": f"Bearer {key}"}
    r = httpx.get(f"{base}/health")
    assert r.status_code == 200 and r.json()["status"] == "ok" and "version" in r.json(), r.text
    ids = [m["id"] for m in httpx.get(f"{base}/v1/models").json()["data"]]
    assert "von-latest" in ids and "von-1.2.0" in ids and "von-1.1.0" in ids, ids
    payload = {
        "model": "von-latest",
        "state": "The user clicked the checkout button but received a credit card decline error.",
        "questions": {
            "error_type": {"type": "choice", "instructions": "What type of error occurred?",
                           "criteria": {"payment_error": "Payment or card transaction failure",
                                        "ui_bug": "Layout or display bug"}},
            "is_payment": {"type": "noul", "instructions": "Is this a payment failure?"},
        },
    }
    r = httpx.post(f"{base}/v1/systemone", json=payload, headers=h)
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["model"] == "von-1.2.0" and data["answers"]["error_type"]["choice"] == "payment_error", data
    assert data["answers"]["is_payment"]["noul"] > 0.5, data
    assert httpx.post(f"{base}/v1/systemone", json=payload).status_code == 401
    assert httpx.post(f"{base}/v1/systemone", json=payload,
                      headers={"Authorization": "Bearer wrong"}).status_code == 401

    # 3. test_client.py's cases, in remote mode.
    res = client.system_one(
        state="Customer requested cancellation of their monthly plan.",
        questions={
            "action": choice("What does the customer want?", {
                "cancel": "Cancel membership or subscription",
                "upgrade": "Upgrade to a higher tier",
                "support": "Help with usage",
            }),
            "is_cancel": noul("Does the user want to cancel?"),
        },
    )
    assert res.model == "von-1.2.0" and res.answers["action"].choice == "cancel", res
    assert res.answers["is_cancel"].noul > 0.5, res
    res = client.system_one(
        state="Error: Connection refused on port 5432.",
        questions={"service": choice("Which service is failing?", {
            "database": "Database server or Postgres port 5432",
            "web": "Web server or HTTP port",
        })},
    )
    assert res.answers["service"].choice == "database", res
    print("python SDK: test_server + test_client (remote) ok")

    if failures:
        print("\n".join(failures[:40]))
        sys.exit(1)


if __name__ == "__main__":
    main()
