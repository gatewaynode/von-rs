"""Dump golden fixtures from the Python reference runtime for von-rs parity tests.

Runs the real OptionMarkerBackend on torch CPU fp32 (the ground truth) over
whole /v1/systemone-style requests and records, per request: every forward pass
(the exact packed string, token ids, [MASK] positions, raw logits), the
effective temperature per question, and the final response. A Rust mismatch
can then be pinned to packing, tokenization, the encoder, calibration, or
answer assembly.

The checkpoint is a local dir, normally a Hub snapshot at a pinned commit
(tools/hub_pins.py prints one), so a golden set always names its model. Usage
(from the repo root):
    HF_HOME=.hf-cache uv run python von-rs/tools/export_golden.py \
        --checkpoint-dir "$(HF_HOME=.hf-cache uv run python von-rs/tools/hub_pins.py 1.2)" \
        --out von-rs/tests/fixtures/golden/v1_2.json
(HF_HOME is wherever the Hub cache lives; `just models` prints it. `just fixtures`
records both sets: 1.2 into v1_2.json, 1.1 into v1.json.)
"""

import argparse
import json
import os
import random
import sys

import torch

HERE = os.path.dirname(__file__)
ROOT = os.path.join(HERE, "..", "..")
sys.path.insert(0, os.environ.get("VON_PY_SRC") or os.path.join(ROOT, "src"))

from von.backends.option_marker_backend import OptionMarkerBackend  # noqa: E402

BENCH_FILE = os.path.join(ROOT, "benchmarks", "data", "authored144.jsonl")
N_BENCH = 60

POLICY = (
    "Section {i}. Change management policy: any modification to production systems, including "
    "configuration edits, schema migrations, and access-control changes, requires an approved "
    "change ticket reviewed by a second engineer. Read-only inspection such as listing files, "
    "viewing dashboards, or exporting metrics does not require a ticket. "
)


def q_choice(instructions, criteria):
    return {"type": "choice", "instructions": instructions, "criteria": criteria}


def q_noul(instructions, criteria=None):
    q = {"type": "noul", "instructions": instructions}
    if criteria is not None:
        q["criteria"] = criteria
    return q


def q_score(instructions, levels):
    return {"type": "score", "instructions": instructions, "criteria": levels}


QUEUES = {"account_access": "Account access and authentication support.",
          "billing": "Billing and payment support.", "sales": "Sales and product evaluation."}

HANDCRAFTED = [
    ("route-account-access", "Customer asks to reset a forgotten password and says the reset email never arrived.",
     {"q": q_choice("Which queue should handle this request?", QUEUES)}),
    ("policy-ticket", "Policy: production deletion requires an approved change ticket. Request: list the names of files in the production backup; do not modify anything.",
     {"q": q_choice("Does the request require an approved change ticket under the stated policy?", {
         "required": "An approved change ticket is required.", "not_required": "An approved change ticket is not required.",
         "insufficient": "The evidence is insufficient to decide."})}),
    ("keys-only-criteria", "I want my money back for the duplicate charge on my card.",
     {"q": q_choice("Which option best describes the state?", {"refund": None, "bug_report": None, "feature_request": None, "cancellation": None})}),
    ("empty-description-falls-back-to-id", "The app crashes when I tap save.",
     {"q": q_choice("Classify", {"bug_report": "", "refund": "  ", "praise": None})}),
    ("type-omitted-defaults-to-choice", "Please cancel my subscription at the end of the month.",
     {"q": {"instructions": "What does the user want?", "criteria": {"cancel": "End the subscription", "upgrade": "Move to a bigger plan"}}}),
    ("single-option", "Anything at all.", {"q": q_choice("Only one choice", {"only": "The only option"})}),
    ("empty-instructions", "Server returned HTTP 503 for every request since 09:00.",
     {"q": q_choice("", {"outage": "Service is down", "normal": "Service is fine"})}),
    ("whitespace-strip", "  \t padded state with trailing spaces \n  ",
     {"q": q_choice("  Which?  ", {"a": "  first option  ", "b": "\tsecond option\n"})}),
    ("dict-state", {"service": "payments-api", "error_rate": 0.37, "healthy": False, "owner": None,
                    "regions": ["us-east-1", "eu-west-1"], "meta": {"tier": 1, "note": "it's paging"}},
     {"q": q_choice("What is the service status?", {"healthy": "Service operating normally",
                                                     "degraded": "Elevated errors but partially serving", "down": "Service fully unavailable"})}),
    ("list-state", ["login failed", "login failed", "account locked", {"ip": "10.0.0.7", "attempts": 12}],
     {"q": q_noul("Does this look like a brute-force attack?", {"true": "Repeated failed logins against one account", "false": "Normal user activity"})}),
    ("number-state", 97.5, {"q": q_score("How high is this CPU utilisation percentage?", ["Low", "Moderate", "High", "Critical"])}),
    ("null-state", None, {"q": q_noul("Is there any information here?")}),
    ("bool-state", True, {"q": q_choice("What value is this?", {"yes": "Affirmative", "no": "Negative"})}),
    ("unicode", "Le client écrit : « Ma commande n°4521 n'est jamais arrivée 📦 » — il demande un remboursement.",
     {"q": q_choice("What does the customer want?", {"refund": "Money back", "replacement": "Send the item again", "info": "Just a status update"})}),
    ("many-options", "Our Kubernetes pods keep getting OOMKilled after the latest release.",
     {"q": q_choice("Which team owns this?", {
         "frontend": "Web UI", "mobile": "iOS and Android apps", "platform": "Kubernetes, infra, runtime",
         "data": "Warehouse and pipelines", "security": "Auth and vulnerabilities", "billing": "Payments",
         "support": "Customer questions", "legal": "Contracts and compliance", "sales": "Deals",
         "marketing": "Campaigns", "hr": "People", "finance": "Budgets"})}),
    ("noul-explicit", "Urgent: database cluster crashed, connection pool completely exhausted.",
     {"q": q_noul("Is there an active database outage?", {"true": "Database crash, pool exhausted, downtime", "false": "Normal operational query, no crash"})}),
    ("noul-zero-shot", "The nightly backup finished successfully and all checksums matched.", {"q": q_noul("Did the backup fail?")}),
    ("noul-only-true", "Three customers reported double charges this morning.", {"q": q_noul("Is there a billing incident?", {"true": "Multiple customers affected by a billing error"})}),
    ("noul-only-false", "Weekly newsletter: new blog posts and a product webinar.", {"q": q_noul("Is this spam?", {"false": "Legitimate expected email"})}),
    ("noul-empty-criteria-is-zero-shot", "The user asked how to export a CSV.", {"q": q_noul("Is the user angry?", {"true": "", "false": ""})}),
    ("score-plain", "Catastrophic multi-region outage affecting all enterprise payments and databases.",
     {"q": q_score("Rate outage severity", ["Minor", "Moderate", "Major", "Critical emergency"])}),
    ("score-detailed", "This is the third time I've asked. Fix it today or we're cancelling the contract.",
     {"q": q_score("Rate the customer frustration level.", [
         {"what": "Calm", "examples": ["thanks!", "no rush"]}, {"what": "Annoyed"}, "Very frustrated",
         {"what": "Furious", "examples": ["cancelling", "lawyer"], "note": "ignored extra key"}])}),
    ("score-ten-levels", "The patch reduced p99 latency from 900ms to 240ms but error rate rose slightly.",
     {"q": q_score("Rate the overall improvement from 0 (much worse) to 9 (transformative).", [str(i) + " out of 9" for i in range(10)])}),
    ("multi-question", "Hi, I've been charged twice for March and nobody answers my emails. I'm switching providers if this isn't fixed today.",
     {"intent": q_choice("What is the primary customer intent?", {"refund": "Requesting money back", "technical_help": "Reporting a bug",
                                                                 "cancellation": "Requesting account closure"}),
      "is_urgent": q_noul("Does the customer communicate urgency?", {"true": "Urgent, immediate attention needed", "false": "Routine"}),
      "churn": q_noul("Is the customer likely to leave?"),
      "frustration": q_score("Rate the customer frustration level.", ["Calm", "Concerned", "Frustrated", "Furious"])}),
    ("long-600-tokens", "".join(POLICY.format(i=i) for i in range(12)) + "Request: export last week's latency metrics to CSV.",
     {"q": q_choice("Does the request require an approved change ticket?", {"required": "A ticket is required", "not_required": "No ticket is required"})}),
    ("long-3000-tokens", "".join(POLICY.format(i=i) for i in range(60)) + "Request: rotate the database root password in production.",
     {"q": q_choice("Does the request require an approved change ticket?", {"required": "A ticket is required", "not_required": "No ticket is required"}),
      "risk": q_score("Rate the operational risk of the request.", ["Low", "Medium", "High"])}),
]


def bench_cases():
    rows = [json.loads(line) for line in open(BENCH_FILE, encoding="utf-8")]
    rng = random.Random(20260922)
    out = []
    for i, row in enumerate(rng.sample(rows, N_BENCH)):
        opts = row["options"]
        if i % 6 == 5 and len(opts) >= 2:
            q = q_noul(row["question"], {"true": opts[0]["description"], "false": opts[1]["description"]})
        elif i % 6 == 4:
            q = q_score(row["question"], [o["description"] for o in opts])
        else:
            q = q_choice(row["question"], {o["id"]: o["description"] for o in opts})
        out.append((f"bench-{row['id']}", row["state"], {"q": q}))
    return out


class Recorder:
    """Wraps model.forward and _effective_temperature to capture each stage."""

    def __init__(self, backend: OptionMarkerBackend):
        self.passes: list[dict] = []
        self.temps: list[float] = []
        model = backend._get_model()
        orig_forward = model.forward
        orig_pack = model.pack_sequence
        orig_temp = backend._effective_temperature
        pending: list[str] = []

        def pack(state, question, options):
            s = orig_pack(state, question, options)
            pending.append(s)
            return s

        def forward(input_ids, attention_mask, mask_positions, **kwargs):
            logits = orig_forward(
                input_ids=input_ids, attention_mask=attention_mask, mask_positions=mask_positions, **kwargs
            )
            self.passes.append({
                "packed_input": pending.pop(0),
                "token_ids": input_ids[0].tolist(),
                "mask_positions": list(mask_positions[0]),
                "logits": logits[0].tolist(),
            })
            return logits

        def temp(*a, **kw):
            t = orig_temp(*a, **kw)
            self.temps.append(float(t))
            return t

        model.pack_sequence = pack
        model.forward = forward
        backend._effective_temperature = temp


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument(
        "--checkpoint-dir",
        required=True,
        help="local checkpoint (option_marker.pt, config, tokenizer, calibration), "
        "e.g. a Hub cache snapshot from tools/hub_pins.py",
    )
    args = ap.parse_args()
    # The backend falls back to the Hub's moving main branch when the dir has no weights.
    if not os.path.isfile(os.path.join(args.checkpoint_dir, "option_marker.pt")):
        raise SystemExit(f"no option_marker.pt in checkpoint dir {args.checkpoint_dir!r}")

    torch.manual_seed(0)
    backend = OptionMarkerBackend(checkpoint_dir=args.checkpoint_dir, device="cpu")
    rec = Recorder(backend)
    out = {
        # A Hub snapshot dir is named after its commit.
        "checkpoint": os.path.basename(os.path.realpath(args.checkpoint_dir)),
        "device": "cpu",
        "torch": torch.__version__,
        "temperature": backend._default_temp,
        "calibration_map": backend._calib_map,
        "independent_options": getattr(backend, "_independent_options", False),
        "cases": [],
    }
    for case_id, state, questions in HANDCRAFTED + bench_cases():
        rec.passes.clear()
        rec.temps.clear()
        resp = backend.evaluate(state, questions)
        out["cases"].append({
            "id": case_id,
            "state": state,
            "questions": questions,
            "passes": list(rec.passes),
            "temperatures": list(rec.temps),
            "response": resp.model_dump(),
        })
        tokens = max(len(p["token_ids"]) for p in rec.passes) if rec.passes else 0
        print(f"{case_id:<40} passes={len(rec.passes)} max_tokens={tokens}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False)
    print(f"wrote {args.out} ({len(out['cases'])} cases, {sum(len(c['passes']) for c in out['cases'])} passes)")


if __name__ == "__main__":
    main()
