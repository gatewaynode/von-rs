//! api.rs, patterns.rs and presets.rs against a fake `Decider` (no weights), plus
//! ports of Python's model-backed tests (`tests/test_patterns_presets.py`,
//! `tests/test_client.py` local mode), which are `#[ignore]`d and need `VON_WEIGHTS`.

use std::sync::Mutex;

use indexmap::IndexMap;
use serde_json::{Value, json};
use von::patterns::{Routed, composite_score, confidence_gate, route, two_stage_choice};
use von::presets::{email_preset, moderation_preset, security_preset, triage_preset};
use von::{
    Answer, Choice, ChoiceAnswer, Decider, Noul, NoulAnswer, Question, Score, ScoreAnswer,
    SystemOneResponse, Usage, VonError, decide, judge, rate,
};

type Call = (Value, IndexMap<String, Question>, String);

/// Answers each call with the next canned response and records what it was asked.
struct Fake {
    responses: Mutex<Vec<SystemOneResponse>>,
    calls: Mutex<Vec<Call>>,
}

impl Fake {
    fn new(answers: Vec<Vec<(&str, Answer)>>) -> Self {
        let responses = answers
            .into_iter()
            .rev()
            .map(|a| SystemOneResponse {
                model: "fake".into(),
                answers: a.into_iter().map(|(k, v)| (k.to_string(), v)).collect(),
                usage: Usage::default(),
            })
            .collect();
        Fake {
            responses: Mutex::new(responses),
            calls: Mutex::new(Vec::new()),
        }
    }

    fn calls(&self) -> Vec<Call> {
        self.calls.lock().unwrap().clone()
    }
}

impl Decider for Fake {
    fn system_one(
        &self,
        state: &Value,
        questions: &IndexMap<String, Question>,
        model: &str,
    ) -> von::Result<SystemOneResponse> {
        self.calls
            .lock()
            .unwrap()
            .push((state.clone(), questions.clone(), model.to_string()));
        Ok(self
            .responses
            .lock()
            .unwrap()
            .pop()
            .expect("no canned response left"))
    }
}

fn noul(p: f64) -> Answer {
    Answer::Noul(NoulAnswer { noul: p })
}

fn choice(id: &str, confidence: f64) -> Answer {
    Answer::Choice(ChoiceAnswer {
        choice: id.into(),
        probabilities: IndexMap::new(),
        confidence,
    })
}

fn score(value: f64, levels: usize) -> Answer {
    Answer::Score(ScoreAnswer {
        score: value,
        confidence: 0.5,
        legend: (0..levels)
            .map(|i| (i.to_string(), format!("L{i}")))
            .collect(),
        probabilities: IndexMap::new(),
    })
}

fn state() -> Value {
    json!("state")
}

// --- api.rs ------------------------------------------------------------------

#[test]
fn decide_sends_python_defaults() {
    let fake = Fake::new(vec![vec![("decision", choice("b", 0.4))]]);
    let answer = decide(&fake, &state(), ["a", "b"], None, None).unwrap();
    assert_eq!(answer.choice, "b");

    let (_, questions, model) = &fake.calls()[0];
    assert_eq!(model, "von-latest");
    let Question::Choice(q) = &questions["decision"] else {
        panic!("expected a Choice")
    };
    assert_eq!(q.instructions, "Which option best describes the state?");
    assert_eq!(q.criteria.keys().collect::<Vec<_>>(), ["a", "b"]);
    assert!(q.criteria.values().all(Option::is_none));
}

#[test]
fn decide_rejects_duplicate_choices_before_asking() {
    let fake = Fake::new(vec![]);
    let err = decide(&fake, &state(), ["a", "a"], None, None).unwrap_err();
    assert!(matches!(err, VonError::InvalidQuestion(_)), "{err}");
    assert!(fake.calls().is_empty());
}

#[test]
fn judge_and_rate_use_python_question_ids() {
    let fake = Fake::new(vec![
        vec![("judgment", noul(0.83))],
        vec![("rating", score(1.5, 3))],
    ]);
    assert_eq!(
        judge(&fake, &state(), "Is it down?", None, None).unwrap(),
        0.83
    );
    let rated = rate(
        &fake,
        &state(),
        vec!["low".into(), "high".into()],
        None,
        Some("v"),
    )
    .unwrap();
    assert_eq!(rated.score, 1.5);

    let calls = fake.calls();
    assert!(matches!(calls[0].1["judgment"], Question::Noul(_)));
    let Question::Score(q) = &calls[1].1["rating"] else {
        panic!("expected a Score")
    };
    assert_eq!(q.instructions, "Rate where the state falls on this scale:");
    assert_eq!(calls[1].2, "v");
}

#[test]
fn helpers_reject_a_wrong_or_missing_answer() {
    let fake = Fake::new(vec![vec![("judgment", choice("a", 1.0))], vec![]]);
    for _ in 0..2 {
        let err = judge(&fake, &state(), "Is it down?", None, None).unwrap_err();
        assert!(matches!(err, VonError::UnexpectedResponse(_)), "{err}");
    }
}

// --- patterns.rs -------------------------------------------------------------

#[test]
fn confidence_gate_routes_nouls_by_distance_from_half() {
    // Mirrors the upstream fix's test_confidence_gate_routes_nouls_by_distance_from_half.
    let fake = Fake::new(vec![vec![
        ("sure_yes", noul(0.95)),
        ("sure_no", noul(0.05)),
        ("unsure", noul(0.55)),
        ("choice_sure", choice("a", 0.9)),
        ("choice_unsure", choice("a", 0.2)),
    ]]);
    let gated = confidence_gate(&fake, &state(), &IndexMap::new(), 0.8).unwrap();
    assert_eq!(
        gated.automatic.keys().collect::<Vec<_>>(),
        ["sure_yes", "sure_no", "choice_sure"]
    );
    assert_eq!(
        gated.escalate.keys().collect::<Vec<_>>(),
        ["unsure", "choice_unsure"]
    );
    assert_eq!(gated.response.answers.len(), 5);
}

#[test]
fn confidence_gate_validates_threshold_like_python() {
    let fake = Fake::new(vec![]);
    for bad in [-0.1, 1.5, f64::NAN] {
        let err = confidence_gate(&fake, &state(), &IndexMap::new(), bad).unwrap_err();
        assert!(
            err.to_string()
                .starts_with("threshold must be in [0.0, 1.0], got ")
        );
    }
    let err = confidence_gate(&fake, &state(), &IndexMap::new(), 1.5).unwrap_err();
    assert_eq!(err.to_string(), "threshold must be in [0.0, 1.0], got 1.5");
    assert!(fake.calls().is_empty());
}

#[test]
fn route_dispatches_and_falls_back() {
    let q = Choice {
        instructions: "Route".into(),
        criteria: IndexMap::from([("refund".into(), None), ("support".into(), None)]),
    };
    let refund = |_: &ChoiceAnswer| "REFUND";
    let support = |_: &ChoiceAnswer| "SUPPORT";
    let fallback = |a: &ChoiceAnswer| if a.choice == "other" { "OTHER" } else { "LOW" };
    let routes: [(&str, von::patterns::Handler<'_, &str>); 2] =
        [("refund", &refund), ("support", &support)];

    let fake = Fake::new(vec![
        vec![("route_question", choice("refund", 0.9))],
        vec![("route_question", choice("refund", 0.1))],
        vec![("route_question", choice("other", 0.9))],
        vec![("route_question", choice("other", 0.9))],
    ]);
    let go = |default| route(&fake, &state(), &q, &routes, default, 0.5).unwrap();
    assert_eq!(go(Some(&fallback)), Routed::Handled("REFUND"));
    assert_eq!(
        go(Some(&fallback)),
        Routed::Handled("LOW"),
        "below min_confidence"
    );
    assert_eq!(go(Some(&fallback)), Routed::Handled("OTHER"), "no handler");
    match go(None) {
        Routed::Unhandled(a) => assert_eq!(a.choice, "other"),
        other => panic!("expected Unhandled, got {other:?}"),
    }
    assert!(
        fake.calls()
            .iter()
            .all(|c| c.1.contains_key("route_question"))
    );
}

#[test]
fn composite_score_normalizes_weights_and_skips_choices() {
    let answers = || {
        vec![
            ("severity", score(1.5, 4)), // 1.5 / 3 = 0.5
            ("blocking", noul(0.8)),
            ("kind", choice("a", 1.0)),
            ("empty_scale", score(0.0, 0)), // max(1, -1) = 1
        ]
    };
    let fake = Fake::new(vec![answers(), answers(), answers()]);
    let weights = IndexMap::from([("blocking".to_string(), 3.0)]);

    let even = composite_score(&fake, &state(), &IndexMap::new(), None, true).unwrap();
    assert_eq!(even.score, 0.4333); // (0.5 + 0.8 + 0) / 3
    assert_eq!(
        even.breakdown.keys().collect::<Vec<_>>(),
        ["severity", "blocking", "empty_scale"]
    );
    assert_eq!(even.breakdown["severity"].raw, 1.5);
    assert_eq!(even.breakdown["severity"].normalized, 0.5);

    let weighted =
        composite_score(&fake, &state(), &IndexMap::new(), Some(&weights), true).unwrap();
    assert_eq!(weighted.score, 0.58); // (0.5 + 2.4 + 0) / 5
    assert_eq!(weighted.breakdown["blocking"].weight, 3.0);

    let summed = composite_score(&fake, &state(), &IndexMap::new(), None, false).unwrap();
    assert_eq!(summed.score, 1.3);
}

#[test]
fn two_stage_choice_asks_category_then_option() {
    let taxonomy: IndexMap<String, IndexMap<String, String>> = IndexMap::from([
        (
            "db".to_string(),
            IndexMap::from([("pg".to_string(), "Postgres".to_string())]),
        ),
        (
            "web".to_string(),
            IndexMap::from([("nginx".to_string(), "Nginx".to_string())]),
        ),
    ]);
    let fake = Fake::new(vec![
        vec![("category", choice("db", 0.9))],
        vec![("option", choice("pg", 0.5))],
    ]);
    let r = two_stage_choice(&fake, &state(), &taxonomy, None, None).unwrap();
    assert_eq!(
        (
            r.category.as_str(),
            r.choice.as_str(),
            r.combined_confidence
        ),
        ("db", "pg", 0.45)
    );

    let calls = fake.calls();
    let Question::Choice(cat) = &calls[0].1["category"] else {
        panic!()
    };
    assert_eq!(
        cat.instructions,
        "Which broad category best matches the state?"
    );
    assert_eq!(
        cat.criteria["web"].as_deref(),
        Some("Category for web operations and topics")
    );
    let Question::Choice(opt) = &calls[1].1["option"] else {
        panic!()
    };
    assert_eq!(
        opt.instructions,
        "Which specific sub-option applies within db?"
    );
    assert_eq!(opt.criteria["pg"].as_deref(), Some("Postgres"));
}

// --- presets.rs (Python's test_presets_structure and explicit-criteria test) ---

#[test]
fn presets_structure() {
    let triage = triage_preset();
    assert!(matches!(triage["intent"], Question::Choice(_)));
    assert!(matches!(triage["is_urgent"], Question::Noul(_)));
    assert!(matches!(triage["frustration"], Question::Score(_)));
    assert!(triage.contains_key("churn_risk"));
    let email = email_preset(None);
    for id in ["destination", "is_spam_or_phishing", "priority"] {
        assert!(email.contains_key(id), "{id}");
    }
    let moderation = moderation_preset();
    assert!(moderation.contains_key("policy_violation") && moderation.contains_key("should_block"));
    let security = security_preset();
    assert!(security.contains_key("event_type") && security.contains_key("is_threat"));
}

#[test]
fn preset_nouls_carry_explicit_criteria() {
    for preset in [
        triage_preset(),
        email_preset(None),
        moderation_preset(),
        security_preset(),
    ] {
        for (id, q) in &preset {
            if let Question::Noul(Noul { criteria, .. }) = q {
                let c = criteria.as_ref().unwrap_or_else(|| panic!("{id}"));
                assert!(
                    c.get("true").is_some_and(|s| !s.is_empty())
                        && c.get("false").is_some_and(|s| !s.is_empty()),
                    "{id}"
                );
            }
        }
    }
}

// --- Model-backed ports of the Python tests ------------------------------------

fn load() -> von::Von {
    let dir = std::env::var("VON_WEIGHTS").expect("set VON_WEIGHTS");
    von::Von::load(von::LoadOptions {
        checkpoint_dir: Some(dir.into()),
        ..Default::default()
    })
    .unwrap()
}

fn questions<const N: usize>(qs: [(&str, Question); N]) -> IndexMap<String, Question> {
    qs.into_iter().map(|(k, q)| (k.to_string(), q)).collect()
}

fn noul_q(instructions: &str, criteria: Option<[(&str, &str); 2]>) -> Question {
    Noul {
        instructions: instructions.into(),
        criteria: criteria.map(|c| {
            c.iter()
                .map(|(k, v)| (k.to_string(), v.to_string()))
                .collect()
        }),
    }
    .into()
}

fn choice_q(instructions: &str, criteria: &[(&str, &str)]) -> Choice {
    Choice {
        instructions: instructions.into(),
        criteria: criteria
            .iter()
            .map(|(k, v)| (k.to_string(), Some(v.to_string())))
            .collect(),
    }
}

/// Python: test_patterns_route, test_patterns_confidence_gate, test_patterns_composite_score.
#[test]
#[ignore = "needs model weights (VON_WEIGHTS)"]
fn python_pattern_tests_on_the_model() {
    let von = load();

    let q = choice_q(
        "Route customer request",
        &[
            ("refund", "Customer asks for refund or payment reversal"),
            ("support", "Customer asks for technical support"),
        ],
    );
    let dispatched = Mutex::new(Vec::new());
    let refund = |_: &ChoiceAnswer| {
        dispatched.lock().unwrap().push("refund_handled");
        "REFUND_PROCESSED"
    };
    let support = |_: &ChoiceAnswer| {
        dispatched.lock().unwrap().push("support_handled");
        "SUPPORT_OPENED"
    };
    let res = route(
        &von,
        &json!("The customer wants an immediate refund for their unused subscription."),
        &q,
        &[("refund", &refund), ("support", &support)],
        None,
        0.0,
    )
    .unwrap();
    assert_eq!(res, Routed::Handled("REFUND_PROCESSED"));
    assert_eq!(*dispatched.lock().unwrap(), ["refund_handled"]);

    let gated = confidence_gate(
        &von,
        &json!("Urgent: database cluster crashed, connection pool completely exhausted."),
        &questions([(
            "is_outage",
            noul_q(
                "Is there an active database outage?",
                Some([
                    ("true", "Database crash, pool exhausted, downtime"),
                    ("false", "Normal operational query, no crash"),
                ]),
            ),
        )]),
        0.1,
    )
    .unwrap();
    assert_eq!(gated.automatic.len() + gated.escalate.len(), 1);

    let scored = composite_score(
        &von,
        &json!("Catastrophic multi-region outage affecting all enterprise payments and databases."),
        &questions([
            (
                "severity",
                Score {
                    instructions: "Rate outage severity".into(),
                    criteria: vec![
                        "Minor".into(),
                        "Moderate".into(),
                        "Critical emergency".into(),
                    ],
                }
                .into(),
            ),
            (
                "blocking",
                noul_q(
                    "Is this blocking?",
                    Some([
                        ("true", "Critical blocking outage"),
                        ("false", "Non-blocking"),
                    ]),
                ),
            ),
        ]),
        None,
        true,
    )
    .unwrap();
    assert!((0.0..=1.0).contains(&scored.score));
    assert!(scored.breakdown.contains_key("severity") && scored.breakdown.contains_key("blocking"));
}

/// Python: test_von_client_local and test_async_von_client_local (local mode is `Von` itself).
#[test]
#[ignore = "needs model weights (VON_WEIGHTS)"]
fn python_client_local_tests_on_the_model() {
    let von = load();
    let res = von::system_one(
        &von,
        &json!("Customer requested cancellation of their monthly plan."),
        &questions([
            (
                "action",
                choice_q(
                    "What does the customer want?",
                    &[
                        ("cancel", "Cancel membership or subscription"),
                        ("upgrade", "Upgrade to a higher tier"),
                        ("support", "Help with usage"),
                    ],
                )
                .into(),
            ),
            ("is_cancel", noul_q("Does the user want to cancel?", None)),
        ]),
        None,
    )
    .unwrap();
    assert_eq!(res.model, von::VON_MODEL_ID);
    assert!(matches!(&res.answers["action"], Answer::Choice(a) if a.choice == "cancel"));
    assert!(matches!(&res.answers["is_cancel"], Answer::Noul(a) if a.noul > 0.5));

    let answer = decide(
        &von,
        &json!("Error: Connection refused on port 5432."),
        IndexMap::from([
            (
                "database".to_string(),
                "Database server or Postgres port 5432".to_string(),
            ),
            ("web".to_string(), "Web server or HTTP port".to_string()),
        ]),
        Some("Which service is failing?"),
        None,
    )
    .unwrap();
    assert_eq!(answer.choice, "database");
}
