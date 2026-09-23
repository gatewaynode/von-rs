//! Ready-made question sets for common operational workflows (port of `presets.py`).
//!
//! Noul presets carry explicit `criteria={"true", "false"}`, matching the
//! upstream Python fix (the legacy `pos_criteria`/`neg_criteria` were dropped).

use indexmap::IndexMap;

use crate::types::{Choice, Noul, Question, Score, ScoreLevel};

pub type Preset = IndexMap<String, Question>;

/// Customer support ticket triage and routing.
pub fn triage_preset() -> Preset {
    preset([
        (
            "intent",
            choice(
                "What is the primary customer intent in the message?",
                &[
                    (
                        "refund",
                        "Requesting money back, refund, or duplicate billing reversal",
                    ),
                    (
                        "technical_help",
                        "Reporting a bug, API error, 500 downtime, or integration issue",
                    ),
                    (
                        "billing_question",
                        "Questions about invoices, subscription plans, or payment methods",
                    ),
                    (
                        "cancellation",
                        "Requesting account closure, cancellation, or downgrading",
                    ),
                    (
                        "general_info",
                        "Inquiring about documentation, pricing tiers, or how-to guidance",
                    ),
                ],
            ),
        ),
        (
            "is_urgent",
            noul(
                "Does the customer communicate extreme urgency, critical outage, or impending deadline?",
                "Urgent, production down, emergency, immediate attention needed",
                "Routine question, low priority, general feedback",
            ),
        ),
        (
            "frustration",
            score(
                "Rate the customer frustration level.",
                &[
                    "Calm and polite",
                    "Slightly concerned or asking for status",
                    "Visibly frustrated or annoyed",
                    "Extremely angry, threatening legal action or cancellation",
                ],
            ),
        ),
        (
            "churn_risk",
            noul(
                "Does the message indicate high risk of the customer leaving or churning?",
                "Threatening to switch to competitors, cancel contract, or stop using product",
                "Committed user asking for help, no mention of leaving",
            ),
        ),
    ])
}

/// Inbound email triage and threat filtering. `custom_categories` replaces the
/// default destination teams; `None` or an empty map keeps the defaults, as in Python.
pub fn email_preset(custom_categories: Option<IndexMap<String, String>>) -> Preset {
    let categories = custom_categories
        .filter(|c| !c.is_empty())
        .map(|c| c.into_iter().map(|(k, v)| (k, Some(v))).collect())
        .unwrap_or_else(|| {
            criteria(&[
                (
                    "billing",
                    "Invoices, payments, credit cards, pricing questions",
                ),
                (
                    "engineering",
                    "Bug reports, API failures, stack traces, system outages",
                ),
                (
                    "sales",
                    "Enterprise demos, contract inquiries, volume discounts",
                ),
                (
                    "security",
                    "Phishing reports, suspicious access, vulnerability disclosures",
                ),
                ("general", "General questions or uncategorized inquiries"),
            ])
        });
    preset([
        (
            "destination",
            Choice {
                instructions: "Which internal team should handle this email?".into(),
                criteria: categories,
            }
            .into(),
        ),
        (
            "is_spam_or_phishing",
            noul(
                "Is this email an unsolicited sales pitch, scam, or phishing attempt?",
                "Spam, promotional blast, credential harvesting, phishing",
                "Legitimate user or customer inquiry",
            ),
        ),
        (
            "priority",
            score(
                "What priority level should be assigned to this email?",
                &[
                    "Low: Newsletter, informational, no action required",
                    "Medium: Standard inquiry with 24-48hr SLA",
                    "High: Blocking issue affecting paying customer",
                    "Critical: Security breach, legal threat, or severe production impact",
                ],
            ),
        ),
    ])
}

/// User-generated content and trust & safety moderation.
pub fn moderation_preset() -> Preset {
    preset([
        (
            "policy_violation",
            choice(
                "Does this content violate acceptable use policies?",
                &[
                    (
                        "clean",
                        "Content is safe, constructive, and follows community guidelines",
                    ),
                    (
                        "harassment",
                        "Direct personal attacks, bullying, threats, or hate speech",
                    ),
                    (
                        "spam",
                        "Repetitive links, commercial spam, crypto scams, or bot text",
                    ),
                    (
                        "sensitive",
                        "Explicit adult content, graphic violence, or illegal goods",
                    ),
                ],
            ),
        ),
        (
            "should_block",
            noul(
                "Should this content be immediately blocked from publication?",
                "Clear violation requiring immediate rejection",
                "Safe or borderline content that can be published or reviewed",
            ),
        ),
        (
            "severity",
            score(
                "Rate the severity of the content risk.",
                &[
                    "Safe: Compliant content",
                    "Low: Minor profanity or mild uncivil behavior",
                    "Medium: Aggressive tone, self-promotion, or borderline spam",
                    "High: Severe violation, harassment, or malicious payload",
                ],
            ),
        ),
    ])
}

/// Security event triage and anomaly assessment.
pub fn security_preset() -> Preset {
    preset([
        (
            "event_type",
            choice(
                "Classify the observed security or authentication anomaly.",
                &[
                    (
                        "benign",
                        "Expected user activity, legitimate IP change, or normal login",
                    ),
                    (
                        "credential_stuffing",
                        "Rapid succession of failed logins across multiple accounts",
                    ),
                    (
                        "brute_force",
                        "Repeated failed attempts targeting a single high-value account",
                    ),
                    (
                        "privilege_escalation",
                        "Attempting unauthorized administrative or sudo operations",
                    ),
                    (
                        "data_exfiltration",
                        "Abnormal volume of export requests or bulk database downloads",
                    ),
                ],
            ),
        ),
        (
            "is_threat",
            noul(
                "Does this state represent an active, confirmed malicious security threat?",
                "Active cyber attack, intrusion, or unauthorized compromise",
                "Normal operational glitch, user error, or benign variance",
            ),
        ),
        (
            "severity",
            score(
                "Rate the incident severity.",
                &[
                    "Informational: Logged for audit trail, no action",
                    "Warning: Suspicious variance, rate-limit triggered",
                    "Elevated: Incident responder paged for triage",
                    "Critical: Active breach, immediate token revocation and IP ban",
                ],
            ),
        ),
    ])
}

fn preset<const N: usize>(questions: [(&str, Question); N]) -> Preset {
    questions
        .into_iter()
        .map(|(id, q)| (id.to_string(), q))
        .collect()
}

fn criteria(options: &[(&str, &str)]) -> IndexMap<String, Option<String>> {
    options
        .iter()
        .map(|(k, v)| (k.to_string(), Some(v.to_string())))
        .collect()
}

fn choice(instructions: &str, options: &[(&str, &str)]) -> Question {
    Choice {
        instructions: instructions.into(),
        criteria: criteria(options),
    }
    .into()
}

fn noul(instructions: &str, when_true: &str, when_false: &str) -> Question {
    Noul {
        instructions: instructions.into(),
        criteria: Some(IndexMap::from([
            ("true".to_string(), when_true.to_string()),
            ("false".to_string(), when_false.to_string()),
        ])),
    }
    .into()
}

fn score(instructions: &str, levels: &[&str]) -> Question {
    Score {
        instructions: instructions.into(),
        criteria: levels.iter().map(|l| ScoreLevel::from(*l)).collect(),
    }
    .into()
}
