//! Composable decision patterns built on a [`Decider`] (port of `patterns.py`).
//!
//! Python's `backend=` argument (switch model version) has no counterpart: Von
//! serves one model. Every pattern sends [`DEFAULT_MODEL`].

use indexmap::IndexMap;
use serde::Serialize;
use serde_json::Value;

use crate::api::{DEFAULT_MODEL, Decider, take_answer, wrong_type};
use crate::error::{Result, VonError};
use crate::pyfmt::{float_repr, round, str_repr};
use crate::types::{Answer, Choice, ChoiceAnswer, Question, SystemOneResponse};

pub const CATEGORY_INSTRUCTIONS: &str = "Which broad category best matches the state?";
pub const OPTION_INSTRUCTIONS: &str = "Which specific sub-option applies within {category}?";

/// Result of [`confidence_gate`].
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Gated {
    /// Answers at or above the threshold.
    pub automatic: IndexMap<String, Answer>,
    /// Answers below the threshold, for a human to review.
    pub escalate: IndexMap<String, Answer>,
    pub response: SystemOneResponse,
}

/// Selective automation: splits answers into automatic vs escalate by
/// confidence. Choice and Score answers use their `confidence`; Noul answers
/// use the distance from uncertainty, `|p - 0.5| * 2`.
pub fn confidence_gate<D: Decider + ?Sized>(
    decider: &D,
    state: &Value,
    questions: &IndexMap<String, Question>,
    threshold: f64,
) -> Result<Gated> {
    if !(0.0..=1.0).contains(&threshold) {
        return Err(VonError::InvalidQuestion(format!(
            "threshold must be in [0.0, 1.0], got {}",
            float_repr(threshold)
        )));
    }
    let response = decider.system_one(state, questions, DEFAULT_MODEL)?;
    let (mut automatic, mut escalate) = (IndexMap::new(), IndexMap::new());
    for (id, answer) in &response.answers {
        let confidence = match answer {
            Answer::Noul(a) => (a.noul - 0.5).abs() * 2.0,
            Answer::Choice(a) => a.confidence,
            Answer::Score(a) => a.confidence,
        };
        let bucket = if confidence >= threshold {
            &mut automatic
        } else {
            &mut escalate
        };
        bucket.insert(id.clone(), answer.clone());
    }
    Ok(Gated {
        automatic,
        escalate,
        response,
    })
}

/// A route handler: receives the Choice answer that selected it.
pub type Handler<'a, R> = &'a dyn Fn(&ChoiceAnswer) -> R;

/// Result of [`route`].
#[derive(Debug, Clone, PartialEq)]
pub enum Routed<R> {
    /// A route handler (or the default handler) ran and returned this.
    Handled(R),
    /// No handler applied and there was no default; Python returns the answer itself.
    Unhandled(ChoiceAnswer),
}

/// Runs a Choice decision and dispatches to the handler for the winning option.
/// Falls back to `default` when no handler matches or the confidence is below
/// `min_confidence`. If `routes` repeats an option id, the first entry wins.
pub fn route<D: Decider + ?Sized, R>(
    decider: &D,
    state: &Value,
    question: &Choice,
    routes: &[(&str, Handler<'_, R>)],
    default: Option<Handler<'_, R>>,
    min_confidence: f64,
) -> Result<Routed<R>> {
    let answer = ask_choice(decider, state, "route_question", question.clone())?;
    let handler = routes
        .iter()
        .find(|(id, _)| *id == answer.choice)
        .map(|(_, h)| *h);
    // Written as Python does, so a NaN confidence still reaches the handler.
    let below_minimum = answer.confidence < min_confidence;
    Ok(match handler {
        Some(h) if !below_minimum => Routed::Handled(h(&answer)),
        _ => match default {
            Some(d) => Routed::Handled(d(&answer)),
            None => Routed::Unhandled(answer),
        },
    })
}

/// One question's share of a [`Composite`] score.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Contribution {
    /// The answer's own value: the Score level or the Noul probability.
    pub raw: f64,
    /// `raw` mapped to [0, 1], rounded to 4 places.
    pub normalized: f64,
    pub weight: f64,
}

/// Result of [`composite_score`].
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Composite {
    /// Weighted mean (or weighted sum when not normalizing), rounded to 4 places.
    pub score: f64,
    pub breakdown: IndexMap<String, Contribution>,
    pub response: SystemOneResponse,
}

/// Combines Score and Noul answers into one index in [0, 1]. Score answers are
/// divided by their top level index, Noul answers use `P(true)`, and Choice
/// answers are left out. Weights default to 1.0 per question.
pub fn composite_score<D: Decider + ?Sized>(
    decider: &D,
    state: &Value,
    questions: &IndexMap<String, Question>,
    weights: Option<&IndexMap<String, f64>>,
    normalize: bool,
) -> Result<Composite> {
    let response = decider.system_one(state, questions, DEFAULT_MODEL)?;
    let (mut weighted_sum, mut total_weight) = (0.0, 0.0);
    let mut breakdown = IndexMap::new();
    for (id, answer) in &response.answers {
        let (raw, normalized) = match answer {
            Answer::Score(a) => {
                let max_level = a.legend.len().saturating_sub(1).max(1);
                (a.score, a.score / max_level as f64)
            }
            Answer::Noul(a) => (a.noul, a.noul),
            Answer::Choice(_) => continue,
        };
        let weight = weights.and_then(|w| w.get(id)).copied().unwrap_or(1.0);
        weighted_sum += normalized * weight;
        total_weight += weight;
        breakdown.insert(
            id.clone(),
            Contribution {
                raw,
                normalized: round(normalized, 4),
                weight,
            },
        );
    }
    let score = if normalize && total_weight > 0.0 {
        weighted_sum / total_weight
    } else {
        weighted_sum
    };
    Ok(Composite {
        score: round(score, 4),
        breakdown,
        response,
    })
}

/// Result of [`two_stage_choice`].
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct TwoStage {
    pub category: String,
    pub category_confidence: f64,
    pub choice: String,
    pub choice_confidence: f64,
    /// Product of the two confidences, rounded to 4 places.
    pub combined_confidence: f64,
}

/// Hierarchical routing for large taxonomies: pick a category, then an option
/// within it. `instructions_option` may contain `{category}` (and `{{`/`}}` for
/// literal braces), as with Python's `str.format`.
pub fn two_stage_choice<D: Decider + ?Sized>(
    decider: &D,
    state: &Value,
    taxonomy: &IndexMap<String, IndexMap<String, String>>,
    instructions_category: Option<&str>,
    instructions_option: Option<&str>,
) -> Result<TwoStage> {
    let template = instructions_option.unwrap_or(OPTION_INSTRUCTIONS);
    let category_q = Choice {
        instructions: instructions_category
            .unwrap_or(CATEGORY_INSTRUCTIONS)
            .to_string(),
        criteria: taxonomy
            .keys()
            .map(|c| {
                (
                    c.clone(),
                    Some(format!("Category for {c} operations and topics")),
                )
            })
            .collect(),
    };
    let category = ask_choice(decider, state, "category", category_q)?;

    let option_q = Choice {
        instructions: format_category(template, &category.choice)?,
        criteria: taxonomy
            .get(&category.choice)
            .map(|opts| {
                opts.iter()
                    .map(|(k, v)| (k.clone(), Some(v.clone())))
                    .collect()
            })
            .unwrap_or_default(),
    };
    let option = ask_choice(decider, state, "option", option_q)?;

    Ok(TwoStage {
        combined_confidence: round(category.confidence * option.confidence, 4),
        category: category.choice,
        category_confidence: category.confidence,
        choice: option.choice,
        choice_confidence: option.confidence,
    })
}

fn ask_choice<D: Decider + ?Sized>(
    decider: &D,
    state: &Value,
    id: &str,
    question: Choice,
) -> Result<ChoiceAnswer> {
    let questions = IndexMap::from([(id.to_string(), Question::from(question))]);
    let mut response = decider.system_one(state, &questions, DEFAULT_MODEL)?;
    match take_answer(&mut response, id)? {
        Answer::Choice(a) => Ok(a),
        other => Err(wrong_type(id, "choice", &other)),
    }
}

/// `template.format(category=category)` for the subset of `str.format` the
/// option template needs: `{category}`, `{{` and `}}`. Anything else is an error,
/// as an unknown field is in Python.
fn format_category(template: &str, category: &str) -> Result<String> {
    let mut out = String::with_capacity(template.len() + category.len());
    let mut rest = template;
    while let Some(i) = rest.find(['{', '}']) {
        out.push_str(&rest[..i]);
        let tail = &rest[i..];
        if let Some(r) = tail
            .strip_prefix("{{")
            .map(|r| ('{', r))
            .or_else(|| tail.strip_prefix("}}").map(|r| ('}', r)))
        {
            out.push(r.0);
            rest = r.1;
        } else if let Some(r) = tail.strip_prefix("{category}") {
            out.push_str(category);
            rest = r;
        } else {
            return Err(VonError::InvalidQuestion(format!(
                "instructions_option supports only {{category}} placeholders: {}",
                str_repr(template)
            )));
        }
    }
    out.push_str(rest);
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn category_template() {
        assert_eq!(
            format_category(OPTION_INSTRUCTIONS, "db").unwrap(),
            "Which specific sub-option applies within db?"
        );
        assert_eq!(
            format_category("{{x}} {category}}}", "a").unwrap(),
            "{x} a}"
        );
        assert!(format_category("{other}", "a").is_err());
        assert!(format_category("dangling {", "a").is_err());
        assert!(format_category("dangling }", "a").is_err());
    }
}
