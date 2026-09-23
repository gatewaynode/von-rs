//! High-level convenience API (port of `api.py`) and the `Decider` trait.
//!
//! Python's helpers use a process-wide default client. Here every helper takes
//! the `Decider` to ask explicitly: a loaded [`Von`] for in-process inference,
//! or a `VonClient` (feature `client`) for a remote server.

use std::sync::Arc;

use indexmap::IndexMap;
use serde_json::Value;

use crate::engine::Von;
use crate::error::{Result, VonError};
use crate::pyfmt::str_repr;
use crate::types::{
    Answer, Choice, ChoiceAnswer, Noul, Question, Score, ScoreAnswer, ScoreLevel, SystemOneResponse,
};

/// Model id the helpers send. Any alias of the current model is accepted.
pub const DEFAULT_MODEL: &str = "von-latest";
pub const DECIDE_INSTRUCTIONS: &str = "Which option best describes the state?";
pub const RATE_INSTRUCTIONS: &str = "Rate where the state falls on this scale:";

/// Anything that can answer a System One request. Implemented by the
/// in-process engine and the remote HTTP client, so patterns work with both.
///
/// Synchronous, like Python's api. (An async variant is planned.)
pub trait Decider: Send + Sync {
    fn system_one(
        &self,
        state: &Value,
        questions: &IndexMap<String, Question>,
        model: &str,
    ) -> Result<SystemOneResponse>;
}

/// The model id is accepted but never reflected: Von serves one model and
/// stamps every response with `VON_MODEL_ID`, as `VonEngine.evaluate` does.
impl Decider for Von {
    fn system_one(
        &self,
        state: &Value,
        questions: &IndexMap<String, Question>,
        _model: &str,
    ) -> Result<SystemOneResponse> {
        self.evaluate(state, questions)
    }
}

impl<D: Decider + ?Sized> Decider for Arc<D> {
    fn system_one(
        &self,
        state: &Value,
        questions: &IndexMap<String, Question>,
        model: &str,
    ) -> Result<SystemOneResponse> {
        (**self).system_one(state, questions, model)
    }
}

impl<D: Decider + ?Sized> Decider for Box<D> {
    fn system_one(
        &self,
        state: &Value,
        questions: &IndexMap<String, Question>,
        model: &str,
    ) -> Result<SystemOneResponse> {
        (**self).system_one(state, questions, model)
    }
}

/// Evaluates `questions` against `state`. `model` defaults to [`DEFAULT_MODEL`].
pub fn system_one<D: Decider + ?Sized>(
    decider: &D,
    state: &Value,
    questions: &IndexMap<String, Question>,
    model: Option<&str>,
) -> Result<SystemOneResponse> {
    decider.system_one(state, questions, model.unwrap_or(DEFAULT_MODEL))
}

/// The options for [`decide`]: bare option ids, or ids with descriptions.
#[derive(Debug, Clone, PartialEq)]
pub enum Choices {
    List(Vec<String>),
    Described(IndexMap<String, Option<String>>),
}

impl From<Vec<String>> for Choices {
    fn from(v: Vec<String>) -> Self {
        Choices::List(v)
    }
}
impl From<Vec<&str>> for Choices {
    fn from(v: Vec<&str>) -> Self {
        Choices::List(v.into_iter().map(String::from).collect())
    }
}
impl From<&[&str]> for Choices {
    fn from(v: &[&str]) -> Self {
        Choices::List(v.iter().map(|s| s.to_string()).collect())
    }
}
impl<const N: usize> From<[&str; N]> for Choices {
    fn from(v: [&str; N]) -> Self {
        Choices::List(v.iter().map(|s| s.to_string()).collect())
    }
}
impl From<IndexMap<String, Option<String>>> for Choices {
    fn from(m: IndexMap<String, Option<String>>) -> Self {
        Choices::Described(m)
    }
}
impl From<IndexMap<String, String>> for Choices {
    fn from(m: IndexMap<String, String>) -> Self {
        Choices::Described(m.into_iter().map(|(k, v)| (k, Some(v))).collect())
    }
}

impl Choices {
    /// Criteria for a Choice question. A list with repeated ids is an error.
    pub fn into_criteria(self) -> Result<IndexMap<String, Option<String>>> {
        match self {
            Choices::Described(m) => Ok(m),
            Choices::List(list) => {
                let criteria: IndexMap<String, Option<String>> =
                    list.iter().map(|c| (c.clone(), None)).collect();
                if criteria.len() != list.len() {
                    let items: Vec<String> = list.iter().map(|c| str_repr(c)).collect();
                    return Err(VonError::InvalidQuestion(format!(
                        "Duplicate choices found in options list: [{}]",
                        items.join(", ")
                    )));
                }
                Ok(criteria)
            }
        }
    }
}

/// Makes a fast discrete decision among `choices`. `instructions` defaults to
/// [`DECIDE_INSTRUCTIONS`].
pub fn decide<D: Decider + ?Sized>(
    decider: &D,
    state: &Value,
    choices: impl Into<Choices>,
    instructions: Option<&str>,
    model: Option<&str>,
) -> Result<ChoiceAnswer> {
    let q = Choice {
        instructions: instructions.unwrap_or(DECIDE_INSTRUCTIONS).to_string(),
        criteria: choices.into().into_criteria()?,
    };
    match ask(decider, state, "decision", q.into(), model)? {
        Answer::Choice(a) => Ok(a),
        other => Err(wrong_type("decision", "choice", &other)),
    }
}

/// Evaluates a yes/no question and returns the probability (0.0 to 1.0) that it holds.
pub fn judge<D: Decider + ?Sized>(
    decider: &D,
    state: &Value,
    instructions: &str,
    criteria: Option<IndexMap<String, String>>,
    model: Option<&str>,
) -> Result<f64> {
    let q = Noul {
        instructions: instructions.to_string(),
        criteria,
    };
    match ask(decider, state, "judgment", q.into(), model)? {
        Answer::Noul(a) => Ok(a.noul),
        other => Err(wrong_type("judgment", "noul", &other)),
    }
}

/// Places the state on an ordered scale (lowest level first). `instructions`
/// defaults to [`RATE_INSTRUCTIONS`].
pub fn rate<D: Decider + ?Sized>(
    decider: &D,
    state: &Value,
    criteria: Vec<ScoreLevel>,
    instructions: Option<&str>,
    model: Option<&str>,
) -> Result<ScoreAnswer> {
    let q = Score {
        instructions: instructions.unwrap_or(RATE_INSTRUCTIONS).to_string(),
        criteria,
    };
    match ask(decider, state, "rating", q.into(), model)? {
        Answer::Score(a) => Ok(a),
        other => Err(wrong_type("rating", "score", &other)),
    }
}

/// Asks a single question under `id` and returns its answer.
fn ask<D: Decider + ?Sized>(
    decider: &D,
    state: &Value,
    id: &str,
    question: Question,
    model: Option<&str>,
) -> Result<Answer> {
    let questions = IndexMap::from([(id.to_string(), question)]);
    let mut resp = system_one(decider, state, &questions, model)?;
    take_answer(&mut resp, id)
}

/// Removes the answer for `id` from a response, or reports that it is missing.
pub(crate) fn take_answer(resp: &mut SystemOneResponse, id: &str) -> Result<Answer> {
    resp.answers
        .shift_remove(id)
        .ok_or_else(|| VonError::UnexpectedResponse(format!("no answer for question '{id}'")))
}

pub(crate) fn wrong_type(id: &str, want: &str, got: &Answer) -> VonError {
    let got = match got {
        Answer::Noul(_) => "noul",
        Answer::Choice(_) => "choice",
        Answer::Score(_) => "score",
    };
    VonError::UnexpectedResponse(format!(
        "question '{id}' was answered as {got}, expected {want}"
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn list_choices_become_undescribed_criteria() {
        let c = Choices::from(["a", "b"]).into_criteria().unwrap();
        assert_eq!(
            c,
            IndexMap::from([("a".to_string(), None), ("b".to_string(), None)])
        );
    }

    #[test]
    fn duplicate_choices_error_like_python() {
        // Python: ValueError("Duplicate choices found in options list: ['a', 'b', 'a']")
        let err = Choices::from(["a", "b", "a"]).into_criteria().unwrap_err();
        assert_eq!(
            err.to_string(),
            "Duplicate choices found in options list: ['a', 'b', 'a']"
        );
    }
}
