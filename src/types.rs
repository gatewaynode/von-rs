//! Decision primitives and the `/v1/systemone` wire format (port of `types.py`).
//!
//! Maps are `IndexMap`s so key order survives a round trip, as it does with
//! Python dicts.

use indexmap::IndexMap;
use serde::de::Error as _;
use serde::{Deserialize, Deserializer, Serialize};
use serde_json::Value;

use crate::state::py_str;

/// A yes/no probability question.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(try_from = "NoulWire")]
pub struct Noul {
    pub instructions: String,
    /// Optional `"true"` / `"false"` descriptions. Without them the backend runs
    /// zero-shot with context-free debiasing.
    pub criteria: Option<IndexMap<String, String>>,
}

/// Pick one option from a fixed set.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Choice {
    pub instructions: String,
    /// Option id → description. A missing description falls back to the id.
    pub criteria: IndexMap<String, Option<String>>,
}

/// A position on an ordered scale.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Score {
    pub instructions: String,
    /// Levels from lowest (index 0) to highest.
    pub criteria: Vec<ScoreLevel>,
}

/// One score level: a plain description, or an object with `what` and
/// optional `examples` (other keys are kept but unused).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ScoreLevel {
    Text(String),
    Detailed(IndexMap<String, Value>),
}

impl From<&str> for ScoreLevel {
    fn from(s: &str) -> Self {
        ScoreLevel::Text(s.to_string())
    }
}
impl From<String> for ScoreLevel {
    fn from(s: String) -> Self {
        ScoreLevel::Text(s)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum Question {
    Noul(Noul),
    Choice(Choice),
    Score(Score),
}

impl Question {
    pub fn instructions(&self) -> &str {
        match self {
            Question::Noul(q) => &q.instructions,
            Question::Choice(q) => &q.instructions,
            Question::Score(q) => &q.instructions,
        }
    }
}

impl From<Noul> for Question {
    fn from(q: Noul) -> Self {
        Question::Noul(q)
    }
}
impl From<Choice> for Question {
    fn from(q: Choice) -> Self {
        Question::Choice(q)
    }
}
impl From<Score> for Question {
    fn from(q: Score) -> Self {
        Question::Score(q)
    }
}

/// A missing `type` means `choice`, as in `OptionMarkerBackend.evaluate`.
impl<'de> Deserialize<'de> for Question {
    fn deserialize<D: Deserializer<'de>>(de: D) -> Result<Self, D::Error> {
        let v = Value::deserialize(de)?;
        if !v.is_object() {
            return Err(D::Error::custom("each question must be a JSON object"));
        }
        let kind = match v.get("type") {
            None => "choice".to_string(),
            Some(Value::String(s)) => s.clone(),
            Some(other) => py_str(other),
        };
        match kind.as_str() {
            "choice" => serde_json::from_value(v).map(Question::Choice),
            "noul" => serde_json::from_value(v).map(Question::Noul),
            "score" => serde_json::from_value(v).map(Question::Score),
            _ => return Err(D::Error::custom(format!("Unknown question type '{kind}'"))),
        }
        .map_err(D::Error::custom)
    }
}

/// Noul as accepted on the wire, including the deprecated `pos_criteria` /
/// `neg_criteria` spellings that the Python presets once used. They are folded
/// into `criteria`, matching the upstream fix.
#[derive(Deserialize)]
struct NoulWire {
    instructions: String,
    #[serde(default)]
    criteria: Option<IndexMap<String, String>>,
    #[serde(default)]
    pos_criteria: Option<Value>,
    #[serde(default)]
    neg_criteria: Option<Value>,
}

impl TryFrom<NoulWire> for Noul {
    type Error = String;

    fn try_from(w: NoulWire) -> Result<Self, String> {
        let legacy = [
            ("pos_criteria", "true", w.pos_criteria),
            ("neg_criteria", "false", w.neg_criteria),
        ];
        if legacy.iter().all(|(_, _, v)| v.is_none()) {
            return Ok(Noul {
                instructions: w.instructions,
                criteria: w.criteria,
            });
        }
        let mut criteria = w.criteria.unwrap_or_default();
        for (name, key, value) in legacy {
            let Some(value) = value else { continue };
            if criteria.contains_key(key) {
                return Err(format!(
                    "Noul got both '{name}' and criteria['{key}']; use criteria only."
                ));
            }
            match value {
                Value::Null => {}
                Value::String(s) if s.is_empty() => {}
                Value::String(s) => {
                    criteria.insert(key.to_string(), s);
                }
                other => return Err(format!("Noul '{name}' must be a string, got {other}")),
            }
        }
        tracing::warn!(
            "Noul pos_criteria/neg_criteria are deprecated; use criteria={{'true': ..., 'false': ...}}."
        );
        Ok(Noul {
            instructions: w.instructions,
            criteria: (!criteria.is_empty()).then_some(criteria),
        })
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NoulAnswer {
    /// Probability in [0, 1] that the condition holds.
    pub noul: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ChoiceAnswer {
    pub choice: String,
    pub probabilities: IndexMap<String, f64>,
    /// How far the top option sits above chance: `(n·p_max − 1)/(n − 1)`, in [0, 1].
    pub confidence: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ScoreAnswer {
    /// Probability-weighted expected level (0-indexed).
    pub score: f64,
    /// Same measure as [`ChoiceAnswer::confidence`], over the levels.
    pub confidence: f64,
    pub legend: IndexMap<String, String>,
    pub probabilities: IndexMap<String, f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum Answer {
    Noul(NoulAnswer),
    Choice(ChoiceAnswer),
    Score(ScoreAnswer),
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Usage {
    #[serde(default)]
    pub input_tokens: u64,
    #[serde(default)]
    pub output_tokens: u64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SystemOneResponse {
    pub model: String,
    pub answers: IndexMap<String, Answer>,
    pub usage: Usage,
}

impl SystemOneResponse {
    pub fn get(&self, question_id: &str) -> Option<&Answer> {
        self.answers.get(question_id)
    }
}
