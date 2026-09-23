//! Decision evaluation (port of `backends/option_marker_backend.py`).

use indexmap::IndexMap;
use serde_json::Value;

use crate::calibration::{Calibration, softmax_f32};
use crate::error::{Result, VonError};
use crate::model::OptionMarkerModel;
use crate::pyfmt::{py_strip, round};
use crate::state::{format_state, py_str};
use crate::types::{
    Answer, Choice, ChoiceAnswer, Noul, NoulAnswer, Question, Score, ScoreAnswer, ScoreLevel,
    SystemOneResponse, Usage,
};

const NOUL_DEFAULT_TRUE: &str = "Yes, condition holds true.";
const NOUL_DEFAULT_FALSE: &str = "No, condition is false.";

pub struct Backend {
    model: OptionMarkerModel,
    calibration: Calibration,
}

impl Backend {
    pub fn new(model: OptionMarkerModel, calibration: Calibration) -> Self {
        Self { model, calibration }
    }

    pub fn model(&self) -> &OptionMarkerModel {
        &self.model
    }

    pub fn calibration(&self) -> &Calibration {
        &self.calibration
    }

    /// The exact model input strings `question` produces for `state_text`, one
    /// per forward pass (two for a zero-shot Noul). Useful for debugging and
    /// for checking packing against the Python reference.
    pub fn packed_inputs(&self, state_text: &str, question: &Question) -> Result<Vec<String>> {
        let pack = |state: &str, descriptions: &[String]| {
            self.model
                .pack(state, question.instructions(), descriptions)
        };
        Ok(match question {
            Question::Choice(q) if q.criteria.is_empty() => vec![],
            Question::Choice(q) => vec![pack(state_text, &choice_descriptions(q))],
            Question::Noul(q) => {
                let (descriptions, zero_shot) = noul_descriptions(q);
                let mut out = vec![pack(state_text, &descriptions)];
                if zero_shot {
                    out.push(pack("", &descriptions));
                }
                out
            }
            Question::Score(q) if q.criteria.is_empty() => vec![],
            Question::Score(q) => vec![pack(state_text, &score_descriptions(q)?)],
        })
    }

    /// Evaluates every question against `state`, in insertion order, stamping
    /// the response with `model_id`.
    pub fn evaluate(
        &self,
        state: &Value,
        questions: &IndexMap<String, Question>,
        model_id: &str,
    ) -> Result<SystemOneResponse> {
        let state_text = format_state(state);
        let mut answers = IndexMap::with_capacity(questions.len());
        let mut question_chars = 0;
        for (id, question) in questions {
            let answer = match question {
                Question::Choice(q) => {
                    Answer::Choice(self.evaluate_choice(&state_text, q, None)?)
                }
                Question::Noul(q) => Answer::Noul(self.evaluate_noul(&state_text, q, None)?),
                Question::Score(q) => Answer::Score(self.evaluate_score(&state_text, q, None)?),
            };
            question_chars += question.instructions().chars().count();
            answers.insert(id.clone(), answer);
        }
        // A rough chars/4 estimate, kept identical to Python for protocol parity.
        let state_tokens = (state_text.chars().count() / 4).max(1);
        let question_tokens = (question_chars / 4).max(1);
        Ok(SystemOneResponse {
            model: model_id.to_string(),
            usage: Usage {
                input_tokens: (state_tokens + question_tokens) as u64,
                output_tokens: answers.len() as u64,
            },
            answers,
        })
    }

    pub fn evaluate_choice(
        &self,
        state_text: &str,
        q: &Choice,
        temperature: Option<f64>,
    ) -> Result<ChoiceAnswer> {
        if q.criteria.is_empty() {
            return Ok(ChoiceAnswer {
                choice: String::new(),
                probabilities: IndexMap::new(),
                confidence: 0.0,
            });
        }
        let descriptions = choice_descriptions(q);
        let logits = self.option_logits(state_text, &q.instructions, &descriptions)?;
        let probs = self.probabilities(&logits, state_text, descriptions.len(), temperature)?;

        let best = argmax(&logits);
        Ok(ChoiceAnswer {
            choice: q
                .criteria
                .get_index(best)
                .expect("argmax is in range")
                .0
                .clone(),
            probabilities: q
                .criteria
                .keys()
                .cloned()
                .zip(probs.iter().map(|p| round(*p, 4)))
                .collect(),
            confidence: margin_confidence(&probs),
        })
    }

    pub fn evaluate_noul(
        &self,
        state_text: &str,
        q: &Noul,
        temperature: Option<f64>,
    ) -> Result<NoulAnswer> {
        let (descriptions, zero_shot) = noul_descriptions(q);
        let mut logits = self.option_logits(state_text, &q.instructions, &descriptions)?;
        if zero_shot {
            // Zero-shot: cancel part of the context-free negative polarity prior.
            let null = self.option_logits("", &q.instructions, &descriptions)?;
            logits[0] -= self.calibration.noul_correction(null[0] - null[1]);
        }
        let probs = self.probabilities(&logits, state_text, 2, temperature)?;
        Ok(NoulAnswer {
            noul: round(probs[0].clamp(0.0, 1.0), 4),
        })
    }

    pub fn evaluate_score(
        &self,
        state_text: &str,
        q: &Score,
        temperature: Option<f64>,
    ) -> Result<ScoreAnswer> {
        if q.criteria.is_empty() {
            return Ok(ScoreAnswer {
                score: 0.0,
                confidence: 0.0,
                legend: IndexMap::new(),
                probabilities: IndexMap::new(),
            });
        }
        let descriptions = score_descriptions(q)?;
        let logits = self.option_logits(state_text, &q.instructions, &descriptions)?;
        let probs = self.probabilities(&logits, state_text, descriptions.len(), temperature)?;

        let expected: f64 = probs.iter().enumerate().map(|(i, p)| i as f64 * p).sum();
        Ok(ScoreAnswer {
            score: round(expected, 2),
            confidence: margin_confidence(&probs),
            legend: descriptions
                .iter()
                .enumerate()
                .map(|(i, d)| (i.to_string(), d.clone()))
                .collect(),
            probabilities: probs
                .iter()
                .enumerate()
                .map(|(i, p)| (i.to_string(), round(*p, 4)))
                .collect(),
        })
    }

    fn option_logits(
        &self,
        state_text: &str,
        instructions: &str,
        descriptions: &[String],
    ) -> Result<Vec<f32>> {
        let packed = self.model.pack(state_text, instructions, descriptions);
        self.model.option_logits(&packed, descriptions.len())
    }

    /// Calibrated probabilities, widened to f64 like torch's `.tolist()`.
    fn probabilities(
        &self,
        logits: &[f32],
        state_text: &str,
        n_options: usize,
        temperature: Option<f64>,
    ) -> Result<Vec<f64>> {
        let mut token_error = None;
        let t = self.calibration.effective_temperature(
            logits,
            n_options,
            || {
                self.model.count_tokens(state_text).unwrap_or_else(|e| {
                    token_error = Some(e);
                    0
                })
            },
            temperature,
        );
        if let Some(e) = token_error {
            return Err(e);
        }
        Ok(softmax_f32(logits, t).into_iter().map(f64::from).collect())
    }
}

/// Option descriptions for a Choice: the description, or the id when it is missing or empty.
fn choice_descriptions(q: &Choice) -> Vec<String> {
    q.criteria
        .iter()
        .map(|(id, desc)| match desc.as_deref() {
            Some(d) if !d.is_empty() => py_strip(d).to_string(),
            _ => py_strip(id).to_string(),
        })
        .collect()
}

/// `[true, false]` descriptions for a Noul, and whether it runs zero-shot
/// (neither side given, so the defaults are used and the prior is debiased).
fn noul_descriptions(q: &Noul) -> (Vec<String>, bool) {
    let pick = |key: &str| {
        q.criteria
            .as_ref()
            .and_then(|c| c.get(key))
            .filter(|d| !d.is_empty())
            .cloned()
    };
    let (pos, neg) = (pick("true"), pick("false"));
    let zero_shot = pos.is_none() && neg.is_none();
    let descriptions = vec![
        pos.unwrap_or_else(|| NOUL_DEFAULT_TRUE.into()),
        neg.unwrap_or_else(|| NOUL_DEFAULT_FALSE.into()),
    ];
    (descriptions, zero_shot)
}

fn score_descriptions(q: &Score) -> Result<Vec<String>> {
    q.criteria.iter().map(level_description).collect()
}

/// Legend text for one score level: `what` plus ` Examples: a, b`, stripped.
fn level_description(level: &ScoreLevel) -> Result<String> {
    match level {
        ScoreLevel::Text(s) => Ok(py_strip(s).to_string()),
        ScoreLevel::Detailed(item) => {
            let what = item.get("what").map(py_str).unwrap_or_default();
            let examples = match item.get("examples") {
                None | Some(Value::Null) => Vec::new(),
                Some(Value::Array(xs)) => xs
                    .iter()
                    .map(|x| {
                        x.as_str().map(str::to_string).ok_or_else(|| {
                            VonError::InvalidQuestion(format!(
                                "score level examples must be strings, got {x}"
                            ))
                        })
                    })
                    .collect::<Result<Vec<_>>>()?,
                Some(other) => {
                    return Err(VonError::InvalidQuestion(format!(
                        "score level examples must be a list of strings, got {other}"
                    )));
                }
            };
            let suffix = if examples.is_empty() {
                String::new()
            } else {
                format!(" Examples: {}", examples.join(", "))
            };
            Ok(py_strip(&format!("{what}{suffix}")).to_string())
        }
    }
}

/// Index of the first maximum, like `torch.argmax`.
fn argmax(xs: &[f32]) -> usize {
    xs.iter()
        .enumerate()
        .fold(0, |best, (i, x)| if *x > xs[best] { i } else { best })
}

/// Top-1 minus top-2 probability, clamped to [0, 1] and rounded to 3 places.
fn margin_confidence(probs: &[f64]) -> f64 {
    let mut sorted = probs.to_vec();
    sorted.sort_by(|a, b| b.total_cmp(a));
    let second = sorted.get(1).copied().unwrap_or(0.0);
    round((sorted[0] - second).clamp(0.0, 1.0), 3)
}
