//! Native Rust runtime for Von, the open-source System One decision model.
//!
//! Von answers typed questions about a state (Choice, Noul, Score) with
//! calibrated probabilities in one non-autoregressive forward pass. The Python
//! package in `src/von/` is the reference implementation; this crate matches it
//! numerically (see `tests/parity.rs`).
//!
//! ```no_run
//! use indexmap::IndexMap;
//! use von::{Choice, Question, Von, LoadOptions};
//!
//! let von = Von::load(LoadOptions::default())?;
//! let mut criteria = IndexMap::new();
//! criteria.insert("refund".to_string(), Some("Wants money back".to_string()));
//! criteria.insert("bug".to_string(), Some("Reports a defect".to_string()));
//! let mut questions = IndexMap::new();
//! questions.insert("intent".to_string(), Question::from(Choice { instructions: "What does the user want?".into(), criteria }));
//! let response = von.evaluate(&"I was charged twice".into(), &questions)?;
//! # Ok::<(), von::VonError>(())
//! ```

pub mod api;
pub mod backend;
pub mod calibration;
pub mod config;
pub mod device;
pub mod engine;
pub mod error;
pub mod model;
pub mod patterns;
pub mod presets;
pub mod pyfmt;
pub mod pyjson;
pub mod state;
pub mod types;
pub mod weights;

#[cfg(feature = "cli")]
pub mod cli;
#[cfg(feature = "client")]
pub mod client;
#[cfg(feature = "server")]
pub mod server;

pub use api::{Choices, Decider, decide, judge, rate, system_one};
#[cfg(feature = "client")]
pub use client::{ClientOptions, VonClient};
pub use engine::{LoadOptions, MODEL_ALIASES, VON_MODEL_ID, VON_VERSION, Von};
pub use error::{Result, VonError};
pub use types::{
    Answer, Choice, ChoiceAnswer, Noul, NoulAnswer, Question, Score, ScoreAnswer, ScoreLevel,
    SystemOneResponse, Usage,
};
