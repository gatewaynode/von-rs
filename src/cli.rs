//! The `von` command line (port of `cli.py`): `serve`, `decide`, `judge`, `rate`
//! and `eval`, with the same flags, defaults, messages, exit codes and JSON output
//! (checked against `tests/fixtures/protocol.json`).
//!
//! stdout carries only the JSON result; logs (including the load line) go to
//! stderr. Python prints its `[von] Loaded ...` line to stdout.

use std::ffi::OsString;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use clap::{Args, CommandFactory, FromArgMatches, Parser, Subcommand};
use indexmap::IndexMap;
use serde_json::{Value, json};

use crate::api::{self, Choices, DEFAULT_MODEL, Decider, RATE_INSTRUCTIONS};
use crate::config::Config;
use crate::engine::{LoadOptions, MODEL_ALIASES, VON_MODEL_ID, VON_VERSION, Von};
use crate::error::VonError;
use crate::pyfmt::py_strip;
use crate::pyjson;
use crate::types::{Question, ScoreLevel};

/// Printed by `von --version`, in click's format. (Python prints a stale `1.0.0`.)
pub const CLI_VERSION: &str = "1.1.0";
const DECIDE_INSTRUCTIONS: &str = "Which option best describes the input?";

#[derive(Parser)]
#[command(
    name = "von",
    about = "Von - Open Source System One Decision Model.",
    disable_version_flag = true
)]
struct Cli {
    /// Show the version and exit.
    #[arg(long)]
    version: bool,
    #[command(subcommand)]
    command: Option<Command>,
}

#[derive(Subcommand)]
enum Command {
    /// Start the Von System One HTTP server.
    Serve(ServeArgs),
    /// Classify input text among discrete choices.
    Decide {
        text: String,
        /// Comma-separated choices (e.g. 'billing,bug_report,feature_request').
        #[arg(short, long)]
        choices: String,
        /// Instructions for classification.
        #[arg(short, long, default_value = DECIDE_INSTRUCTIONS)]
        instructions: String,
        #[command(flatten)]
        device: DeviceArg,
    },
    /// Evaluate a yes/no judgment (Noul) and return the probability.
    Judge {
        text: String,
        /// Boolean judgment question (e.g. 'Is the server down?').
        #[arg(short, long)]
        instructions: String,
        /// Explicit criteria description for True condition.
        #[arg(long, default_value = "")]
        pos: String,
        /// Explicit criteria description for False condition.
        #[arg(long, default_value = "")]
        neg: String,
        #[command(flatten)]
        device: DeviceArg,
    },
    /// Rate text on an ordered multi-level scale (Score).
    Rate {
        text: String,
        /// Comma-separated descriptions of ordered levels from 0 to N-1.
        #[arg(short, long)]
        levels: String,
        /// Instructions for rating.
        #[arg(short, long, default_value = RATE_INSTRUCTIONS)]
        instructions: String,
        #[command(flatten)]
        device: DeviceArg,
    },
    /// Evaluate a JSON request file containing state and questions.
    Eval {
        #[arg(value_name = "REQUEST_FILE")]
        request_file: String,
    },
}

#[derive(Args)]
struct DeviceArg {
    /// Compute device: 'auto', 'metal' (alias 'mps') or 'cpu'.
    #[arg(long, default_value = "auto")]
    device: String,
}

impl DeviceArg {
    /// Python only overrides `VON_DEVICE` when the flag is not `auto`.
    fn load_options(&self) -> LoadOptions {
        LoadOptions {
            device: (self.device != "auto").then(|| self.device.clone()),
            ..Default::default()
        }
    }
}

#[derive(Args)]
pub struct ServeArgs {
    /// Host interface to bind on.
    #[arg(long, default_value = "0.0.0.0")]
    host: String,
    /// Port to listen on.
    #[arg(long, default_value_t = 8000)]
    port: u16,
    /// Von model version to load (Von 1.1 is the only model).
    #[arg(long, default_value = "von-1.1", value_parser = sorted_aliases())]
    model: String,
    /// Compute device: 'auto', 'metal' (alias 'mps') or 'cpu'.
    #[arg(long, default_value = "auto")]
    device: String,
    /// Accepted for compatibility and ignored.
    #[arg(long)]
    reload: bool,
}

fn sorted_aliases() -> Vec<&'static str> {
    let mut aliases = MODEL_ALIASES.to_vec();
    aliases.sort_unstable();
    aliases
}

/// Loads (or connects to) the decider a command runs against.
pub type Connect<'a> = dyn FnMut(LoadOptions) -> crate::Result<Arc<dyn Decider>> + 'a;

/// Runs the CLI and returns the process exit code. `connect` is called only
/// after argument validation passes, as Python loads the engine lazily.
/// `config` fills settings that neither the flags nor the environment give.
pub fn run<I, T>(
    args: I,
    config: &Config,
    connect: &mut Connect<'_>,
    out: &mut dyn Write,
    err: &mut dyn Write,
) -> i32
where
    I: IntoIterator<Item = T>,
    T: Into<OsString> + Clone,
{
    let cli = match Cli::command()
        .try_get_matches_from(args)
        .and_then(|m| Cli::from_arg_matches(&m))
    {
        Ok(cli) => cli,
        Err(e) => {
            let text = e.render().to_string();
            let _ = if e.use_stderr() {
                write!(err, "{text}")
            } else {
                write!(out, "{text}")
            };
            return e.exit_code();
        }
    };
    if cli.version {
        let _ = writeln!(out, "von, version {CLI_VERSION}");
        return 0;
    }
    let Some(command) = cli.command else {
        let _ = write!(out, "{}", Cli::command().render_help());
        return 0;
    };
    let mut connect = |opts: LoadOptions| connect(fill_from(opts, config));
    match execute(command, config, &mut connect, out, err) {
        Ok(code) => code,
        Err(e) => {
            let _ = writeln!(err, "Error: {e}");
            1
        }
    }
}

/// Fills load options that the flags left unset from the settings file, where
/// the environment does not set them (the library reads the environment itself).
fn fill_from(mut opts: LoadOptions, config: &Config) -> LoadOptions {
    opts.model = opts.model.or_else(|| config.fallback("VON_BACKEND"));
    opts.device = opts.device.or_else(|| config.fallback("VON_DEVICE"));
    opts.checkpoint_dir = opts
        .checkpoint_dir
        .or_else(|| config.fallback("VON_CHECKPOINT_DIR").map(PathBuf::from));
    opts
}

fn execute(
    command: Command,
    config: &Config,
    connect: &mut Connect<'_>,
    out: &mut dyn Write,
    err: &mut dyn Write,
) -> Result<i32, VonError> {
    let output = match command {
        Command::Serve(args) => return serve(args, config, out).map(|()| 0),
        Command::Decide {
            text,
            choices,
            instructions,
            device,
        } => {
            let options = split_list(&choices);
            if options.is_empty() {
                return print_error(err, "Error: At least one choice must be provided.");
            }
            // Python checks for duplicates before the engine loads.
            Choices::from(options.clone()).into_criteria()?;
            let decider = connect(device.load_options())?;
            let answer = api::decide(
                &*decider,
                &Value::String(text),
                options,
                Some(&instructions),
                None,
            )?;
            json!({
                "choice": answer.choice,
                "confidence": answer.confidence,
                "probabilities": answer.probabilities,
            })
        }
        Command::Judge {
            text,
            instructions,
            pos,
            neg,
            device,
        } => {
            let mut criteria = IndexMap::new();
            if !pos.is_empty() {
                criteria.insert("true".to_string(), pos);
            }
            if !neg.is_empty() {
                criteria.insert("false".to_string(), neg);
            }
            let decider = connect(device.load_options())?;
            let p = api::judge(
                &*decider,
                &Value::String(text),
                &instructions,
                (!criteria.is_empty()).then_some(criteria),
                None,
            )?;
            json!({ "type": "noul", "instructions": instructions, "noul": p })
        }
        Command::Rate {
            text,
            levels,
            instructions,
            device,
        } => {
            let levels = split_list(&levels);
            if levels.len() < 2 {
                return print_error(err, "Error: At least two levels must be provided.");
            }
            let decider = connect(device.load_options())?;
            let answer = api::rate(
                &*decider,
                &Value::String(text),
                levels.into_iter().map(ScoreLevel::from).collect(),
                Some(&instructions),
                None,
            )?;
            json!({
                "type": "score",
                "score": answer.score,
                "confidence": answer.confidence,
                "legend": answer.legend,
                "probabilities": answer.probabilities,
            })
        }
        Command::Eval { request_file } => match eval(&request_file, connect, err)? {
            Ok(response) => response,
            Err(code) => return Ok(code),
        },
    };
    let _ = writeln!(out, "{}", pyjson::pretty(&output));
    Ok(0)
}

/// `von eval`: the request file's `state`, `questions` and optional `model`.
/// Returns `Err(exit code)` after printing a usage-style error.
fn eval(
    request_file: &str,
    connect: &mut Connect<'_>,
    err: &mut dyn Write,
) -> Result<Result<Value, i32>, VonError> {
    let path = Path::new(request_file);
    if !path.exists() {
        // click.Path(exists=True)
        let _ = write!(
            err,
            "Usage: von eval [OPTIONS] REQUEST_FILE\nTry 'von eval --help' for help.\n\n\
             Error: Invalid value for 'REQUEST_FILE': Path '{request_file}' does not exist.\n"
        );
        return Ok(Err(2));
    }
    let text = match std::fs::read_to_string(path) {
        Ok(t) => t,
        Err(e) => {
            let _ = writeln!(err, "Error: cannot read {request_file}: {e}");
            return Ok(Err(1));
        }
    };
    let data: Value = match serde_json::from_str(&text) {
        Ok(d) => d,
        Err(e) => {
            let _ = writeln!(err, "Error: {request_file} is not valid JSON: {e}");
            return Ok(Err(1));
        }
    };
    let Value::Object(data) = data else {
        return Err(VonError::InvalidQuestion(
            "the request file must contain a JSON object".into(),
        ));
    };
    let (Some(state), Some(questions)) = (
        data.get("state").filter(|v| !v.is_null()),
        data.get("questions").filter(|v| !v.is_null()),
    ) else {
        let _ = writeln!(
            err,
            "Error: JSON must contain 'state' and 'questions' fields."
        );
        return Ok(Err(1));
    };
    let questions: IndexMap<String, Question> = serde_json::from_value(questions.clone())
        .map_err(|e| VonError::InvalidQuestion(e.to_string()))?;
    let model = data
        .get("model")
        .and_then(Value::as_str)
        .unwrap_or(DEFAULT_MODEL);
    let decider = connect(LoadOptions::default())?;
    let response = api::system_one(&*decider, state, &questions, Some(model))?;
    Ok(Ok(
        serde_json::to_value(response).expect("responses serialize")
    ))
}

/// `[x.strip() for x in raw.split(",") if x.strip()]`
fn split_list(raw: &str) -> Vec<String> {
    raw.split(',')
        .map(py_strip)
        .filter(|s| !s.is_empty())
        .map(String::from)
        .collect()
}

fn print_error(err: &mut dyn Write, message: &str) -> Result<i32, VonError> {
    let _ = writeln!(err, "{message}");
    Ok(1)
}

/// `von serve`: loads the model eagerly, then serves until Ctrl-C.
fn serve(args: ServeArgs, config: &Config, out: &mut dyn Write) -> Result<(), VonError> {
    let opts = fill_from(
        LoadOptions {
            model: Some(args.model.clone()),
            device: (args.device != "auto").then(|| args.device.clone()),
            ..Default::default()
        },
        config,
    );
    let device = crate::device::resolve_device(opts.device.as_deref())?;
    let _ = writeln!(
        out,
        "Starting Von Decision Server [{} on {}] on http://{}:{}",
        args.model,
        crate::device::describe(&device),
        args.host,
        args.port
    );
    if args.reload {
        tracing::warn!("--reload is accepted for compatibility and ignored");
    }
    // Load before starting the runtime: the Hub download uses its own blocking runtime.
    let von = Von::load(opts)?;
    let max_in_flight = if von.device().is_metal() {
        1
    } else {
        std::thread::available_parallelism().map_or(1, |n| n.get())
    };
    let config = crate::server::ServerConfig::from_lookup(max_in_flight, |k| config.var(k));
    let app = crate::server::router(Arc::new(von), config);
    let io = |e: std::io::Error| VonError::Io {
        path: format!("{}:{}", args.host, args.port).into(),
        source: e,
    };
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .map_err(io)?;
    runtime.block_on(async {
        let listener = tokio::net::TcpListener::bind((args.host.as_str(), args.port))
            .await
            .map_err(io)?;
        tracing::info!(
            "Von {VON_VERSION} ({VON_MODEL_ID}) listening on http://{}",
            listener.local_addr().map_err(io)?
        );
        crate::server::serve(listener, app).await.map_err(io)
    })
}

/// Entry point for the `von` binary.
pub fn main() -> std::process::ExitCode {
    tracing_subscriber::fmt()
        .with_writer(std::io::stderr)
        .with_target(false)
        .init();
    let config = Config::load();
    let mut connect =
        |opts: LoadOptions| -> crate::Result<Arc<dyn Decider>> { Ok(Arc::new(Von::load(opts)?)) };
    let code = run(
        std::env::args_os(),
        &config,
        &mut connect,
        &mut std::io::stdout(),
        &mut std::io::stderr(),
    );
    std::process::ExitCode::from(u8::try_from(code).unwrap_or(1))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flags_win_over_the_settings_file() {
        // A key no environment sets, so the file value applies.
        let config = Config::from_text("VON_DEVICE=metal\nVON_TEST_UNSET_KEY=x\n");
        assert_eq!(config.fallback("VON_TEST_UNSET_KEY").as_deref(), Some("x"));
        let opts = fill_from(
            LoadOptions {
                device: Some("cpu".into()),
                ..Default::default()
            },
            &config,
        );
        assert_eq!(opts.device.as_deref(), Some("cpu"));
    }
}
