//! Hanzo Edge CLI — on-device AI inference from the command line.
//!
//! Subcommands:
//!   run    — Run inference, streaming tokens to stdout
//!   info   — Print model architecture and metadata
//!   bench  — Benchmark TTFT, throughput, and memory
//!   serve  — Start an OpenAI-compatible HTTP server

use anyhow::Result;
use clap::{Parser, Subcommand};
use tracing_subscriber::EnvFilter;

mod cmd;
mod loader;

#[derive(Parser)]
#[command(
    name = "hanzo-edge",
    about = "Hanzo Edge — on-device AI inference",
    version,
    propagate_version = true
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Run inference on a model.
    Run {
        /// Model ID (HuggingFace repo) or local .gguf path.
        #[arg(short, long, default_value = "zenlm/zen3-nano")]
        model: String,

        /// Prompt text.
        #[arg(short, long)]
        prompt: String,

        /// Maximum tokens to generate.
        #[arg(long, default_value_t = 256)]
        max_tokens: usize,

        /// Sampling temperature (0 = greedy).
        #[arg(long, default_value_t = 0.7)]
        temperature: f64,

        /// Top-p nucleus sampling threshold.
        #[arg(long, default_value_t = 0.9)]
        top_p: f64,

        /// HuggingFace revision/branch.
        #[arg(long)]
        revision: Option<String>,
    },

    /// Show model info (architecture, params, quantization).
    Info {
        /// Model ID (HuggingFace repo) or local .gguf path.
        #[arg(short, long, default_value = "zenlm/zen3-nano")]
        model: String,

        /// HuggingFace revision/branch.
        #[arg(long)]
        revision: Option<String>,
    },

    /// Benchmark model inference.
    Bench {
        /// Model ID (HuggingFace repo) or local .gguf path.
        #[arg(short, long, default_value = "zenlm/zen3-nano")]
        model: String,

        /// Prompt text for benchmarking.
        #[arg(short, long, default_value = "Hello")]
        prompt: String,

        /// Maximum tokens to generate per iteration.
        #[arg(long, default_value_t = 128)]
        max_tokens: usize,

        /// Number of benchmark iterations.
        #[arg(short = 'n', long, default_value_t = 5)]
        iterations: usize,

        /// HuggingFace revision/branch.
        #[arg(long)]
        revision: Option<String>,
    },

    /// Start a local OpenAI-compatible HTTP server.
    Serve {
        /// Model ID (HuggingFace repo) or local .gguf path.
        #[arg(short, long, default_value = "zenlm/zen3-nano")]
        model: String,

        /// Port to listen on.
        #[arg(short, long, default_value_t = 8080)]
        port: u16,

        /// HuggingFace revision/branch.
        #[arg(long)]
        revision: Option<String>,
    },
}

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .init();

    let cli = Cli::parse();

    match cli.command {
        Commands::Run {
            model,
            prompt,
            max_tokens,
            temperature,
            top_p,
            revision,
        } => {
            let device = hanzo_edge_core::default_device()?;
            cmd::run::execute(
                &model,
                &prompt,
                max_tokens,
                temperature,
                top_p,
                revision.as_deref(),
                &device,
            )
        }

        Commands::Info { model, revision } => {
            cmd::info::execute(&model, revision.as_deref())
        }

        Commands::Bench {
            model,
            prompt,
            max_tokens,
            iterations,
            revision,
        } => {
            let device = hanzo_edge_core::default_device()?;
            cmd::bench::execute(
                &model,
                &prompt,
                max_tokens,
                iterations,
                revision.as_deref(),
                &device,
            )
        }

        Commands::Serve {
            model,
            port,
            revision,
        } => {
            let device = hanzo_edge_core::default_device()?;
            let rt = tokio::runtime::Runtime::new()?;
            rt.block_on(cmd::serve::execute(
                &model,
                port,
                revision.as_deref(),
                &device,
            ))
        }
    }
}
