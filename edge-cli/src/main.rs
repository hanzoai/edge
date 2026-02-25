//! Hanzo Edge CLI — run on-device inference from the command line.

use anyhow::Result;
use clap::{Parser, Subcommand};
use tracing_subscriber::EnvFilter;

#[derive(Parser)]
#[command(
    name = "hanzo-edge",
    about = "Hanzo Edge - On-device AI inference",
    version
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Run inference on a model.
    Run {
        /// Model ID (HuggingFace repo or local path).
        #[arg(short, long, default_value = "zenlm/zen3-nano")]
        model: String,

        /// Prompt text.
        #[arg(short, long)]
        prompt: String,

        /// Maximum tokens to generate.
        #[arg(long, default_value_t = 256)]
        max_tokens: usize,

        /// Sampling temperature.
        #[arg(long, default_value_t = 0.7)]
        temperature: f64,

        /// HuggingFace revision/branch.
        #[arg(long)]
        revision: Option<String>,
    },
    /// Show available backends and device info.
    Info,
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
            revision,
        } => cmd_run(model, prompt, max_tokens, temperature, revision),
        Commands::Info => cmd_info(),
    }
}

fn cmd_run(
    model_id: String,
    prompt: String,
    max_tokens: usize,
    temperature: f64,
    _revision: Option<String>,
) -> Result<()> {
    let device = edge_core::default_device()?;
    tracing::info!(model = %model_id, "loading model");

    let params = edge_core::SamplingParams {
        temperature,
        max_tokens,
        ..Default::default()
    };

    // TODO: load model weights from HF Hub via hf-hub crate,
    // instantiate a concrete Model impl, create tokenizer,
    // then call edge_core::session::generate_text().
    //
    // For now, print what we would do:
    println!("Hanzo Edge");
    println!("  Model:       {model_id}");
    println!("  Device:      {device:?}");
    println!("  Prompt:      {prompt}");
    println!("  Max tokens:  {max_tokens}");
    println!("  Temperature: {temperature}");
    println!();
    println!("[model loading not yet implemented — scaffold only]");
    println!("Use `hanzo-edge info` to check available backends.");

    let _ = params; // suppress unused warning

    Ok(())
}

fn cmd_info() -> Result<()> {
    println!("Hanzo Edge v{}", env!("CARGO_PKG_VERSION"));
    println!();

    let device = edge_core::default_device()?;
    println!("Device: {device:?}");

    println!("Backends:");
    #[cfg(feature = "metal")]
    println!("  - Metal (enabled)");
    #[cfg(not(feature = "metal"))]
    println!("  - Metal (disabled)");

    #[cfg(feature = "cuda")]
    println!("  - CUDA (enabled)");
    #[cfg(not(feature = "cuda"))]
    println!("  - CUDA (disabled)");

    println!("  - CPU (always available)");

    Ok(())
}
