//! Hanzo Edge Core -- on-device AI inference runtime.
//!
//! Loads GGUF-quantized transformer models from HuggingFace Hub
//! and runs autoregressive generation using Candle as the tensor backend.

use anyhow::Result;
use candle_core::Device;
use serde::{Deserialize, Serialize};

pub mod model;
pub mod sampling;
pub mod session;
pub mod tokenizer;

pub use model::{Model, ModelArchitecture, ModelConfig};
pub use sampling::{sample_token, SamplingParams};
pub use session::InferenceSession;
pub use tokenizer::TokenizerWrapper;

/// Crate version.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Select the best available compute device for the current platform.
pub fn default_device() -> Result<Device> {
    #[cfg(feature = "metal")]
    {
        tracing::info!("using Metal backend");
        return Ok(Device::new_metal(0)?);
    }

    #[cfg(feature = "cuda")]
    {
        tracing::info!("using CUDA backend");
        return Ok(Device::new_cuda(0)?);
    }

    #[allow(unreachable_code)]
    {
        tracing::info!("using CPU backend");
        Ok(Device::Cpu)
    }
}

/// Output from a generation call.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerateOutput {
    pub text: String,
    pub tokens: Vec<u32>,
    pub prompt_tokens: usize,
    pub generated_tokens: usize,
}
