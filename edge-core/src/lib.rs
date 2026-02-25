//! Hanzo Edge Core — on-device AI inference runtime.
//!
//! Provides traits and implementations for running transformer models
//! on-device using Hanzo ML (Candle) as the tensor backend.

use anyhow::Result;
use candle_core::{Device, Tensor};
use serde::{Deserialize, Serialize};

pub mod model;
pub mod session;

pub use model::{Model, ModelConfig};
pub use session::{InferenceSession, SamplingParams};

/// Select the best available device for the current platform.
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

/// Token output from generation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerateOutput {
    pub text: String,
    pub tokens: Vec<u32>,
    pub prompt_tokens: usize,
    pub generated_tokens: usize,
}
