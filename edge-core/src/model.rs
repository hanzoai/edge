//! Model trait and configuration.

use anyhow::Result;
use candle_core::{Device, Tensor};
use serde::{Deserialize, Serialize};

/// Configuration for loading a model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    /// HuggingFace model ID or local path.
    pub model_id: String,
    /// Optional revision/branch.
    pub revision: Option<String>,
    /// Maximum sequence length.
    pub max_seq_len: usize,
    /// Data type (f32, f16, bf16).
    pub dtype: String,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            model_id: "zenlm/zen3-nano".to_string(),
            revision: None,
            max_seq_len: 2048,
            dtype: "f32".to_string(),
        }
    }
}

/// Core trait for a loadable, runnable model.
pub trait Model: Send {
    /// Forward pass: given input token IDs, return logits.
    fn forward(&mut self, input_ids: &Tensor, position: usize) -> Result<Tensor>;

    /// Reset the model's KV cache (if any).
    fn reset(&mut self);

    /// Device this model is loaded on.
    fn device(&self) -> &Device;

    /// Maximum sequence length.
    fn max_seq_len(&self) -> usize;
}
