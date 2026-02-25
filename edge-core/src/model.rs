//! Model loading and the core `Model` trait.
//!
//! Supports GGUF quantized models (Llama architecture) loaded from
//! HuggingFace Hub or local files. Uses `candle_transformers::models::quantized_llama`
//! as the concrete implementation.

use std::io::BufReader;
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use candle_core::quantized::gguf_file;
use candle_core::{Device, Tensor};
use candle_transformers::models::quantized_llama::ModelWeights;
use serde::{Deserialize, Serialize};

use crate::tokenizer::TokenizerWrapper;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Which model architecture the GGUF file contains.
///
/// All listed architectures use the Llama-family GGUF layout
/// (attention_norm, ffn_norm, attn_q/k/v/output, ffn_gate/down/up)
/// so they share the same `ModelWeights` loader.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ModelArchitecture {
    Llama,
    Qwen2,
    Phi3,
    Gemma2,
    SmolLM,
}

impl std::fmt::Display for ModelArchitecture {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            Self::Llama => "llama",
            Self::Qwen2 => "qwen2",
            Self::Phi3 => "phi3",
            Self::Gemma2 => "gemma2",
            Self::SmolLM => "smollm",
        };
        f.write_str(s)
    }
}

/// Configuration for downloading and loading a model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    /// HuggingFace model ID (e.g. "TheBloke/Llama-2-7B-Chat-GGUF")
    /// or an absolute path to a local GGUF file.
    pub model_id: String,

    /// Specific filename within the HF repo (e.g. "llama-2-7b-chat.Q4_K_M.gguf").
    /// Required for HF repos that contain multiple GGUF files.
    pub model_file: Option<String>,

    /// HF revision / branch. Default: "main".
    pub revision: Option<String>,

    /// Architecture hint. When `None`, auto-detected from GGUF metadata.
    pub architecture: Option<ModelArchitecture>,

    /// Device to load onto.
    #[serde(skip)]
    pub device: Option<Device>,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            model_id: "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF".to_string(),
            model_file: Some("tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf".to_string()),
            revision: None,
            architecture: None,
            device: None,
        }
    }
}

// ---------------------------------------------------------------------------
// Model trait
// ---------------------------------------------------------------------------

/// Core trait for a loadable, runnable language model.
pub trait Model: Send {
    /// Forward pass: given input token IDs `(batch=1, seq_len)` and a
    /// starting position, return logits `(vocab_size,)`.
    fn forward(&mut self, input_ids: &Tensor, position: usize) -> Result<Tensor>;

    /// Reset internal KV-cache state (start a new conversation).
    fn reset(&mut self);

    /// Device this model lives on.
    fn device(&self) -> &Device;

    /// Maximum context length the model supports.
    fn max_seq_len(&self) -> usize;
}

// ---------------------------------------------------------------------------
// Quantized Llama implementation
// ---------------------------------------------------------------------------

/// A GGUF-quantized Llama-family model.
pub struct QuantizedLlama {
    weights: ModelWeights,
    device: Device,
    max_seq_len: usize,
}

impl Model for QuantizedLlama {
    fn forward(&mut self, input_ids: &Tensor, position: usize) -> Result<Tensor> {
        let logits = self
            .weights
            .forward(input_ids, position)
            .context("quantized llama forward pass")?;
        Ok(logits)
    }

    fn reset(&mut self) {
        // ModelWeights stores KV cache inside each LayerWeights.
        // The simplest reset is to reload, but for now we accept
        // that callers create a fresh QuantizedLlama per conversation.
        // A future improvement: expose cache clearing in candle_transformers.
    }

    fn device(&self) -> &Device {
        &self.device
    }

    fn max_seq_len(&self) -> usize {
        self.max_seq_len
    }
}

// ---------------------------------------------------------------------------
// Loading
// ---------------------------------------------------------------------------

/// Resolve a GGUF model file path, downloading from HF Hub if needed.
fn resolve_model_path(config: &ModelConfig) -> Result<PathBuf> {
    let model_id = &config.model_id;

    // If model_id is a local file, use it directly.
    let local = Path::new(model_id);
    if local.exists() && local.is_file() {
        tracing::info!(path = %local.display(), "using local GGUF file");
        return Ok(local.to_path_buf());
    }

    // Otherwise treat it as an HF repo ID and download.
    let revision = config.revision.as_deref().unwrap_or("main");
    let api = hf_hub::api::sync::Api::new().context("failed to create HF Hub API client")?;
    let repo = api.repo(hf_hub::Repo::with_revision(
        model_id.clone(),
        hf_hub::RepoType::Model,
        revision.to_string(),
    ));

    let filename = match &config.model_file {
        Some(f) => f.clone(),
        None => {
            // Attempt to find a GGUF file in the repo.
            // The hf-hub crate does not list files, so we require model_file
            // to be specified when model_id is a repo with multiple files.
            bail!(
                "model_file must be specified for HF repo '{}'; \
                 set it to the GGUF filename (e.g. 'model.Q4_K_M.gguf')",
                model_id
            );
        }
    };

    tracing::info!(repo = %model_id, file = %filename, rev = %revision, "downloading from HF Hub");
    let path = repo
        .get(&filename)
        .with_context(|| format!("failed to download {filename} from {model_id}"))?;
    Ok(path)
}

/// Resolve and download the tokenizer.json for the given HF repo.
fn resolve_tokenizer_path(config: &ModelConfig) -> Result<PathBuf> {
    let model_id = &config.model_id;

    // Check for a local tokenizer.json next to a local GGUF file.
    let local = Path::new(model_id);
    if local.exists() && local.is_file() {
        if let Some(parent) = local.parent() {
            let tok_path = parent.join("tokenizer.json");
            if tok_path.exists() {
                return Ok(tok_path);
            }
        }
    }

    let revision = config.revision.as_deref().unwrap_or("main");
    let api = hf_hub::api::sync::Api::new()?;
    let repo = api.repo(hf_hub::Repo::with_revision(
        model_id.clone(),
        hf_hub::RepoType::Model,
        revision.to_string(),
    ));

    let path = repo
        .get("tokenizer.json")
        .with_context(|| format!("failed to download tokenizer.json from {model_id}"))?;
    Ok(path)
}

/// Load a GGUF quantized model and its tokenizer from config.
///
/// Returns `(Box<dyn Model>, TokenizerWrapper)`.
pub fn load_model(config: &ModelConfig) -> Result<(Box<dyn Model>, TokenizerWrapper)> {
    let device = config.device.clone().unwrap_or(Device::Cpu);

    // --- Load GGUF weights ---
    let model_path = resolve_model_path(config)?;
    tracing::info!(path = %model_path.display(), "loading GGUF model");

    let mut file = BufReader::new(
        std::fs::File::open(&model_path)
            .with_context(|| format!("cannot open {}", model_path.display()))?,
    );
    let gguf_content = gguf_file::Content::read(&mut file)
        .context("failed to parse GGUF file")?;

    // Extract max sequence length from metadata (default 4096).
    let max_seq_len = gguf_content
        .metadata
        .get("llama.context_length")
        .and_then(|v| v.to_u32().ok())
        .map(|v| v as usize)
        .unwrap_or(4096);

    let weights = ModelWeights::from_gguf(gguf_content, &mut file, &device)
        .context("failed to load quantized llama weights from GGUF")?;

    tracing::info!(max_seq_len, "model loaded");

    let model = QuantizedLlama {
        weights,
        device,
        max_seq_len,
    };

    // --- Load tokenizer ---
    let tok_path = resolve_tokenizer_path(config)?;
    tracing::info!(path = %tok_path.display(), "loading tokenizer");
    let mut tokenizer = TokenizerWrapper::from_file(
        tok_path.to_str().context("non-UTF8 tokenizer path")?,
    )?;

    // Override EOS from GGUF metadata if available.
    // Many GGUF files store the EOS token ID in metadata.
    if let Some(val) = gguf_content_eos_id(&model_path)? {
        tokenizer.set_eos_token_id(val);
    }

    Ok((Box::new(model), tokenizer))
}

/// Re-read GGUF to extract EOS token ID from metadata.
/// Separate function because Content borrows the reader.
fn gguf_content_eos_id(path: &Path) -> Result<Option<u32>> {
    let mut file = BufReader::new(std::fs::File::open(path)?);
    let content = gguf_file::Content::read(&mut file)?;

    // Try "tokenizer.ggml.eos_token_id" first, then "general.eos_token_id".
    for key in &[
        "tokenizer.ggml.eos_token_id",
        "general.eos_token_id",
    ] {
        if let Some(v) = content.metadata.get(*key) {
            if let Ok(id) = v.to_u32() {
                return Ok(Some(id));
            }
        }
    }
    Ok(None)
}
