//! Model loader: downloads from HuggingFace Hub and loads GGUF weights.
//!
//! Supports two paths:
//! 1. Local .gguf file  -> load directly
//! 2. HF repo ID        -> download tokenizer + GGUF weights, then load

use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use candle_core::quantized::gguf_file;
use candle_core::Device;
use candle_transformers::models::quantized_llama::ModelWeights;
use hf_hub::api::sync::Api;
use hf_hub::Repo;
use indicatif::{ProgressBar, ProgressStyle};
use tokenizers::Tokenizer;

/// Everything needed for inference after loading.
pub struct LoadedModel {
    pub weights: ModelWeights,
    pub tokenizer: Tokenizer,
    pub device: Device,
    pub model_id: String,
    pub gguf_path: PathBuf,
}

/// Metadata extracted from a GGUF file without loading the full weights.
pub struct GgufMetadata {
    pub architecture: String,
    pub name: String,
    pub param_count: u64,
    pub context_length: u64,
    pub embedding_length: u64,
    pub block_count: u64,
    pub head_count: u64,
    pub head_count_kv: u64,
    pub quantization: String,
    pub file_size: u64,
}

/// Resolve a model identifier to a local GGUF path + tokenizer path.
///
/// If `model_id` is a local path ending in .gguf, use it directly.
/// Otherwise treat it as a HuggingFace repo and download.
pub fn resolve_model(
    model_id: &str,
    revision: Option<&str>,
) -> Result<(PathBuf, PathBuf)> {
    let path = Path::new(model_id);

    // Local .gguf file.
    if path.extension().map_or(false, |e| e == "gguf") {
        if !path.exists() {
            bail!("GGUF file not found: {}", path.display());
        }
        // Look for tokenizer.json next to the gguf file.
        let dir = path.parent().unwrap_or(Path::new("."));
        let tok_path = dir.join("tokenizer.json");
        if !tok_path.exists() {
            bail!(
                "tokenizer.json not found next to GGUF file (looked in {})",
                dir.display()
            );
        }
        return Ok((path.to_path_buf(), tok_path));
    }

    // HuggingFace repo — download.
    download_from_hub(model_id, revision)
}

/// Download model files from HuggingFace Hub.
///
/// Looks for a single .gguf file in the repo. If multiple exist,
/// picks the first one found. Also downloads tokenizer.json.
fn download_from_hub(
    model_id: &str,
    revision: Option<&str>,
) -> Result<(PathBuf, PathBuf)> {
    let pb = ProgressBar::new_spinner();
    pb.set_style(
        ProgressStyle::default_spinner()
            .template("{spinner:.cyan} {msg}")
            .expect("valid template"),
    );
    pb.set_message(format!("Connecting to HuggingFace Hub for {model_id}..."));

    let api = Api::new().context("failed to initialize HuggingFace Hub API")?;
    let repo = if let Some(rev) = revision {
        api.repo(Repo::with_revision(
            model_id.to_string(),
            hf_hub::RepoType::Model,
            rev.to_string(),
        ))
    } else {
        api.model(model_id.to_string())
    };

    // Download tokenizer.
    pb.set_message("Downloading tokenizer.json...");
    let tok_path = repo
        .get("tokenizer.json")
        .context("failed to download tokenizer.json — is this a valid model repo?")?;

    // Try common GGUF filenames in order of preference.
    let gguf_candidates = [
        format!("{}.gguf", model_id.split('/').last().unwrap_or(model_id)),
        "model.gguf".to_string(),
        "model-q4_0.gguf".to_string(),
        "model-q4_k_m.gguf".to_string(),
        "model-q5_k_m.gguf".to_string(),
        "model-q8_0.gguf".to_string(),
    ];

    let mut gguf_path = None;
    for name in &gguf_candidates {
        pb.set_message(format!("Trying {name}..."));
        match repo.get(name) {
            Ok(p) => {
                gguf_path = Some(p);
                break;
            }
            Err(_) => continue,
        }
    }

    let gguf_path = gguf_path.context(
        "no .gguf file found in repo — tried common names. \
         Specify a local .gguf path instead.",
    )?;

    pb.finish_with_message(format!("Model files cached for {model_id}"));

    Ok((gguf_path, tok_path))
}

/// Load a GGUF model and tokenizer, ready for inference.
pub fn load_model(
    model_id: &str,
    revision: Option<&str>,
    device: &Device,
) -> Result<LoadedModel> {
    let (gguf_path, tok_path) = resolve_model(model_id, revision)?;

    tracing::info!(path = %gguf_path.display(), "loading GGUF weights");

    let mut file = std::fs::File::open(&gguf_path)
        .with_context(|| format!("cannot open {}", gguf_path.display()))?;

    let content = gguf_file::Content::read(&mut file)
        .map_err(|e| anyhow::anyhow!("failed to read GGUF content: {e}"))?;

    let weights = ModelWeights::from_gguf(content, &mut file, device)
        .map_err(|e| anyhow::anyhow!("failed to load model weights: {e}"))?;

    tracing::info!(path = %tok_path.display(), "loading tokenizer");
    let tokenizer = Tokenizer::from_file(&tok_path)
        .map_err(|e| anyhow::anyhow!("failed to load tokenizer: {e}"))?;

    Ok(LoadedModel {
        weights,
        tokenizer,
        device: device.clone(),
        model_id: model_id.to_string(),
        gguf_path,
    })
}

/// Read GGUF metadata without loading weights.
pub fn read_gguf_metadata(path: &Path) -> Result<GgufMetadata> {
    let file_size = std::fs::metadata(path)
        .with_context(|| format!("cannot stat {}", path.display()))?
        .len();

    let mut file = std::fs::File::open(path)
        .with_context(|| format!("cannot open {}", path.display()))?;

    let content = gguf_file::Content::read(&mut file)
        .map_err(|e| anyhow::anyhow!("failed to read GGUF: {e}"))?;

    let md = &content.metadata;

    let get_str = |key: &str| -> String {
        md.get(key)
            .and_then(|v| v.to_string().ok())
            .cloned()
            .unwrap_or_default()
    };

    let get_u64 = |key: &str| -> u64 {
        md.get(key)
            .and_then(|v| v.to_u32().ok().map(|x| x as u64).or_else(|| v.to_u64().ok()))
            .unwrap_or(0)
    };

    let architecture = get_str("general.architecture");
    let name = get_str("general.name");

    // Try arch-prefixed keys, fall back to generic.
    let arch = if architecture.is_empty() {
        "llama".to_string()
    } else {
        architecture.clone()
    };

    let context_length = get_u64(&format!("{arch}.context_length"));
    let embedding_length = get_u64(&format!("{arch}.embedding_length"));
    let block_count = get_u64(&format!("{arch}.block_count"));
    let head_count = get_u64(&format!("{arch}.attention.head_count"));
    let head_count_kv = get_u64(&format!("{arch}.attention.head_count_kv"));

    // Estimate parameter count from tensor info.
    let param_count: u64 = content
        .tensor_infos
        .values()
        .map(|t| t.shape.elem_count() as u64)
        .sum();

    // Detect quantization from first large tensor's dtype.
    let quantization = content
        .tensor_infos
        .values()
        .filter(|t| t.shape.elem_count() > 1000)
        .map(|t| format!("{:?}", t.ggml_dtype))
        .next()
        .unwrap_or_else(|| "unknown".to_string());

    Ok(GgufMetadata {
        architecture,
        name,
        param_count,
        context_length,
        embedding_length,
        block_count,
        head_count,
        head_count_kv,
        quantization,
        file_size,
    })
}
