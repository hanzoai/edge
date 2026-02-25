//! Hanzo Edge WASM -- browser-side AI inference via WebAssembly.
//!
//! Loads GGUF quantized models and tokenizers from `ArrayBuffer`,
//! runs autoregressive generation on `Device::Cpu`, and exposes
//! both synchronous and streaming (JS callback) APIs.
//!
//! Constraints:
//! - No filesystem access.
//! - No async runtime (no tokio).
//! - CPU only (no Metal / CUDA in WASM).

use std::io::Cursor;

use candle_core::quantized::gguf_file;
use candle_core::{DType, Device, Tensor};
use candle_transformers::generation::LogitsProcessor;
use candle_transformers::models::quantized_qwen3::ModelWeights;
use tokenizers::Tokenizer;
use wasm_bindgen::prelude::*;

// ---------------------------------------------------------------------------
// console.log bridge
// ---------------------------------------------------------------------------

#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = console)]
    fn log(s: &str);
}

macro_rules! console_log {
    ($($t:tt)*) => { log(&format_args!($($t)*).to_string()) }
}

// ---------------------------------------------------------------------------
// EdgeModel -- the public WASM API
// ---------------------------------------------------------------------------

/// A quantized LLM loaded entirely in WASM memory.
///
/// Create with `EdgeModel::new`, then call `generate` or `generate_stream`.
#[wasm_bindgen]
pub struct EdgeModel {
    model: ModelWeights,
    tokenizer: Tokenizer,
    eos_token: u32,
}

#[wasm_bindgen]
impl EdgeModel {
    /// Load a GGUF model and a HuggingFace `tokenizer.json` from byte arrays.
    ///
    /// Both arguments originate from `fetch()` → `ArrayBuffer` on the JS side.
    #[wasm_bindgen(constructor)]
    pub fn new(model_bytes: Vec<u8>, tokenizer_bytes: Vec<u8>) -> Result<EdgeModel, JsError> {
        console_error_panic_hook::set_once();

        let device = Device::Cpu;

        // -- tokenizer --
        console_log!("loading tokenizer ({} bytes)", tokenizer_bytes.len());
        let tokenizer = Tokenizer::from_bytes(&tokenizer_bytes)
            .map_err(|e| JsError::new(&format!("tokenizer: {e}")))?;

        let vocab = tokenizer.get_vocab(true);
        let eos_token = vocab
            .get("<|endoftext|>")
            .or_else(|| vocab.get("<|im_end|>"))
            .or_else(|| vocab.get("</s>"))
            .copied()
            .unwrap_or(0);

        // -- GGUF model --
        console_log!(
            "loading GGUF model ({:.2} MB)",
            model_bytes.len() as f64 / 1_048_576.0
        );
        let mut cursor = Cursor::new(model_bytes);
        let content = gguf_file::Content::read(&mut cursor)
            .map_err(|e| JsError::new(&format!("gguf parse: {e}")))?;

        let model = ModelWeights::from_gguf(content, &mut cursor, &device)
            .map_err(|e| JsError::new(&format!("model load: {e}")))?;

        console_log!("model ready (eos_token={})", eos_token);

        Ok(EdgeModel {
            model,
            tokenizer,
            eos_token,
        })
    }

    /// Synchronous text generation.
    ///
    /// Returns the full generated string. Blocks the calling thread
    /// (use a Web Worker to avoid freezing the UI).
    #[wasm_bindgen]
    pub fn generate(
        &mut self,
        prompt: &str,
        max_tokens: u32,
        temperature: f64,
    ) -> Result<String, JsError> {
        let (tokens, _) = self
            .run_generation(prompt, max_tokens, temperature, None)
            .map_err(|e| JsError::new(&e.to_string()))?;
        Ok(tokens)
    }

    /// Streaming generation -- calls `callback(token_string)` for every
    /// decoded token, then returns the concatenated result.
    #[wasm_bindgen]
    pub fn generate_stream(
        &mut self,
        prompt: &str,
        max_tokens: u32,
        temperature: f64,
        callback: &js_sys::Function,
    ) -> Result<String, JsError> {
        let (text, _) = self
            .run_generation(prompt, max_tokens, temperature, Some(callback))
            .map_err(|e| JsError::new(&e.to_string()))?;
        Ok(text)
    }

    /// Reset the KV cache so the next generation starts fresh.
    #[wasm_bindgen]
    pub fn reset(&mut self) {
        self.model.clear_kv_cache();
    }
}

// ---------------------------------------------------------------------------
// Internal generation loop
// ---------------------------------------------------------------------------

impl EdgeModel {
    /// Shared generation core.
    ///
    /// If `callback` is `Some`, each decoded token is passed to JS.
    /// Returns `(full_text, token_count)`.
    fn run_generation(
        &mut self,
        prompt: &str,
        max_tokens: u32,
        temperature: f64,
        callback: Option<&js_sys::Function>,
    ) -> anyhow::Result<(String, usize)> {
        // Reset KV cache for a fresh generation.
        self.model.clear_kv_cache();

        // Encode prompt.
        let encoding = self
            .tokenizer
            .encode(prompt, true)
            .map_err(|e| anyhow::anyhow!("encode: {e}"))?;
        let prompt_ids: Vec<u32> = encoding.get_ids().to_vec();

        if prompt_ids.is_empty() {
            anyhow::bail!("prompt produced zero tokens");
        }

        let device = Device::Cpu;

        // Build logits processor (temperature + top-p).
        let temp = if temperature < 1e-7 {
            None
        } else {
            Some(temperature)
        };
        let mut logits_processor = LogitsProcessor::new(
            /* seed */ js_sys::Date::now() as u64,
            temp,
            Some(0.9), // top-p
        );

        // -- prefill: feed the entire prompt at once --
        let input = Tensor::new(prompt_ids.as_slice(), &device)?.unsqueeze(0)?;
        let logits = self.model.forward(&input, 0)?;
        let logits = logits.squeeze(0)?.to_dtype(DType::F32)?;
        let mut next_token = logits_processor.sample(&logits)?;

        let mut output = String::new();
        let mut count: usize = 0;

        // Emit first token.
        if next_token != self.eos_token {
            let piece = self.decode_token(next_token);
            Self::maybe_callback(callback, &piece)?;
            output.push_str(&piece);
            count += 1;
        }

        // -- decode loop --
        for i in 0..(max_tokens as usize).saturating_sub(1) {
            if next_token == self.eos_token {
                break;
            }

            let pos = prompt_ids.len() + i;
            let input = Tensor::new(&[next_token], &device)?.unsqueeze(0)?;
            let logits = self.model.forward(&input, pos)?;
            let logits = logits.squeeze(0)?.to_dtype(DType::F32)?;
            next_token = logits_processor.sample(&logits)?;

            if next_token == self.eos_token {
                break;
            }

            let piece = self.decode_token(next_token);
            Self::maybe_callback(callback, &piece)?;
            output.push_str(&piece);
            count += 1;
        }

        Ok((output, count))
    }

    /// Decode a single token ID to a string.
    fn decode_token(&self, id: u32) -> String {
        self.tokenizer
            .decode(&[id], /* skip_special */ false)
            .unwrap_or_default()
    }

    /// If a JS callback is provided, invoke it with the token string.
    fn maybe_callback(cb: Option<&js_sys::Function>, text: &str) -> anyhow::Result<()> {
        if let Some(f) = cb {
            let this = JsValue::NULL;
            let arg = JsValue::from_str(text);
            f.call1(&this, &arg)
                .map_err(|e| anyhow::anyhow!("callback error: {e:?}"))?;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Free functions
// ---------------------------------------------------------------------------

/// Return the crate version.
#[wasm_bindgen]
pub fn get_version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}

/// Return the compute device identifier.
///
/// Always `"wasm-cpu"` in this build -- WebGPU support is future work.
#[wasm_bindgen]
pub fn get_device_info() -> String {
    "wasm-cpu".to_string()
}

/// Initialise the WASM panic hook. Idempotent.
#[wasm_bindgen]
pub fn init() {
    console_error_panic_hook::set_once();
}

// ---------------------------------------------------------------------------
// Tests (native only -- `cargo test` on host)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn version_is_set() {
        let v = get_version();
        assert!(!v.is_empty());
        assert!(v.starts_with("0."));
    }

    #[test]
    fn device_info_is_wasm_cpu() {
        assert_eq!(get_device_info(), "wasm-cpu");
    }
}
