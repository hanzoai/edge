//! Tokenizer wrapper around HuggingFace `tokenizers` crate.
//!
//! Handles loading from HF Hub, encoding text to token IDs,
//! decoding token IDs back to text, and EOS detection.

use anyhow::Result;
use tokenizers::Tokenizer;

/// Wraps a HuggingFace tokenizer with convenience methods and EOS tracking.
pub struct TokenizerWrapper {
    inner: Tokenizer,
    bos_token_id: Option<u32>,
    eos_token_id: Option<u32>,
}

impl TokenizerWrapper {
    /// Build from a `tokenizers::Tokenizer` instance.
    ///
    /// Attempts to resolve BOS/EOS token IDs from the tokenizer vocabulary
    /// if they exist.
    pub fn new(tokenizer: Tokenizer) -> Self {
        let bos_token_id = tokenizer.token_to_id("<s>");
        // Try common EOS token names.
        let eos_token_id = tokenizer
            .token_to_id("</s>")
            .or_else(|| tokenizer.token_to_id("<|endoftext|>"))
            .or_else(|| tokenizer.token_to_id("<|end|>"))
            .or_else(|| tokenizer.token_to_id("<|im_end|>"));
        Self {
            inner: tokenizer,
            bos_token_id,
            eos_token_id,
        }
    }

    /// Load a tokenizer from a local `tokenizer.json` file.
    pub fn from_file(path: &str) -> Result<Self> {
        let tokenizer = Tokenizer::from_file(path)
            .map_err(|e| anyhow::anyhow!("failed to load tokenizer from {path}: {e}"))?;
        Ok(Self::new(tokenizer))
    }

    /// Override the EOS token ID (useful when GGUF metadata specifies it).
    pub fn set_eos_token_id(&mut self, id: u32) {
        self.eos_token_id = Some(id);
    }

    /// Override the BOS token ID.
    pub fn set_bos_token_id(&mut self, id: u32) {
        self.bos_token_id = Some(id);
    }

    /// Encode text to token IDs. If `add_special` is true,
    /// the tokenizer adds BOS/EOS per its configuration.
    pub fn encode(&self, text: &str, add_special: bool) -> Result<Vec<u32>> {
        let encoding = self
            .inner
            .encode(text, add_special)
            .map_err(|e| anyhow::anyhow!("tokenizer encode error: {e}"))?;
        Ok(encoding.get_ids().to_vec())
    }

    /// Decode token IDs back to text, optionally skipping special tokens.
    pub fn decode(&self, tokens: &[u32], skip_special: bool) -> Result<String> {
        self.inner
            .decode(tokens, skip_special)
            .map_err(|e| anyhow::anyhow!("tokenizer decode error: {e}"))
    }

    /// Decode a single token to its string representation.
    pub fn decode_one(&self, token: u32) -> Result<String> {
        self.decode(&[token], true)
    }

    /// Check whether a token is the end-of-sequence marker.
    pub fn is_eos(&self, token: u32) -> bool {
        self.eos_token_id.map_or(false, |eos| token == eos)
    }

    pub fn bos_token_id(&self) -> Option<u32> {
        self.bos_token_id
    }

    pub fn eos_token_id(&self) -> Option<u32> {
        self.eos_token_id
    }

    /// Access the underlying tokenizer.
    pub fn inner(&self) -> &Tokenizer {
        &self.inner
    }
}
