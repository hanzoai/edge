//! Inference session: drives token-by-token autoregressive generation.

use anyhow::{bail, Result};
use candle_core::Tensor;

use crate::model::Model;
use crate::sampling::{sample_token, SamplingParams};
use crate::tokenizer::TokenizerWrapper;
use crate::GenerateOutput;

/// Drives autoregressive generation from a model + tokenizer pair.
pub struct InferenceSession<'a> {
    model: &'a mut dyn Model,
    tokenizer: &'a TokenizerWrapper,
    params: SamplingParams,
}

impl<'a> InferenceSession<'a> {
    pub fn new(
        model: &'a mut dyn Model,
        tokenizer: &'a TokenizerWrapper,
        params: SamplingParams,
    ) -> Self {
        Self {
            model,
            tokenizer,
            params,
        }
    }

    /// Generate text for the given prompt.
    ///
    /// Tokenizes the prompt, runs the prefill + decode loop,
    /// and returns the full output including prompt and generation stats.
    pub fn generate(&mut self, prompt: &str) -> Result<GenerateOutput> {
        let prompt_tokens = self.tokenizer.encode(prompt, true)?;
        if prompt_tokens.is_empty() {
            bail!("prompt must not be empty");
        }

        let generated_ids = self.generate_from_tokens(&prompt_tokens)?;

        let text = self.tokenizer.decode(&generated_ids, true)?;

        Ok(GenerateOutput {
            text,
            tokens: generated_ids.clone(),
            prompt_tokens: prompt_tokens.len(),
            generated_tokens: generated_ids.len(),
        })
    }

    /// Streaming generation: yields one decoded token string at a time.
    ///
    /// Returns an iterator of `Result<String>` where each item is
    /// the text fragment for a single generated token.
    pub fn generate_stream<'b>(
        &'b mut self,
        prompt: &str,
    ) -> Result<StreamIter<'b, 'a>> {
        let prompt_tokens = self.tokenizer.encode(prompt, true)?;
        if prompt_tokens.is_empty() {
            bail!("prompt must not be empty");
        }

        // Prefill: process entire prompt at once.
        let device = self.model.device().clone();
        let input = Tensor::new(prompt_tokens.as_slice(), &device)?.unsqueeze(0)?;
        let logits = self.model.forward(&input, 0)?;
        let first_token = sample_token(&logits, &self.params, &prompt_tokens)?;

        let mut all_tokens = prompt_tokens.clone();
        all_tokens.push(first_token);

        Ok(StreamIter {
            session: self,
            all_tokens,
            prompt_len: prompt_tokens.len(),
            pos: prompt_tokens.len(), // next decode position
            first_token: Some(first_token),
            done: false,
        })
    }

    /// Low-level generation from pre-tokenized input.
    fn generate_from_tokens(&mut self, prompt_tokens: &[u32]) -> Result<Vec<u32>> {
        let device = self.model.device().clone();
        let mut all_tokens = prompt_tokens.to_vec();
        let mut generated = Vec::new();

        // Prefill: full prompt in one forward pass.
        let input = Tensor::new(prompt_tokens, &device)?.unsqueeze(0)?;
        let logits = self.model.forward(&input, 0)?;
        let mut next_token = sample_token(&logits, &self.params, &all_tokens)?;
        all_tokens.push(next_token);
        generated.push(next_token);

        if self.tokenizer.is_eos(next_token) {
            return Ok(generated);
        }

        // Decode loop: one token at a time.
        for i in 1..self.params.max_tokens {
            let pos = prompt_tokens.len() + i - 1;
            if pos >= self.model.max_seq_len() {
                tracing::warn!(pos, max = self.model.max_seq_len(), "reached max sequence length");
                break;
            }

            let input = Tensor::new(&[next_token], &device)?.unsqueeze(0)?;
            let logits = self.model.forward(&input, pos)?;
            next_token = sample_token(&logits, &self.params, &all_tokens)?;

            if self.tokenizer.is_eos(next_token) {
                break;
            }

            all_tokens.push(next_token);
            generated.push(next_token);
        }

        Ok(generated)
    }
}

/// Iterator that yields one decoded token string per step.
pub struct StreamIter<'b, 'a: 'b> {
    session: &'b mut InferenceSession<'a>,
    all_tokens: Vec<u32>,
    prompt_len: usize,
    pos: usize,
    first_token: Option<u32>,
    done: bool,
}

impl<'b, 'a: 'b> Iterator for StreamIter<'b, 'a> {
    type Item = Result<String>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.done {
            return None;
        }

        let generated_count = self.all_tokens.len() - self.prompt_len;
        if generated_count >= self.session.params.max_tokens {
            self.done = true;
            return None;
        }

        // Yield the first token from prefill if we have it.
        if let Some(tok) = self.first_token.take() {
            if self.session.tokenizer.is_eos(tok) {
                self.done = true;
                return None;
            }
            return Some(self.session.tokenizer.decode_one(tok));
        }

        // Decode next token.
        let last_token = match self.all_tokens.last() {
            Some(&t) => t,
            None => {
                self.done = true;
                return None;
            }
        };

        if self.pos >= self.session.model.max_seq_len() {
            self.done = true;
            return None;
        }

        let device = self.session.model.device().clone();
        let input = match Tensor::new(&[last_token], &device).and_then(|t| t.unsqueeze(0)) {
            Ok(t) => t,
            Err(e) => {
                self.done = true;
                return Some(Err(anyhow::anyhow!("tensor creation: {e}")));
            }
        };

        let logits = match self.session.model.forward(&input, self.pos) {
            Ok(l) => l,
            Err(e) => {
                self.done = true;
                return Some(Err(e));
            }
        };

        let next_token =
            match sample_token(&logits, &self.session.params, &self.all_tokens) {
                Ok(t) => t,
                Err(e) => {
                    self.done = true;
                    return Some(Err(e));
                }
            };

        self.pos += 1;

        if self.session.tokenizer.is_eos(next_token) {
            self.done = true;
            return None;
        }

        self.all_tokens.push(next_token);
        Some(self.session.tokenizer.decode_one(next_token))
    }
}

/// Convenience: encode prompt, generate, decode, return output.
pub fn generate_text(
    model: &mut dyn Model,
    tokenizer: &TokenizerWrapper,
    prompt: &str,
    params: SamplingParams,
) -> Result<GenerateOutput> {
    let mut session = InferenceSession::new(model, tokenizer, params);
    session.generate(prompt)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sampling_params_default_is_sane() {
        let p = SamplingParams::default();
        assert!(p.temperature > 0.0);
        assert!(p.top_p > 0.0 && p.top_p <= 1.0);
        assert!(p.max_tokens > 0);
        assert!(p.top_k > 0);
    }
}
