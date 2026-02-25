//! Inference session: drives token-by-token generation.

use anyhow::{bail, Result};
use candle_core::{DType, Tensor};
use serde::{Deserialize, Serialize};

use crate::model::Model;
use crate::GenerateOutput;

/// Sampling parameters for text generation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SamplingParams {
    pub temperature: f64,
    pub top_p: f64,
    pub max_tokens: usize,
    pub repeat_penalty: f32,
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self {
            temperature: 0.7,
            top_p: 0.9,
            max_tokens: 256,
            repeat_penalty: 1.1,
        }
    }
}

/// Drives token-by-token generation from a model.
pub struct InferenceSession {
    params: SamplingParams,
}

impl InferenceSession {
    pub fn new(params: SamplingParams) -> Self {
        Self { params }
    }

    /// Generate text from a sequence of prompt token IDs.
    ///
    /// Returns generated token IDs. The caller is responsible for
    /// encoding the prompt and decoding the output via a tokenizer.
    pub fn generate(
        &self,
        model: &mut dyn Model,
        prompt_tokens: &[u32],
    ) -> Result<Vec<u32>> {
        if prompt_tokens.is_empty() {
            bail!("prompt must not be empty");
        }

        let device = model.device().clone();
        let mut all_tokens = prompt_tokens.to_vec();
        let mut generated = Vec::new();

        // Prefill: process the entire prompt.
        let input = Tensor::new(prompt_tokens, &device)?.unsqueeze(0)?;
        let logits = model.forward(&input, 0)?;
        let next_token = self.sample_token(&logits)?;
        all_tokens.push(next_token);
        generated.push(next_token);

        // Decode: generate one token at a time.
        for i in 1..self.params.max_tokens {
            let pos = prompt_tokens.len() + i - 1;
            if pos >= model.max_seq_len() {
                break;
            }

            let input = Tensor::new(&[next_token], &device)?.unsqueeze(0)?;
            let logits = model.forward(&input, pos)?;
            let next_token = self.sample_token(&logits)?;

            all_tokens.push(next_token);
            generated.push(next_token);

            // TODO: check for EOS token
        }

        Ok(generated)
    }

    /// Sample a single token from logits using temperature + top-p.
    fn sample_token(&self, logits: &Tensor) -> Result<u32> {
        let logits = logits.squeeze(0)?.to_dtype(DType::F32)?;
        let dim = logits.dim(0)?;

        // Greedy for temperature ~0.
        if self.params.temperature < 1e-7 {
            let token = logits
                .argmax(0)?
                .to_scalar::<u32>()?;
            return Ok(token);
        }

        // Temperature scaling.
        let logits = (&logits / self.params.temperature)?;
        let probs = candle_nn::ops::softmax(&logits, 0)?;
        let probs_vec: Vec<f32> = probs.to_vec1()?;

        // Simple top-p nucleus sampling.
        let mut indexed: Vec<(usize, f32)> = probs_vec.iter().copied().enumerate().collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        let mut cumsum = 0.0;
        let mut cutoff = indexed.len();
        for (i, (_, p)) in indexed.iter().enumerate() {
            cumsum += p;
            if cumsum >= self.params.top_p as f32 {
                cutoff = i + 1;
                break;
            }
        }

        let candidates = &indexed[..cutoff];
        let total: f32 = candidates.iter().map(|(_, p)| p).sum();
        let r: f32 = rand_float() * total;

        let mut acc = 0.0;
        for &(idx, p) in candidates {
            acc += p;
            if acc >= r {
                return Ok(idx as u32);
            }
        }

        Ok(candidates.last().map(|(idx, _)| *idx as u32).unwrap_or(0))
    }
}

/// Simple pseudo-random float [0, 1) without pulling in rand crate.
fn rand_float() -> f32 {
    // Use a timestamp-based seed for minimal randomness.
    // In production, use a proper RNG.
    use std::time::SystemTime;
    let nanos = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap_or_default()
        .subsec_nanos();
    (nanos % 10000) as f32 / 10000.0
}

/// Convenience function: encode prompt, generate, decode.
pub fn generate_text(
    model: &mut dyn Model,
    tokenizer: &tokenizers::Tokenizer,
    prompt: &str,
    params: SamplingParams,
) -> Result<GenerateOutput> {
    let encoding = tokenizer
        .encode(prompt, true)
        .map_err(|e| anyhow::anyhow!("tokenizer encode error: {e}"))?;
    let prompt_tokens: Vec<u32> = encoding.get_ids().to_vec();

    let session = InferenceSession::new(params);
    let generated_ids = session.generate(model, &prompt_tokens)?;

    let text = tokenizer
        .decode(&generated_ids, true)
        .map_err(|e| anyhow::anyhow!("tokenizer decode error: {e}"))?;

    Ok(GenerateOutput {
        text,
        tokens: generated_ids.clone(),
        prompt_tokens: prompt_tokens.len(),
        generated_tokens: generated_ids.len(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sampling_params_default() {
        let p = SamplingParams::default();
        assert!(p.temperature > 0.0);
        assert!(p.top_p > 0.0 && p.top_p <= 1.0);
        assert!(p.max_tokens > 0);
    }
}
