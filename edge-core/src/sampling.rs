//! Token sampling strategies: temperature, top-k, top-p, repeat penalty.

use anyhow::{Context, Result};
use candle_core::{DType, Tensor};
use rand::Rng;
use serde::{Deserialize, Serialize};

/// Parameters controlling token sampling during generation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SamplingParams {
    /// Temperature for logit scaling. 0.0 = greedy.
    pub temperature: f64,
    /// Nucleus sampling threshold.
    pub top_p: f64,
    /// Top-k filtering. 0 = disabled.
    pub top_k: usize,
    /// Maximum tokens to generate.
    pub max_tokens: usize,
    /// Penalty applied to already-generated tokens.
    pub repeat_penalty: f32,
    /// Window of recent tokens to apply repeat penalty over.
    pub repeat_last_n: usize,
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self {
            temperature: 0.7,
            top_p: 0.9,
            top_k: 40,
            max_tokens: 256,
            repeat_penalty: 1.1,
            repeat_last_n: 64,
        }
    }
}

/// Sample a single token from a 1-D logits tensor.
///
/// Applies repeat penalty, temperature scaling, top-k filtering,
/// and top-p (nucleus) sampling in that order.
pub fn sample_token(
    logits: &Tensor,
    params: &SamplingParams,
    previous_tokens: &[u32],
) -> Result<u32> {
    // Squeeze to 1-D and cast to f32 for numerical stability.
    let logits = logits.squeeze(0)?.to_dtype(DType::F32)?;
    let mut logits_vec: Vec<f32> = logits
        .to_vec1()
        .context("failed to extract logits to vec")?;

    // --- Repeat penalty ---
    if params.repeat_penalty != 1.0 && !previous_tokens.is_empty() {
        let start = previous_tokens.len().saturating_sub(params.repeat_last_n);
        let recent = &previous_tokens[start..];
        for &tok in recent {
            let idx = tok as usize;
            if idx < logits_vec.len() {
                let score = logits_vec[idx];
                logits_vec[idx] = if score > 0.0 {
                    score / params.repeat_penalty
                } else {
                    score * params.repeat_penalty
                };
            }
        }
    }

    // --- Greedy (temperature ~ 0) ---
    if params.temperature < 1e-7 {
        let (best_idx, _) = logits_vec
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .context("empty logits")?;
        return Ok(best_idx as u32);
    }

    // --- Temperature scaling ---
    let inv_temp = 1.0 / params.temperature as f32;
    for v in logits_vec.iter_mut() {
        *v *= inv_temp;
    }

    // Build (index, logit) pairs sorted descending by logit.
    let mut indexed: Vec<(usize, f32)> = logits_vec.iter().copied().enumerate().collect();
    indexed.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    // --- Top-k filtering ---
    if params.top_k > 0 && params.top_k < indexed.len() {
        indexed.truncate(params.top_k);
    }

    // --- Softmax over candidates ---
    let max_logit = indexed[0].1;
    let mut probs: Vec<(usize, f32)> = indexed
        .iter()
        .map(|&(i, l)| (i, (l - max_logit).exp()))
        .collect();
    let sum: f32 = probs.iter().map(|(_, p)| p).sum();
    for (_, p) in probs.iter_mut() {
        *p /= sum;
    }

    // --- Top-p (nucleus) filtering ---
    if params.top_p < 1.0 {
        let mut cumsum = 0.0f32;
        let mut cutoff = probs.len();
        for (i, &(_, p)) in probs.iter().enumerate() {
            cumsum += p;
            if cumsum >= params.top_p as f32 {
                cutoff = i + 1;
                break;
            }
        }
        probs.truncate(cutoff);
    }

    // --- Weighted random selection ---
    let total: f32 = probs.iter().map(|(_, p)| p).sum();
    let mut rng = rand::thread_rng();
    let r: f32 = rng.gen::<f32>() * total;

    let mut acc = 0.0f32;
    for &(idx, p) in &probs {
        acc += p;
        if acc >= r {
            return Ok(idx as u32);
        }
    }

    // Fallback: last candidate.
    Ok(probs.last().map(|(idx, _)| *idx as u32).unwrap_or(0))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn greedy_picks_max() {
        let device = candle_core::Device::Cpu;
        let logits = Tensor::new(&[1.0f32, 5.0, 3.0, 2.0], &device).unwrap();
        let params = SamplingParams {
            temperature: 0.0,
            ..Default::default()
        };
        let tok = sample_token(&logits, &params, &[]).unwrap();
        assert_eq!(tok, 1); // index of 5.0
    }

    #[test]
    fn repeat_penalty_suppresses() {
        let device = candle_core::Device::Cpu;
        // Token 1 has highest logit but is penalized.
        let logits = Tensor::new(&[4.9f32, 5.0, 0.0, 0.0], &device).unwrap();
        let params = SamplingParams {
            temperature: 0.0,
            repeat_penalty: 100.0,
            repeat_last_n: 10,
            ..Default::default()
        };
        let tok = sample_token(&logits, &params, &[1]).unwrap();
        assert_eq!(tok, 0); // token 0 wins after penalty on token 1
    }
}
