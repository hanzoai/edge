//! `hanzo-edge run` — run inference, streaming tokens to stdout.

use std::io::Write;
use std::time::Instant;

use anyhow::Result;
use candle_core::{Device, Tensor};

use crate::loader;

/// Run inference with streaming token output.
pub fn execute(
    model_id: &str,
    prompt: &str,
    max_tokens: usize,
    temperature: f64,
    top_p: f64,
    revision: Option<&str>,
    device: &Device,
) -> Result<()> {
    let loaded = loader::load_model(model_id, revision, device)?;
    let mut weights = loaded.weights;
    let tokenizer = loaded.tokenizer;

    // Encode prompt.
    let encoding = tokenizer
        .encode(prompt, true)
        .map_err(|e| anyhow::anyhow!("tokenizer encode error: {e}"))?;
    let prompt_tokens = encoding.get_ids();

    if prompt_tokens.is_empty() {
        anyhow::bail!("prompt produced no tokens");
    }

    eprintln!(
        "Model:  {}\nTokens: {} prompt tokens\n",
        model_id,
        prompt_tokens.len()
    );

    let mut stdout = std::io::stdout().lock();

    // --- Prefill phase ---
    let t_start = Instant::now();
    let input = Tensor::new(prompt_tokens, device)?.unsqueeze(0)?;
    let logits = weights
        .forward(&input, 0)
        .map_err(|e| anyhow::anyhow!("forward error: {e}"))?;

    let first_token = sample_token(&logits, temperature, top_p)?;
    let ttft = t_start.elapsed();

    // Print first generated token.
    if let Some(text) = decode_token(&tokenizer, first_token) {
        write!(stdout, "{text}")?;
        stdout.flush()?;
    }

    // --- Decode phase ---
    let decode_start = Instant::now();
    let mut generated_count: usize = 1;
    let mut prev_token = first_token;

    for i in 1..max_tokens {
        let pos = prompt_tokens.len() + i - 1;
        let input = Tensor::new(&[prev_token], device)?.unsqueeze(0)?;
        let logits = weights
            .forward(&input, pos)
            .map_err(|e| anyhow::anyhow!("forward error at pos {pos}: {e}"))?;

        let token = sample_token(&logits, temperature, top_p)?;
        generated_count += 1;

        // Check for common EOS tokens.
        if is_eos(&tokenizer, token) {
            break;
        }

        if let Some(text) = decode_token(&tokenizer, token) {
            write!(stdout, "{text}")?;
            stdout.flush()?;
        }

        prev_token = token;
    }

    // Trailing newline after generation.
    writeln!(stdout)?;

    let decode_elapsed = decode_start.elapsed();
    let total_elapsed = t_start.elapsed();
    let tokens_per_sec = if decode_elapsed.as_secs_f64() > 0.0 {
        (generated_count - 1) as f64 / decode_elapsed.as_secs_f64()
    } else {
        0.0
    };

    eprintln!();
    eprintln!("--- generation stats ---");
    eprintln!("Prompt tokens:    {}", prompt_tokens.len());
    eprintln!("Generated tokens: {generated_count}");
    eprintln!("Time to first:    {:.1}ms", ttft.as_secs_f64() * 1000.0);
    eprintln!("Decode speed:     {tokens_per_sec:.1} tok/s");
    eprintln!("Total time:       {:.2}s", total_elapsed.as_secs_f64());

    Ok(())
}

/// Sample a single token from logits with temperature + top-p.
fn sample_token(logits: &Tensor, temperature: f64, top_p: f64) -> Result<u32> {
    use candle_core::DType;

    // logits shape: [1, seq_len, vocab] or [1, vocab] — take last position.
    let logits = logits.squeeze(0)?;
    let logits = if logits.dims().len() == 2 {
        logits.get(logits.dim(0)? - 1)?
    } else {
        logits
    };
    let logits = logits.to_dtype(DType::F32)?;

    // Greedy.
    if temperature < 1e-7 {
        let token = logits
            .argmax(0)
            .map_err(|e| anyhow::anyhow!("argmax: {e}"))?
            .to_scalar::<u32>()
            .map_err(|e| anyhow::anyhow!("to_scalar: {e}"))?;
        return Ok(token);
    }

    // Temperature scaling.
    let logits = (&logits / temperature)
        .map_err(|e| anyhow::anyhow!("temperature scaling: {e}"))?;
    let probs = candle_nn::ops::softmax(&logits, 0)
        .map_err(|e| anyhow::anyhow!("softmax: {e}"))?;
    let probs_vec: Vec<f32> = probs
        .to_vec1()
        .map_err(|e| anyhow::anyhow!("to_vec1: {e}"))?;

    // Top-p nucleus sampling.
    let mut indexed: Vec<(usize, f32)> = probs_vec.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let mut cumsum = 0.0f32;
    let mut cutoff = indexed.len();
    for (i, (_, p)) in indexed.iter().enumerate() {
        cumsum += p;
        if cumsum >= top_p as f32 {
            cutoff = i + 1;
            break;
        }
    }

    let candidates = &indexed[..cutoff];
    let total: f32 = candidates.iter().map(|(_, p)| p).sum();
    let r = rand_float() * total;

    let mut acc = 0.0f32;
    for &(idx, p) in candidates {
        acc += p;
        if acc >= r {
            return Ok(idx as u32);
        }
    }

    Ok(candidates.last().map(|(idx, _)| *idx as u32).unwrap_or(0))
}

/// Decode a single token to text.
fn decode_token(tokenizer: &tokenizers::Tokenizer, token: u32) -> Option<String> {
    tokenizer.decode(&[token], false).ok()
}

/// Check if a token is an end-of-sequence marker.
fn is_eos(tokenizer: &tokenizers::Tokenizer, token: u32) -> bool {
    // Check the tokenizer's configured EOS.
    if let Some(id) = tokenizer
        .get_added_vocabulary()
        .get_vocab()
        .get("</s>")
    {
        if token == *id {
            return true;
        }
    }
    if let Some(id) = tokenizer
        .get_added_vocabulary()
        .get_vocab()
        .get("<|endoftext|>")
    {
        if token == *id {
            return true;
        }
    }
    if let Some(id) = tokenizer
        .get_added_vocabulary()
        .get_vocab()
        .get("<|im_end|>")
    {
        if token == *id {
            return true;
        }
    }
    false
}

/// Simple pseudo-random float in [0, 1).
fn rand_float() -> f32 {
    use std::time::SystemTime;
    let nanos = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap_or_default()
        .subsec_nanos();
    (nanos % 10000) as f32 / 10000.0
}
