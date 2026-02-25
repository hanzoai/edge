//! `hanzo-edge bench` — benchmark model inference.
//!
//! Reports: time to first token (TTFT), tokens/sec, memory, averaged over N iterations.

use std::time::Instant;

use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use candle_transformers::models::quantized_llama::ModelWeights;

use crate::loader;

/// Benchmark results for a single iteration.
struct IterResult {
    ttft_ms: f64,
    tokens_per_sec: f64,
    generated_tokens: usize,
    total_ms: f64,
}

/// Run benchmark across N iterations.
pub fn execute(
    model_id: &str,
    prompt: &str,
    max_tokens: usize,
    iterations: usize,
    revision: Option<&str>,
    device: &Device,
) -> Result<()> {
    eprintln!("Hanzo Edge Benchmark");
    eprintln!("====================");
    eprintln!("Model:      {model_id}");
    eprintln!("Prompt:     \"{prompt}\"");
    eprintln!("Max tokens: {max_tokens}");
    eprintln!("Iterations: {iterations}");
    eprintln!("Device:     {device:?}");
    eprintln!();

    // Load model once — reuse across iterations.
    let loaded = loader::load_model(model_id, revision, device)?;
    let tokenizer = loaded.tokenizer;

    let encoding = tokenizer
        .encode(prompt, true)
        .map_err(|e| anyhow::anyhow!("tokenizer encode error: {e}"))?;
    let prompt_tokens = encoding.get_ids().to_vec();

    if prompt_tokens.is_empty() {
        anyhow::bail!("prompt produced no tokens");
    }

    eprintln!("Prompt tokens: {}", prompt_tokens.len());

    // Measure RSS before inference.
    let rss_before = current_rss_bytes();

    let mut results = Vec::with_capacity(iterations);

    for iter in 0..iterations {
        eprint!("  Iteration {}/{}...", iter + 1, iterations);

        // Clone weights for each iteration by re-loading.
        // This ensures KV cache is reset between runs.
        let mut weights = if iter == 0 {
            loaded.weights.clone()
        } else {
            // Re-load from disk for clean KV cache.
            let reloaded = loader::load_model(model_id, revision, device)?;
            reloaded.weights
        };

        let result = bench_one_iteration(
            &mut weights,
            &prompt_tokens,
            max_tokens,
            device,
        )?;

        eprintln!(
            " TTFT={:.1}ms, {:.1} tok/s, {} tokens",
            result.ttft_ms, result.tokens_per_sec, result.generated_tokens
        );

        results.push(result);
    }

    let rss_after = current_rss_bytes();

    // Aggregate.
    let n = results.len() as f64;
    let avg_ttft = results.iter().map(|r| r.ttft_ms).sum::<f64>() / n;
    let avg_tps = results.iter().map(|r| r.tokens_per_sec).sum::<f64>() / n;
    let avg_total = results.iter().map(|r| r.total_ms).sum::<f64>() / n;
    let avg_tokens = results.iter().map(|r| r.generated_tokens).sum::<usize>() as f64 / n;

    let min_ttft = results
        .iter()
        .map(|r| r.ttft_ms)
        .fold(f64::INFINITY, f64::min);
    let max_ttft = results
        .iter()
        .map(|r| r.ttft_ms)
        .fold(f64::NEG_INFINITY, f64::max);

    let min_tps = results
        .iter()
        .map(|r| r.tokens_per_sec)
        .fold(f64::INFINITY, f64::min);
    let max_tps = results
        .iter()
        .map(|r| r.tokens_per_sec)
        .fold(f64::NEG_INFINITY, f64::max);

    println!();
    println!("=== Benchmark Results ===");
    println!();
    println!("Time to First Token (TTFT):");
    println!("  avg: {avg_ttft:.1}ms  min: {min_ttft:.1}ms  max: {max_ttft:.1}ms");
    println!();
    println!("Decode Throughput:");
    println!("  avg: {avg_tps:.1} tok/s  min: {min_tps:.1} tok/s  max: {max_tps:.1} tok/s");
    println!();
    println!("Avg generated tokens: {avg_tokens:.0}");
    println!("Avg total time:       {avg_total:.0}ms");
    println!();
    println!("Memory (RSS):");
    println!("  Before: {}", format_bytes(rss_before));
    println!("  After:  {}", format_bytes(rss_after));
    println!(
        "  Delta:  {}",
        format_bytes(rss_after.saturating_sub(rss_before))
    );

    Ok(())
}

/// Run a single benchmark iteration: prefill + decode.
fn bench_one_iteration(
    weights: &mut ModelWeights,
    prompt_tokens: &[u32],
    max_tokens: usize,
    device: &Device,
) -> Result<IterResult> {
    // Prefill.
    let t0 = Instant::now();
    let input = Tensor::new(prompt_tokens, device)
        .map_err(|e| anyhow::anyhow!("tensor: {e}"))?
        .unsqueeze(0)
        .map_err(|e| anyhow::anyhow!("unsqueeze: {e}"))?;
    let logits = weights
        .forward(&input, 0)
        .map_err(|e| anyhow::anyhow!("prefill forward: {e}"))?;
    let first_token = greedy_sample(&logits)?;
    let ttft = t0.elapsed();

    // Decode.
    let decode_start = Instant::now();
    let mut generated = 1usize;
    let mut prev = first_token;

    for i in 1..max_tokens {
        let pos = prompt_tokens.len() + i - 1;
        let input = Tensor::new(&[prev], device)
            .map_err(|e| anyhow::anyhow!("tensor: {e}"))?
            .unsqueeze(0)
            .map_err(|e| anyhow::anyhow!("unsqueeze: {e}"))?;
        let logits = weights
            .forward(&input, pos)
            .map_err(|e| anyhow::anyhow!("decode forward: {e}"))?;
        prev = greedy_sample(&logits)?;
        generated += 1;
    }

    let decode_elapsed = decode_start.elapsed();
    let total = t0.elapsed();

    let tps = if decode_elapsed.as_secs_f64() > 0.0 && generated > 1 {
        (generated - 1) as f64 / decode_elapsed.as_secs_f64()
    } else {
        0.0
    };

    Ok(IterResult {
        ttft_ms: ttft.as_secs_f64() * 1000.0,
        tokens_per_sec: tps,
        generated_tokens: generated,
        total_ms: total.as_secs_f64() * 1000.0,
    })
}

/// Greedy (argmax) sampling — deterministic for benchmarks.
fn greedy_sample(logits: &Tensor) -> Result<u32> {
    let logits = logits
        .squeeze(0)
        .map_err(|e| anyhow::anyhow!("squeeze: {e}"))?;
    let logits = if logits.dims().len() == 2 {
        logits
            .get(logits.dim(0).map_err(|e| anyhow::anyhow!("{e}"))? - 1)
            .map_err(|e| anyhow::anyhow!("get last: {e}"))?
    } else {
        logits
    };
    let logits = logits
        .to_dtype(DType::F32)
        .map_err(|e| anyhow::anyhow!("dtype: {e}"))?;
    let token = logits
        .argmax(0)
        .map_err(|e| anyhow::anyhow!("argmax: {e}"))?
        .to_scalar::<u32>()
        .map_err(|e| anyhow::anyhow!("scalar: {e}"))?;
    Ok(token)
}

/// Get current process RSS in bytes (platform-specific).
fn current_rss_bytes() -> u64 {
    #[cfg(target_os = "macos")]
    {
        macos_rss()
    }
    #[cfg(target_os = "linux")]
    {
        linux_rss()
    }
    #[cfg(not(any(target_os = "macos", target_os = "linux")))]
    {
        0
    }
}

#[cfg(target_os = "macos")]
fn macos_rss() -> u64 {
    use std::mem;
    unsafe {
        let mut info: libc::mach_task_basic_info_data_t = mem::zeroed();
        let mut count = (mem::size_of::<libc::mach_task_basic_info_data_t>()
            / mem::size_of::<libc::natural_t>()) as libc::mach_msg_type_number_t;
        let kr = libc::task_info(
            libc::mach_task_self(),
            libc::MACH_TASK_BASIC_INFO,
            &mut info as *mut _ as libc::task_info_t,
            &mut count,
        );
        if kr == libc::KERN_SUCCESS {
            info.resident_size as u64
        } else {
            0
        }
    }
}

#[cfg(target_os = "linux")]
fn linux_rss() -> u64 {
    // Read from /proc/self/statm — field 1 is RSS in pages.
    std::fs::read_to_string("/proc/self/statm")
        .ok()
        .and_then(|s| {
            s.split_whitespace()
                .nth(1)
                .and_then(|v| v.parse::<u64>().ok())
        })
        .map(|pages| pages * 4096)
        .unwrap_or(0)
}

fn format_bytes(bytes: u64) -> String {
    if bytes == 0 {
        return "N/A".to_string();
    }
    if bytes >= 1_073_741_824 {
        format!("{:.2} GiB", bytes as f64 / 1_073_741_824.0)
    } else if bytes >= 1_048_576 {
        format!("{:.1} MiB", bytes as f64 / 1_048_576.0)
    } else {
        format!("{:.0} KiB", bytes as f64 / 1024.0)
    }
}
