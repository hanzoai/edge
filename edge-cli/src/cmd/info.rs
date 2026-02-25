//! `hanzo-edge info` — print model metadata.

use std::path::Path;

use anyhow::{Context, Result};
use candle_core::quantized::gguf_file;

use crate::loader;

/// Print model info to stdout.
pub fn execute(model_id: &str, revision: Option<&str>) -> Result<()> {
    let (gguf_path, _tok_path) = loader::resolve_model(model_id, revision)?;
    let meta = loader::read_gguf_metadata(&gguf_path)?;

    println!("Hanzo Edge - Model Info");
    println!("=======================");
    println!();
    println!("Model ID:       {model_id}");
    if !meta.name.is_empty() {
        println!("Name:           {}", meta.name);
    }
    println!(
        "Architecture:   {}",
        if meta.architecture.is_empty() {
            "unknown"
        } else {
            &meta.architecture
        }
    );
    println!("Parameters:     {}", format_params(meta.param_count));
    println!("Context length: {}", meta.context_length);
    println!("Embedding dim:  {}", meta.embedding_length);
    println!("Layers:         {}", meta.block_count);
    println!(
        "Attention heads: {} (KV: {})",
        meta.head_count, meta.head_count_kv
    );
    println!("Quantization:   {}", meta.quantization);
    println!("File size:      {}", format_bytes(meta.file_size));
    println!("File:           {}", gguf_path.display());

    // Print additional raw metadata keys if verbose.
    println!();
    println!("GGUF Metadata Keys");
    println!("-------------------");
    print_raw_metadata(&gguf_path)?;

    Ok(())
}

/// Print all metadata key-value pairs from the GGUF header.
fn print_raw_metadata(path: &Path) -> Result<()> {
    let mut file =
        std::fs::File::open(path).with_context(|| format!("cannot open {}", path.display()))?;
    let content = gguf_file::Content::read(&mut file)
        .map_err(|e| anyhow::anyhow!("failed to read GGUF: {e}"))?;

    let mut keys: Vec<&String> = content.metadata.keys().collect();
    keys.sort();

    for key in keys {
        let val = &content.metadata[key];
        let display = format_value(val);
        // Truncate very long values.
        if display.len() > 120 {
            println!("  {key}: {}...", &display[..117]);
        } else {
            println!("  {key}: {display}");
        }
    }

    Ok(())
}

/// Format a GGUF metadata value for display.
fn format_value(val: &gguf_file::Value) -> String {
    if let Ok(s) = val.to_string() {
        return format!("\"{s}\"");
    }
    if let Ok(v) = val.to_u32() {
        return v.to_string();
    }
    if let Ok(v) = val.to_i32() {
        return v.to_string();
    }
    if let Ok(v) = val.to_f32() {
        return format!("{v}");
    }
    if let Ok(v) = val.to_u64() {
        return v.to_string();
    }
    if let Ok(v) = val.to_bool() {
        return v.to_string();
    }
    if let Ok(arr) = val.to_vec() {
        return format!("[{} items]", arr.len());
    }
    format!("{:?}", val.value_type())
}

/// Format a parameter count for human display.
fn format_params(count: u64) -> String {
    if count >= 1_000_000_000 {
        format!("{:.1}B", count as f64 / 1e9)
    } else if count >= 1_000_000 {
        format!("{:.1}M", count as f64 / 1e6)
    } else if count >= 1_000 {
        format!("{:.0}K", count as f64 / 1e3)
    } else {
        count.to_string()
    }
}

/// Format bytes for human display.
fn format_bytes(bytes: u64) -> String {
    if bytes >= 1_073_741_824 {
        format!("{:.2} GiB", bytes as f64 / 1_073_741_824.0)
    } else if bytes >= 1_048_576 {
        format!("{:.1} MiB", bytes as f64 / 1_048_576.0)
    } else if bytes >= 1024 {
        format!("{:.0} KiB", bytes as f64 / 1024.0)
    } else {
        format!("{bytes} B")
    }
}
