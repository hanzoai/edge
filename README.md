# Hanzo Edge

> Deploy AI across mobile, web, and embedded applications

[![Crates.io](https://img.shields.io/crates/v/hanzo-edge)](https://crates.io/crates/hanzo-edge)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue)](LICENSE)

On-device AI inference with zero cloud dependency. Run Zen models and any GGUF model locally on macOS, Linux, iOS, Android, Web (WASM), and embedded devices.

## Quick Start

```bash
# Install via curl
curl -sSL https://edge.hanzo.ai/install.sh | sh

# Or via cargo
cargo install hanzo-edge

# Run a model
hanzo-edge run --model zenlm/zen3-nano --prompt "Hello!"

# Start local API server
hanzo-edge serve --model zenlm/zen3-nano --port 8080
```

## Features

- **On-Device**: Zero network latency, works offline, full data privacy
- **Cross-Platform**: macOS, Linux, iOS, Android, Web (WASM), embedded
- **Hardware Acceleration**: Metal (Apple Silicon), CUDA (NVIDIA), CPU (AVX2/AVX-512)
- **GGUF Native**: First-class support for quantized models (Q4_K, Q5_K, Q8_0)
- **OpenAI Compatible**: Drop-in replacement API at localhost
- **Streaming**: Token-by-token streaming via SSE and callbacks
- **HuggingFace Hub**: Automatic model download and caching from any HF repo

## Crates

| Crate | Description | Install |
|-------|-------------|---------|
| `hanzo-edge-core` | Core inference runtime and `Model` trait | `cargo add hanzo-edge-core` |
| `hanzo-edge` | CLI binary with run, serve, bench, info | `cargo install hanzo-edge` |
| `hanzo-edge-wasm` | Browser WASM module with streaming | `wasm-pack build edge-wasm` |

## SDK Usage

### Rust

```rust
use hanzo_edge_core::{load_model, InferenceSession, SamplingParams, ModelConfig};

// Load from HuggingFace Hub (auto-downloads and caches)
let config = ModelConfig {
    model_id: "zenlm/zen3-nano".to_string(),
    model_file: Some("zen3-nano.Q4_K_M.gguf".to_string()),
    ..Default::default()
};
let (mut model, tokenizer) = load_model(&config)?;

// Generate
let params = SamplingParams {
    temperature: 0.7,
    top_p: 0.9,
    top_k: 40,
    max_tokens: 256,
    repeat_penalty: 1.1,
    repeat_last_n: 64,
};
let mut session = InferenceSession::new(&mut *model, &tokenizer, params);
let output = session.generate("Explain quantum computing")?;
println!("{}", output.text);

// Streaming
let stream = session.generate_stream("Write a haiku")?;
for token_result in stream {
    print!("{}", token_result?);
}
```

### CLI

```bash
# Run inference with streaming output
hanzo-edge run --model zenlm/zen3-nano --prompt "Write a haiku" \
    --max-tokens 128 --temperature 0.7 --top-p 0.9

# Model info (architecture, params, quantization, context length)
hanzo-edge info --model zenlm/zen3-nano

# Benchmark (TTFT, tokens/sec, memory, averaged over N iterations)
hanzo-edge bench --model zenlm/zen3-nano --prompt "Hello" \
    --max-tokens 128 -n 5

# Start OpenAI-compatible API server
hanzo-edge serve --model zenlm/zen3-nano --port 8080
```

### JavaScript (WASM)

```javascript
import init, { EdgeModel, get_version, get_device_info } from 'hanzo-edge-wasm';

await init();
console.log(`Hanzo Edge v${get_version()} [${get_device_info()}]`);

// Load model and tokenizer as ArrayBuffers
const modelBytes = await fetch('model.gguf').then(r => r.arrayBuffer());
const tokenizerBytes = await fetch('tokenizer.json').then(r => r.arrayBuffer());

const model = new EdgeModel(
    new Uint8Array(modelBytes),
    new Uint8Array(tokenizerBytes)
);

// Synchronous generation
const output = model.generate("Hello!", 256, 0.7);
console.log(output);

// Streaming (token-by-token callback)
model.generate_stream("Write a poem", 256, 0.7, (token) => {
    process.stdout.write(token);
});

// Reset KV cache between conversations
model.reset();
```

### Python (via local API)

```python
from openai import OpenAI

# Point at the local hanzo-edge server
client = OpenAI(base_url="http://localhost:8080/v1", api_key="local")

# Chat completions (streaming)
stream = client.chat.completions.create(
    model="zen3-nano",
    messages=[{"role": "user", "content": "Hello!"}],
    stream=True
)
for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")

# Text completions
response = client.completions.create(
    model="zen3-nano",
    prompt="The quick brown fox",
    max_tokens=64
)
print(response.choices[0].text)
```

## API Server

The built-in server is OpenAI-compatible:

```bash
hanzo-edge serve --model zenlm/zen3-nano --port 8080
```

Endpoints:

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/v1/chat/completions` | Chat completion (streaming + non-streaming) |
| `POST` | `/v1/completions` | Text completion |
| `GET` | `/v1/models` | List loaded models |
| `GET` | `/health` | Health check |

Chat completions support `stream: true` for server-sent events (SSE), with `[DONE]` sentinel and ChatML-formatted prompts.

## Zen Models for Edge

Pre-quantized models optimized for on-device inference:

| Model | Params | Memory | Use Case |
|-------|--------|--------|----------|
| zen-nano | 600M | ~400MB | Ultra-lightweight, embedded |
| zen-eco | 4B | ~2.5GB | General purpose, mobile |
| zen4-mini | 8B | ~5GB | High quality, desktop/laptop |

All available at [huggingface.co/zenlm](https://huggingface.co/zenlm) in GGUF format.

## Supported Model Architectures

Hanzo Edge supports any GGUF model using these architectures:

| Architecture | Examples |
|-------------|----------|
| Llama | Llama 2/3, Zen, Mistral, Yi |
| Qwen2 | Qwen 2/3, Zen 4 |
| Phi3 | Phi-3-mini, Phi-3-small |
| Gemma2 | Gemma 2 |
| SmolLM | SmolLM 135M-1.7B |

Architecture is auto-detected from GGUF metadata. Any GGUF file using the Llama-family tensor layout is supported.

## Platform Support

| Platform | Backend | Status |
|----------|---------|--------|
| macOS (Apple Silicon) | Metal | Production |
| macOS (Intel) | CPU/Accelerate | Production |
| Linux x86_64 | CPU/CUDA | Production |
| Linux ARM64 | CPU | Production |
| Web (WASM) | CPU | Beta |
| iOS | Metal/CoreML | Coming Soon |
| Android | Vulkan/NNAPI | Coming Soon |

## Feature Flags

| Feature | Description | Build |
|---------|-------------|-------|
| `cpu` | CPU backend (default) | `cargo build --release` |
| `metal` | Metal backend for macOS/iOS | `cargo build --release --features metal` |
| `cuda` | CUDA backend for NVIDIA GPUs | `cargo build --release --features cuda` |

## Building from Source

```bash
git clone https://github.com/hanzoai/edge
cd edge

# Build CLI (CPU)
cargo build --release -p hanzo-edge

# Build CLI (Metal, Apple Silicon)
cargo build --release -p hanzo-edge --features metal

# Build CLI (CUDA)
cargo build --release -p hanzo-edge --features cuda

# Build WASM
cd edge-wasm && cargo build --target wasm32-unknown-unknown --release
wasm-bindgen target/wasm32-unknown-unknown/release/edge_wasm.wasm \
    --out-dir pkg --target web

# Run tests
cargo test --workspace

# Lint
cargo clippy --workspace -- -D warnings

# Format
cargo fmt --all
```

## Architecture

```
hanzo-edge (workspace)
├── edge-core/              # Core inference runtime (library)
│   ├── lib.rs              # Public API: Model, InferenceSession, SamplingParams
│   ├── model.rs            # Model trait, GGUF loading, HF Hub download
│   ├── session.rs          # Autoregressive generation + streaming iterator
│   ├── sampling.rs         # Temperature, top-k, top-p, repeat penalty
│   └── tokenizer.rs        # HF tokenizer wrapper with EOS detection
├── edge-cli/               # CLI binary
│   ├── main.rs             # Clap-based CLI with 4 subcommands
│   ├── loader.rs           # HF Hub download with progress bars
│   └── cmd/
│       ├── run.rs          # Streaming inference to stdout
│       ├── serve.rs        # OpenAI-compatible HTTP server (Axum)
│       ├── bench.rs        # TTFT, throughput, memory benchmarking
│       └── info.rs         # Model metadata inspection
└── edge-wasm/              # WebAssembly module
    └── lib.rs              # WASM bindings: EdgeModel, generate, generate_stream
```

Built on [Hanzo ML](https://github.com/hanzoai/ml) (Candle) for tensor operations.

## Related

- [Hanzo Engine](https://engine.hanzo.ai) -- Cloud GPU inference (production serving)
- [Hanzo ML](https://github.com/hanzoai/ml) -- Rust ML framework
- [Zen Models](https://huggingface.co/zenlm) -- Pre-trained model weights
- [Hanzo AI](https://hanzo.ai) -- Full AI platform

## License

Apache-2.0
