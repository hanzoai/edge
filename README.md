# Hanzo Edge

> On-device AI inference -- deploy Zen models across mobile, web, and embedded applications

[![CI](https://github.com/hanzoai/edge/actions/workflows/ci.yml/badge.svg)](https://github.com/hanzoai/edge/actions/workflows/ci.yml)
[![WASM](https://github.com/hanzoai/edge/actions/workflows/wasm.yml/badge.svg)](https://github.com/hanzoai/edge/actions/workflows/wasm.yml)
[![Crates.io](https://img.shields.io/crates/v/hanzo-edge)](https://crates.io/crates/hanzo-edge)
[![Rust](https://img.shields.io/badge/rust-stable-orange)](https://www.rust-lang.org)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue)](LICENSE)

On-device AI inference with zero cloud dependency. Run Zen models and any GGUF model locally on macOS, Linux, iOS, Android, Web (WASM), and embedded devices. Full data privacy, zero network latency, works completely offline.

**[Full Documentation](https://docs.hanzo.ai/docs/services/edge)** | **[API Reference](https://edge.hanzo.ai)** | **[Zen Models](https://huggingface.co/zenlm)**

---

## Edge vs Engine

| | **Hanzo Edge** | **[Hanzo Engine](https://engine.hanzo.ai)** |
|---|---|---|
| **Where** | On-device (local CPU/GPU) | Cloud GPU clusters |
| **Latency** | Zero network overhead | Network round-trip |
| **Privacy** | Data never leaves device | Data sent to cloud |
| **Models** | Quantized GGUF (Q4/Q5/Q8) | Full-precision (FP16/BF16) |
| **Best for** | Mobile, embedded, offline, privacy | Production serving, large models, scale |

Use **Edge** when data privacy, offline capability, or minimal latency matter. Use **Engine** when you need full-precision models or high-throughput serving across many concurrent users.

---

## Quick Start

```bash
# Install via curl
curl -sSL https://edge.hanzo.ai/install.sh | sh

# Or via cargo
cargo install hanzo-edge

# Run a model (auto-downloads from HuggingFace)
hanzo-edge run --model zenlm/zen3-nano --prompt "Hello!"

# Start local OpenAI-compatible API server
hanzo-edge serve --model zenlm/zen3-nano --port 8080
```

### Docker

```bash
# Run with Docker (CPU)
docker run --rm -it ghcr.io/hanzoai/edge:latest \
    run --model zenlm/zen3-nano --prompt "Hello!"

# Serve as API (expose port 8080)
docker run --rm -p 8080:8080 ghcr.io/hanzoai/edge:latest \
    serve --model zenlm/zen3-nano --port 8080
```

## Features

- **On-Device**: Zero network latency, works offline, full data privacy
- **Cross-Platform**: macOS, Linux, iOS, Android, Web (WASM), embedded
- **Hardware Acceleration**: Metal (Apple Silicon), CUDA (NVIDIA), CPU (AVX2/AVX-512)
- **GGUF Native**: First-class support for quantized models (Q4_K, Q5_K, Q8_0)
- **OpenAI Compatible**: Drop-in replacement API at localhost
- **Streaming**: Token-by-token streaming via SSE and callbacks
- **HuggingFace Hub**: Automatic model download and caching from any HF repo

## Platform Support

| Platform | Backend | Status | Notes |
|----------|---------|--------|-------|
| macOS (Apple Silicon) | Metal | **Production** | M1/M2/M3/M4, hardware-accelerated |
| macOS (Intel) | CPU / Accelerate | **Production** | AVX2 optimized |
| Linux x86_64 | CPU | **Production** | AVX2/AVX-512 auto-detected |
| Linux x86_64 | CUDA | **Production** | NVIDIA GPUs, requires CUDA toolkit |
| Linux ARM64 | CPU | **Production** | Raspberry Pi, ARM servers |
| Web (WASM) | CPU | **Stable** | All modern browsers, WebAssembly SIMD |
| iOS | Metal / CoreML | Planned | |
| Android | Vulkan / NNAPI | Planned | |
| Embedded (ARM) | CPU | Experimental | Cortex-A class and above |

## Crates

| Crate | Description | Install |
|-------|-------------|---------|
| `hanzo-edge-core` | Core inference runtime and `Model` trait | `cargo add hanzo-edge-core` |
| `hanzo-edge` | CLI binary with run, serve, bench, info | `cargo install hanzo-edge` |
| `hanzo-edge-wasm` | Browser WASM module with streaming | `wasm-pack build edge-wasm` |

## Zen Models for Edge

Pre-quantized models optimized for on-device inference, available at [huggingface.co/zenlm](https://huggingface.co/zenlm) in GGUF format:

| Model | Params | Quantized Size | Use Case |
|-------|--------|----------------|----------|
| `zenlm/zen3-nano` | 600M | ~400MB (Q4_K_M) | Ultra-lightweight, embedded, IoT |
| `zenlm/zen-eco` | 4B | ~2.5GB (Q4_K_M) | General purpose, mobile, tablets |
| `zenlm/zen4-mini` | 8B | ~5GB (Q4_K_M) | High quality, desktop and laptop |

```bash
# Run the smallest model (embedded/IoT)
hanzo-edge run --model zenlm/zen3-nano --prompt "Summarize this report"

# General-purpose mobile model
hanzo-edge run --model zenlm/zen-eco --prompt "Draft an email response"

# High-quality desktop model
hanzo-edge run --model zenlm/zen4-mini --prompt "Explain quantum entanglement" \
    --max-tokens 512 --temperature 0.7
```

## SDK Usage

### Rust

```rust
use hanzo_edge_core::{load_model, InferenceSession, SamplingParams, ModelConfig};

// Load zen4-mini from HuggingFace Hub (auto-downloads and caches)
let config = ModelConfig {
    model_id: "zenlm/zen4-mini".to_string(),
    model_file: Some("zen4-mini.Q4_K_M.gguf".to_string()),
    ..Default::default()
};
let (mut model, tokenizer) = load_model(&config)?;

// Generate with sampling parameters
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

// Streaming (token-by-token)
let stream = session.generate_stream("Write a haiku about rust")?;
for token_result in stream {
    print!("{}", token_result?);
}
```

### CLI

```bash
# Streaming inference with zen-eco
hanzo-edge run --model zenlm/zen-eco --prompt "Write a haiku" \
    --max-tokens 128 --temperature 0.7 --top-p 0.9

# Model info (architecture, params, quantization, context length)
hanzo-edge info --model zenlm/zen4-mini

# Benchmark (TTFT, tokens/sec, memory, averaged over N iterations)
hanzo-edge bench --model zenlm/zen3-nano --prompt "Hello" \
    --max-tokens 128 -n 5

# Start OpenAI-compatible API server with zen4-mini
hanzo-edge serve --model zenlm/zen4-mini --port 8080
```

### Python (via local API)

```python
from openai import OpenAI

# Point at the local hanzo-edge server
client = OpenAI(base_url="http://localhost:8080/v1", api_key="local")

# Chat completions (streaming) with zen4-mini
stream = client.chat.completions.create(
    model="zen4-mini",
    messages=[{"role": "user", "content": "Explain edge computing in 3 sentences"}],
    stream=True
)
for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")

# Text completions with zen3-nano
response = client.completions.create(
    model="zen3-nano",
    prompt="The quick brown fox",
    max_tokens=64
)
print(response.choices[0].text)
```

## WebAssembly (WASM)

Hanzo Edge compiles to WebAssembly for in-browser inference. No server required -- models run entirely in the browser tab.

### Building the WASM Module

```bash
# Install wasm-pack
curl https://rustwasm.github.io/wasm-pack/installer/init.sh -sSf | sh

# Build for web
wasm-pack build edge-wasm --target web

# Output in edge-wasm/pkg/:
#   edge_wasm.js       - JavaScript bindings
#   edge_wasm_bg.wasm  - WebAssembly binary
#   edge_wasm.d.ts     - TypeScript declarations
```

### Browser Usage

```html
<script type="module">
import init, { EdgeModel, get_version, get_device_info } from './pkg/edge_wasm.js';

async function main() {
    await init();
    console.log(`Hanzo Edge v${get_version()} [${get_device_info()}]`);

    // Load a quantized Zen model (e.g., zen3-nano Q4_K_M)
    const modelBytes = await fetch('/models/zen3-nano.Q4_K_M.gguf')
        .then(r => r.arrayBuffer());
    const tokenizerBytes = await fetch('/models/tokenizer.json')
        .then(r => r.arrayBuffer());

    const model = new EdgeModel(
        new Uint8Array(modelBytes),
        new Uint8Array(tokenizerBytes)
    );

    // Synchronous generation
    const output = model.generate("Hello!", 256, 0.7);
    document.getElementById('output').textContent = output;

    // Streaming (token-by-token callback)
    model.generate_stream("Write a poem about the web", 256, 0.7, (token) => {
        document.getElementById('output').textContent += token;
    });

    // Reset KV cache between conversations
    model.reset();
}

main();
</script>
```

### JavaScript / TypeScript (Node or Bundler)

```javascript
import init, { EdgeModel, get_version, get_device_info } from 'hanzo-edge-wasm';

await init();
console.log(`Hanzo Edge v${get_version()} [${get_device_info()}]`);

// Load model and tokenizer as ArrayBuffers
const modelBytes = await fetch('zen3-nano.Q4_K_M.gguf').then(r => r.arrayBuffer());
const tokenizerBytes = await fetch('tokenizer.json').then(r => r.arrayBuffer());

const model = new EdgeModel(
    new Uint8Array(modelBytes),
    new Uint8Array(tokenizerBytes)
);

// Generate
const output = model.generate("Summarize this document", 512, 0.7);
console.log(output);

// Streaming
model.generate_stream("Explain WASM in simple terms", 256, 0.7, (token) => {
    process.stdout.write(token);
});

model.reset();
```

### WASM Considerations

- **Model size**: Use small quantized models for browser (zen3-nano at ~400MB, zen-eco at ~2.5GB)
- **Memory**: Browser tabs typically have 2-4GB memory limits; zen3-nano fits comfortably
- **Threading**: WASM runs single-threaded; expect lower throughput than native builds
- **Caching**: Use the Cache API or IndexedDB to persist downloaded model files across sessions
- **SIMD**: WebAssembly SIMD is supported in all modern browsers and is auto-detected

## API Server

The built-in server is OpenAI-compatible:

```bash
hanzo-edge serve --model zenlm/zen4-mini --port 8080
```

Endpoints:

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/v1/chat/completions` | Chat completion (streaming + non-streaming) |
| `POST` | `/v1/completions` | Text completion |
| `GET` | `/v1/models` | List loaded models |
| `GET` | `/health` | Health check |

Chat completions support `stream: true` for server-sent events (SSE), with `[DONE]` sentinel and ChatML-formatted prompts.

```bash
# Example: curl against local Edge server
curl http://localhost:8080/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "zen4-mini",
        "messages": [{"role": "user", "content": "Hello!"}],
        "stream": true
    }'
```

## Supported Model Architectures

Hanzo Edge supports any GGUF model using these architectures:

| Architecture | Zen Models |
|-------------|------------|
| Dense Transformer | zen3-nano (600M), zen-eco (4B), zen4-mini (8B) |
| MoDE (Mixture of Distilled Experts) | zen4, zen4-pro, zen4-max, zen4-coder |
| Grouped-Query Attention | zen3-vl, zen3-omni |
| Vision-Language | zen3-vl (multimodal) |

Architecture is auto-detected from GGUF metadata. Any GGUF file using standard tensor layouts is supported.

## Performance Benchmarks

Benchmarks measured with `hanzo-edge bench` on representative hardware. Values are tokens per second (tok/s) for generation.

| Model | Apple M3 Max (Metal) | Intel i9-13900K (CPU) | NVIDIA RTX 4090 (CUDA) |
|-------|---------------------|-----------------------|------------------------|
| zen3-nano (Q4_K_M) | -- | -- | -- |
| zen-eco (Q4_K_M) | -- | -- | -- |
| zen4-mini (Q4_K_M) | -- | -- | -- |

> Benchmarks are being collected across hardware configurations. Run your own with:
> ```bash
> hanzo-edge bench --model zenlm/zen3-nano --prompt "Hello" --max-tokens 256 -n 10
> ```
> Contributions of benchmark results are welcome via GitHub issues.

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
+-- edge-core/              # Core inference runtime (library)
|   +-- lib.rs              # Public API: Model, InferenceSession, SamplingParams
|   +-- model.rs            # Model trait, GGUF loading, HF Hub download
|   +-- session.rs          # Autoregressive generation + streaming iterator
|   +-- sampling.rs         # Temperature, top-k, top-p, repeat penalty
|   +-- tokenizer.rs        # HF tokenizer wrapper with EOS detection
+-- edge-cli/               # CLI binary
|   +-- main.rs             # Clap-based CLI with 4 subcommands
|   +-- loader.rs           # HF Hub download with progress bars
|   +-- cmd/
|       +-- run.rs          # Streaming inference to stdout
|       +-- serve.rs        # OpenAI-compatible HTTP server (Axum)
|       +-- bench.rs        # TTFT, throughput, memory benchmarking
|       +-- info.rs         # Model metadata inspection
+-- edge-wasm/              # WebAssembly module
    +-- lib.rs              # WASM bindings: EdgeModel, generate, generate_stream
```

Built on [Hanzo ML](https://github.com/hanzoai/ml) (Candle) for tensor operations.

## Related Projects

| Project | Description | Link |
|---------|-------------|------|
| **Hanzo Engine** | Cloud GPU inference for production serving at scale | [engine.hanzo.ai](https://engine.hanzo.ai) |
| **Hanzo Gateway** | Unified LLM proxy for 100+ providers (OpenAI, Anthropic, Zen, etc.) | [github.com/hanzoai/llm](https://github.com/hanzoai/llm) |
| **Hanzo Ingress** | Edge routing and load balancing for AI services | [github.com/hanzoai/ingress](https://github.com/hanzoai/ingress) |
| **Hanzo ML** | Rust ML framework (tensor ops, neural network layers) | [github.com/hanzoai/ml](https://github.com/hanzoai/ml) |
| **Zen Models** | Pre-trained Zen model weights (GGUF, safetensors) | [huggingface.co/zenlm](https://huggingface.co/zenlm) |
| **Hanzo AI** | Full AI platform and infrastructure | [hanzo.ai](https://hanzo.ai) |

## License

Apache-2.0
