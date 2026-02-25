# Hanzo Edge

On-device AI inference runtime for mobile, web, and embedded platforms.

Built on [Hanzo ML](https://github.com/hanzoai/ml) (Candle) and sharing model formats with [Hanzo Engine](https://github.com/hanzoai/engine).

## Targets

| Platform | Backend | Status |
|----------|---------|--------|
| macOS/iOS | Metal (AFQ 4-bit) | In Development |
| Linux/Android | CPU / Vulkan | Planned |
| Web | WASM + WebGPU | In Development |
| Embedded | CPU (ARM/RISC-V) | Planned |

## Quick Start

```bash
# Build
cargo build --release

# Run inference
cargo run --release -p edge-cli -- run --model zen3-nano --prompt "Hello world"

# Build WASM
make build-wasm
```

## Architecture

```
+-------------------------------------+
|           Hanzo Edge                 |
|  +----------+  +------------------+ |
|  | edge-cli |  | edge-wasm (JS)   | |
|  +----+-----+  +--------+---------+ |
|       +----------+-------+          |
|         +----+-----+                |
|         |edge-core |                |
|         +----+-----+                |
|         +----+-----+                |
|         | Hanzo ML |  (Candle)      |
|         +----------+                |
+-------------------------------------+
```

## Crates

| Crate | Description |
|-------|-------------|
| `edge-core` | Core inference runtime, model/session traits |
| `edge-cli` | Command-line interface for testing |
| `edge-wasm` | WebAssembly bindings for browser inference |

## Feature Flags

| Feature | Description |
|---------|-------------|
| `cpu` | CPU backend (default) |
| `metal` | Metal backend for macOS/iOS |
| `cuda` | CUDA backend for NVIDIA GPUs |

## Related

- [Hanzo Engine](https://github.com/hanzoai/engine) -- Cloud inference (GPU servers)
- [Hanzo ML](https://github.com/hanzoai/ml) -- Rust ML framework
- [Hanzo Ingress](https://github.com/hanzoai/ingress) -- Reverse proxy
- [Hanzo Gateway](https://github.com/hanzoai/gateway) -- API gateway
