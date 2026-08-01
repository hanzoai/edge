<p align="center"><img src=".github/hero.svg" alt="edge" width="880"></p>

# Hanzo Edge

Hanzo Edge is an on-device inference runtime. It loads a quantized GGUF model
off local disk and generates tokens on the machine it is running on — a laptop,
a phone, a browser tab, an ARM board. Nothing is sent anywhere. There is no
network call at inference time and no API key.

It is three things in one workspace: a Rust library (`hanzo-edge-core`), a CLI
that wraps it (`hanzo-edge`), and a WebAssembly module for the browser
(`hanzo-edge-wasm`). Weights and math come from [candle](https://github.com/huggingface/candle);
tokenization from `tokenizers`.

[![License](https://img.shields.io/badge/license-Apache--2.0-blue)](LICENSE)

## Which "edge" this is

The word is overloaded inside Hanzo. This repository is exactly one of the five
and none of the others:

| | What it is | Where it lives |
|---|---|---|
| **Hanzo Edge** — this repo | On-device inference runtime. A binary and a library you run locally. | `hanzoai/edge` |
| the edge worker | A Cloudflare Worker that serves the model catalog. | `hanzoai/catalog` |
| the cloud edge | CORS and rate limiting in front of the API. Not a product. | `cloud/apps/gateway` |
| `/v1/edge/nodes` | Edge routers in the zero-trust fabric. A `zt` resource, live on the API. | `zt` |
| edge audiences | A list of JWT audiences. A config value. | IAM |

If you are reading a document that says "edge" and it is not about running a
model on the device in front of you, it is not this.

Related and distinct: **Hanzo Engine** serves models from cloud GPUs, at
full precision, for many concurrent users. Edge trades that away for privacy,
offline operation and no network round-trip. Different jobs.

## Build

Edge is not on crates.io yet, and there is no published container image. Build
it from source:

```bash
git clone https://github.com/hanzoai/edge
cd edge
cargo build --release -p hanzo-edge                    # CPU
cargo build --release -p hanzo-edge --features metal   # Apple Silicon
cargo build --release -p hanzo-edge --features cuda    # NVIDIA
```

The binary lands at `target/release/hanzo-edge`. `make build`, `make build-metal`
and `make build-wasm` are the same commands.

The `v0.1.0` release carries prebuilt tarballs for `darwin-amd64`, `darwin-arm64`
and `linux-amd64`. `https://edge.hanzo.ai/install.sh` fetches them, but it
composes the asset name from `uname -m` output (`x86_64` / `aarch64`) while the
assets are published as `amd64` / `arm64`, so it currently resolves a 404 on
every platform. Build from source until that is fixed.

## Run a model

Point it at a `.gguf` file with a `tokenizer.json` beside it:

```bash
hanzo-edge run --model ./models/zen-nano-0.6b-Q4_K_M.gguf --prompt "Hello!"
```

Four subcommands, all taking `--model`:

```bash
hanzo-edge run   --model M --prompt "..." [--max-tokens 256] [--temperature 0.7] [--top-p 0.9]
hanzo-edge info  --model M                       # architecture, params, quantization, context length
hanzo-edge bench --model M --prompt "..." [-n 5] # tokens/sec over N iterations
hanzo-edge serve --model M --port 8080           # local HTTP server
```

`run` streams to stdout as tokens are produced.

## Models

Edge loads GGUF and only GGUF. Any GGUF using standard tensor layouts works —
the architecture is read from the file's own metadata, so there is nothing to
configure.

Zen GGUF builds are published at [huggingface.co/zenlm](https://huggingface.co/zenlm).
The ones that carry GGUF today:

| Repo | File |
|---|---|
| `zenlm/zen-nano-0.6b` | `gguf/zen-nano-0.6b-{Q4_K_M,Q5_K_M,Q8_0}.gguf` |
| `zenlm/zen-eco-4b-instruct` | `gguf/zen-eco-4b-instruct-f16.gguf` |
| `zenlm/zen-eco-4b-thinking` | `gguf/zen-eco-4b-thinking-f16.gguf` |
| `zenlm/zen-agent-4b` | `gguf/zen-eco-4b-agent-*.gguf` |
| `zenlm/zen-embedding-0.6B-GGUF`, `zenlm/zen-reranker-0.6B-GGUF` | root-level `.gguf` |

Passing a bare repo id (`--model zenlm/zen-nano-0.6b`) does not work for these
yet. The hub resolver tries a short list of root-level names — `<repo>.gguf`,
`model.gguf`, `model-q4_k_m.gguf` and a few more — and the Zen builds live in a
`gguf/` subdirectory under longer names. Download the file and pass its path:

```bash
hf download zenlm/zen-nano-0.6b gguf/zen-nano-0.6b-Q4_K_M.gguf tokenizer.json --local-dir ./zen-nano
hanzo-edge run --model ./zen-nano/gguf/zen-nano-0.6b-Q4_K_M.gguf --prompt "Hello!"
```

Widening that resolver is tracked work; until then, a local path is the
supported route.

## Serving locally

`hanzo-edge serve` brings up an HTTP server on the loopback interface with the
same paths as the Hanzo AI API, so client code written against `api.hanzo.ai`
works unchanged once you repoint the base URL at `localhost`:

| Method | Path |
|---|---|
| `POST` | `/v1/chat/completions` |
| `POST` | `/v1/completions` |
| `GET` | `/v1/models` |
| `GET` | `/health` |

Chat completions honour `"stream": true` and emit server-sent events terminated
by a `[DONE]` sentinel. Prompts are formatted as ChatML.

```bash
hanzo-edge serve --model ./zen-nano/gguf/zen-nano-0.6b-Q4_K_M.gguf --port 8080

curl http://localhost:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"zen-nano","messages":[{"role":"user","content":"Hello!"}],"stream":true}'
```

There is no auth on this server. It is meant for loopback; do not bind it to a
public interface.

## Library

```rust
use hanzo_edge_core::{load_model, InferenceSession, SamplingParams, ModelConfig};

let config = ModelConfig {
    model_id: "./zen-nano/gguf/zen-nano-0.6b-Q4_K_M.gguf".to_string(),
    ..Default::default()
};
let (mut model, tokenizer) = load_model(&config)?;

let params = SamplingParams {
    temperature: 0.7,
    top_p: 0.9,
    top_k: 40,
    max_tokens: 256,
    repeat_penalty: 1.1,
    repeat_last_n: 64,
};
let mut session = InferenceSession::new(&mut *model, &tokenizer, params);

println!("{}", session.generate("Explain quantum computing")?.text);

for token in session.generate_stream("Write a haiku about rust")? {
    print!("{}", token?);
}
```

`ModelConfig::model_id` takes either a local `.gguf` path or a hub repo id;
`model_file` names the file within a repo when the repo holds more than one.

## Browser

`hanzo-edge-wasm` compiles the same runtime to WebAssembly. The model runs in
the tab — there is no server.

```bash
make build-wasm      # or: wasm-pack build edge-wasm --target web
```

It is built from this workspace and is not published to npm; import it from the
`pkg/` directory `wasm-pack` writes.

```js
import init, { EdgeModel, get_version, get_device_info } from './pkg/edge_wasm.js';

await init();
console.log(`Hanzo Edge v${get_version()} [${get_device_info()}]`);

const model = new EdgeModel(
  new Uint8Array(await fetch('/models/zen-nano-0.6b-Q4_K_M.gguf').then(r => r.arrayBuffer())),
  new Uint8Array(await fetch('/models/tokenizer.json').then(r => r.arrayBuffer())),
);

model.generate_stream('Explain WebAssembly', 256, 0.7, t => output.append(t));
model.reset();   // clear the KV cache between conversations
```

Practical limits: the whole model is held in the tab's memory, so a browser
build wants the smallest quantization you can accept — `zen-nano-0.6b` at
`Q4_K_M` is the one that fits comfortably. WASM runs single-threaded and is
slower than the native build. Cache the downloaded `.gguf` in IndexedDB or the
Cache API so it survives a reload.

## Hardware backends

Selected at compile time, one cargo feature each:

| Feature | Backend |
|---|---|
| `cpu` (default) | Portable CPU |
| `metal` | Metal, on Apple Silicon |
| `cuda` | CUDA, on NVIDIA GPUs — needs the CUDA toolkit |

The WASM target runs on CPU only. iOS and Android are not implemented; the
Rust core builds for those targets but no platform bindings ship in this repo.

## Development

```bash
make test     # cargo test --workspace
make lint     # cargo clippy --workspace -- -D warnings
make fmt      # cargo fmt --all
```

Layout:

```
edge-core/   inference runtime — Model trait, GGUF loading, sessions, sampling, tokenizer
edge-cli/    the hanzo-edge binary — run, info, bench, serve
edge-wasm/   wasm-bindgen wrapper around edge-core
```

Deeper notes for people working on the internals: [LLM.md](LLM.md).

## License

Apache-2.0 — see [LICENSE](LICENSE).
