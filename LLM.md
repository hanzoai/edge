# LLM.md - Hanzo Edge

## Overview
> On-device AI inference -- deploy Zen models across mobile, web, and embedded applications

## Tech Stack
- **Language**: Rust

## Build & Run
```bash
cargo build
cargo test
```

## Structure
```
edge/
  Cargo.lock
  Cargo.toml
  Dockerfile
  LLM.md
  Makefile
  README.md
  docs/
  edge-cli/
  edge-core/
  edge-wasm/
```

## Key Files
- `README.md` -- Project documentation
- `Cargo.toml` -- Rust crate config
- `Makefile` -- Build automation
- `Dockerfile` -- Container build
