.PHONY: build build-metal build-wasm test clean check fmt lint

build:
	cargo build --release

build-metal:
	cargo build --release --features metal

build-wasm:
	cd edge-wasm && cargo build --target wasm32-unknown-unknown --release
	wasm-bindgen target/wasm32-unknown-unknown/release/edge_wasm.wasm --out-dir pkg --target web

test:
	cargo test

check:
	cargo check --workspace

fmt:
	cargo fmt --all

lint:
	cargo clippy --workspace -- -D warnings

clean:
	cargo clean
	rm -rf pkg/
