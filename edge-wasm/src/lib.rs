//! Hanzo Edge WASM — browser inference via WebAssembly.
//!
//! Exports a minimal API for running inference in the browser.
//! Uses Hanzo ML (Candle) compiled to WASM.

use wasm_bindgen::prelude::*;

/// Initialize the WASM runtime. Call once before generate().
#[wasm_bindgen]
pub fn init() {
    console_error_panic_hook::set_once();
}

/// Run inference on the given prompt and return generated text.
///
/// This is a scaffold — actual model loading requires fetching
/// weights via fetch() and constructing the model in WASM memory.
#[wasm_bindgen]
pub fn generate(prompt: &str) -> String {
    // TODO: implement actual inference pipeline:
    // 1. Load model weights (pre-fetched into ArrayBuffer)
    // 2. Build candle model on Device::Cpu (WASM)
    // 3. Tokenize prompt
    // 4. Run forward pass + sampling loop
    // 5. Decode and return text
    format!("[edge-wasm scaffold] prompt received: {prompt}")
}

/// Load model weights from a JavaScript ArrayBuffer.
///
/// Call this after init() and before generate() to load
/// a safetensors model into WASM memory.
#[wasm_bindgen]
pub fn load_model(_weights: &[u8], _tokenizer_json: &str) -> Result<(), JsValue> {
    // TODO: deserialize safetensors, build model graph
    Ok(())
}

/// Return the runtime version.
#[wasm_bindgen]
pub fn version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}
