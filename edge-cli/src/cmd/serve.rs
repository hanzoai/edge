//! `hanzo-edge serve` — local OpenAI-compatible HTTP server.
//!
//! Endpoints:
//!   POST /v1/chat/completions
//!   POST /v1/completions
//!   GET  /v1/models

use std::sync::{Arc, Mutex};

use anyhow::Result;
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Json};
use axum::routing::{get, post};
use axum::Router;
use candle_core::{DType, Device, Tensor};
use candle_transformers::models::quantized_llama::ModelWeights;
use serde::{Deserialize, Serialize};
use tokio::sync::oneshot;
use tokenizers::Tokenizer;

use crate::loader;

// ---------------------------------------------------------------------------
// Shared server state
// ---------------------------------------------------------------------------

struct ServerState {
    weights: Mutex<ModelWeights>,
    tokenizer: Tokenizer,
    device: Device,
    model_id: String,
}

// ---------------------------------------------------------------------------
// OpenAI-compatible request/response types
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
struct ChatCompletionRequest {
    #[serde(default = "default_model")]
    model: String,
    messages: Vec<ChatMessage>,
    #[serde(default = "default_max_tokens")]
    max_tokens: Option<usize>,
    #[serde(default)]
    temperature: Option<f64>,
    #[serde(default)]
    top_p: Option<f64>,
    #[serde(default)]
    stream: Option<bool>,
}

#[derive(Debug, Deserialize)]
struct CompletionRequest {
    #[serde(default = "default_model")]
    model: String,
    prompt: String,
    #[serde(default = "default_max_tokens")]
    max_tokens: Option<usize>,
    #[serde(default)]
    temperature: Option<f64>,
    #[serde(default)]
    top_p: Option<f64>,
    #[serde(default)]
    stream: Option<bool>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ChatMessage {
    role: String,
    content: String,
}

#[derive(Serialize)]
struct ChatCompletionResponse {
    id: String,
    object: String,
    created: u64,
    model: String,
    choices: Vec<ChatChoice>,
    usage: Usage,
}

#[derive(Serialize)]
struct ChatChoice {
    index: usize,
    message: ChatMessage,
    finish_reason: String,
}

#[derive(Serialize)]
struct CompletionResponse {
    id: String,
    object: String,
    created: u64,
    model: String,
    choices: Vec<CompletionChoice>,
    usage: Usage,
}

#[derive(Serialize)]
struct CompletionChoice {
    index: usize,
    text: String,
    finish_reason: String,
}

#[derive(Serialize)]
struct Usage {
    prompt_tokens: usize,
    completion_tokens: usize,
    total_tokens: usize,
}

#[derive(Serialize)]
struct ModelsResponse {
    object: String,
    data: Vec<ModelEntry>,
}

#[derive(Serialize)]
struct ModelEntry {
    id: String,
    object: String,
    created: u64,
    owned_by: String,
}

// Stream chunk types for SSE.
#[derive(Serialize)]
struct ChatCompletionChunk {
    id: String,
    object: String,
    created: u64,
    model: String,
    choices: Vec<ChatChunkChoice>,
}

#[derive(Serialize)]
struct ChatChunkChoice {
    index: usize,
    delta: ChatDelta,
    finish_reason: Option<String>,
}

#[derive(Serialize)]
struct ChatDelta {
    #[serde(skip_serializing_if = "Option::is_none")]
    role: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<String>,
}

fn default_model() -> String {
    "default".to_string()
}

fn default_max_tokens() -> Option<usize> {
    Some(256)
}

fn unix_timestamp() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

fn request_id() -> String {
    format!("edge-{:x}", unix_timestamp())
}

// ---------------------------------------------------------------------------
// Server entry point
// ---------------------------------------------------------------------------

pub async fn execute(
    model_id: &str,
    port: u16,
    revision: Option<&str>,
    device: &Device,
) -> Result<()> {
    eprintln!("Loading model {model_id}...");

    let loaded = loader::load_model(model_id, revision, device)?;

    let state = Arc::new(ServerState {
        weights: Mutex::new(loaded.weights),
        tokenizer: loaded.tokenizer,
        device: device.clone(),
        model_id: model_id.to_string(),
    });

    let app = Router::new()
        .route("/v1/chat/completions", post(handle_chat_completions))
        .route("/v1/completions", post(handle_completions))
        .route("/v1/models", get(handle_models))
        .route("/health", get(handle_health))
        .with_state(state);

    let addr = format!("0.0.0.0:{port}");
    eprintln!();
    eprintln!("Hanzo Edge server listening on http://{addr}");
    eprintln!("  POST /v1/chat/completions");
    eprintln!("  POST /v1/completions");
    eprintln!("  GET  /v1/models");
    eprintln!();

    let listener = tokio::net::TcpListener::bind(&addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}

// ---------------------------------------------------------------------------
// Handlers
// ---------------------------------------------------------------------------

async fn handle_health() -> &'static str {
    "ok"
}

async fn handle_models(
    State(state): State<Arc<ServerState>>,
) -> Json<ModelsResponse> {
    Json(ModelsResponse {
        object: "list".to_string(),
        data: vec![ModelEntry {
            id: state.model_id.clone(),
            object: "model".to_string(),
            created: unix_timestamp(),
            owned_by: "hanzo".to_string(),
        }],
    })
}

async fn handle_chat_completions(
    State(state): State<Arc<ServerState>>,
    Json(req): Json<ChatCompletionRequest>,
) -> Result<impl IntoResponse, (StatusCode, String)> {
    let max_tokens = req.max_tokens.unwrap_or(256);
    let temperature = req.temperature.unwrap_or(0.7);
    let top_p = req.top_p.unwrap_or(0.9);
    let stream = req.stream.unwrap_or(false);

    // Build a text prompt from chat messages.
    let prompt = format_chat_prompt(&req.messages);

    if stream {
        return Ok(handle_chat_stream(state, prompt, max_tokens, temperature, top_p).await.into_response());
    }

    // Non-streaming: generate all tokens, return complete response.
    let (text, prompt_tok_count, gen_tok_count) =
        generate_blocking(state.clone(), &prompt, max_tokens, temperature, top_p)
            .await
            .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    let resp = ChatCompletionResponse {
        id: request_id(),
        object: "chat.completion".to_string(),
        created: unix_timestamp(),
        model: state.model_id.clone(),
        choices: vec![ChatChoice {
            index: 0,
            message: ChatMessage {
                role: "assistant".to_string(),
                content: text,
            },
            finish_reason: "stop".to_string(),
        }],
        usage: Usage {
            prompt_tokens: prompt_tok_count,
            completion_tokens: gen_tok_count,
            total_tokens: prompt_tok_count + gen_tok_count,
        },
    };

    Ok(Json(resp).into_response())
}

async fn handle_completions(
    State(state): State<Arc<ServerState>>,
    Json(req): Json<CompletionRequest>,
) -> Result<Json<CompletionResponse>, (StatusCode, String)> {
    let max_tokens = req.max_tokens.unwrap_or(256);
    let temperature = req.temperature.unwrap_or(0.7);
    let top_p = req.top_p.unwrap_or(0.9);

    let (text, prompt_tok_count, gen_tok_count) =
        generate_blocking(state.clone(), &req.prompt, max_tokens, temperature, top_p)
            .await
            .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    Ok(Json(CompletionResponse {
        id: request_id(),
        object: "text_completion".to_string(),
        created: unix_timestamp(),
        model: state.model_id.clone(),
        choices: vec![CompletionChoice {
            index: 0,
            text,
            finish_reason: "stop".to_string(),
        }],
        usage: Usage {
            prompt_tokens: prompt_tok_count,
            completion_tokens: gen_tok_count,
            total_tokens: prompt_tok_count + gen_tok_count,
        },
    }))
}

// ---------------------------------------------------------------------------
// Streaming SSE handler for chat completions
// ---------------------------------------------------------------------------

async fn handle_chat_stream(
    state: Arc<ServerState>,
    prompt: String,
    max_tokens: usize,
    temperature: f64,
    top_p: f64,
) -> Sse<impl futures_core::Stream<Item = Result<Event, std::convert::Infallible>>> {
    let (tx, rx) = tokio::sync::mpsc::channel::<Result<Event, std::convert::Infallible>>(64);

    let id = request_id();
    let model_id = state.model_id.clone();

    // Spawn blocking inference on a thread.
    tokio::task::spawn_blocking(move || {
        let run = || -> Result<()> {
            let encoding = state.tokenizer
                .encode(prompt.as_str(), true)
                .map_err(|e| anyhow::anyhow!("encode: {e}"))?;
            let prompt_tokens = encoding.get_ids();
            let device = &state.device;

            let mut weights = state.weights.lock().map_err(|e| anyhow::anyhow!("lock: {e}"))?;

            // Send role delta.
            let chunk = ChatCompletionChunk {
                id: id.clone(),
                object: "chat.completion.chunk".to_string(),
                created: unix_timestamp(),
                model: model_id.clone(),
                choices: vec![ChatChunkChoice {
                    index: 0,
                    delta: ChatDelta {
                        role: Some("assistant".to_string()),
                        content: None,
                    },
                    finish_reason: None,
                }],
            };
            let data = serde_json::to_string(&chunk)?;
            let _ = tx.blocking_send(Ok(Event::default().data(data)));

            // Prefill.
            let input = Tensor::new(prompt_tokens, device)
                .map_err(|e| anyhow::anyhow!("{e}"))?
                .unsqueeze(0)
                .map_err(|e| anyhow::anyhow!("{e}"))?;
            let logits = weights
                .forward(&input, 0)
                .map_err(|e| anyhow::anyhow!("{e}"))?;
            let mut prev = sample_token(&logits, temperature, top_p)?;
            let mut generated = 1usize;

            // Send first token.
            send_content_chunk(&tx, &id, &model_id, &state.tokenizer, prev);

            // Decode loop.
            for i in 1..max_tokens {
                let pos = prompt_tokens.len() + i - 1;
                let input = Tensor::new(&[prev], device)
                    .map_err(|e| anyhow::anyhow!("{e}"))?
                    .unsqueeze(0)
                    .map_err(|e| anyhow::anyhow!("{e}"))?;
                let logits = weights
                    .forward(&input, pos)
                    .map_err(|e| anyhow::anyhow!("{e}"))?;
                let token = sample_token(&logits, temperature, top_p)?;
                generated += 1;

                if is_eos(&state.tokenizer, token) {
                    break;
                }

                send_content_chunk(&tx, &id, &model_id, &state.tokenizer, token);
                prev = token;
            }

            // Send stop chunk.
            let chunk = ChatCompletionChunk {
                id: id.clone(),
                object: "chat.completion.chunk".to_string(),
                created: unix_timestamp(),
                model: model_id.clone(),
                choices: vec![ChatChunkChoice {
                    index: 0,
                    delta: ChatDelta {
                        role: None,
                        content: None,
                    },
                    finish_reason: Some("stop".to_string()),
                }],
            };
            let data = serde_json::to_string(&chunk)?;
            let _ = tx.blocking_send(Ok(Event::default().data(data)));

            // Send [DONE].
            let _ = tx.blocking_send(Ok(Event::default().data("[DONE]")));

            Ok(())
        };

        if let Err(e) = run() {
            tracing::error!("stream generation error: {e}");
        }
    });

    let stream = tokio_stream::wrappers::ReceiverStream::new(rx);
    Sse::new(stream).keep_alive(KeepAlive::default())
}

fn send_content_chunk(
    tx: &tokio::sync::mpsc::Sender<Result<Event, std::convert::Infallible>>,
    id: &str,
    model_id: &str,
    tokenizer: &Tokenizer,
    token: u32,
) {
    let text = tokenizer.decode(&[token], false).unwrap_or_default();
    if text.is_empty() {
        return;
    }
    let chunk = ChatCompletionChunk {
        id: id.to_string(),
        object: "chat.completion.chunk".to_string(),
        created: unix_timestamp(),
        model: model_id.to_string(),
        choices: vec![ChatChunkChoice {
            index: 0,
            delta: ChatDelta {
                role: None,
                content: Some(text),
            },
            finish_reason: None,
        }],
    };
    if let Ok(data) = serde_json::to_string(&chunk) {
        let _ = tx.blocking_send(Ok(Event::default().data(data)));
    }
}

// ---------------------------------------------------------------------------
// Blocking generation (non-streaming)
// ---------------------------------------------------------------------------

/// Generate text on a blocking thread. Returns (text, prompt_tokens, gen_tokens).
async fn generate_blocking(
    state: Arc<ServerState>,
    prompt: &str,
    max_tokens: usize,
    temperature: f64,
    top_p: f64,
) -> Result<(String, usize, usize)> {
    let prompt = prompt.to_string();
    let (tx, rx) = oneshot::channel();

    tokio::task::spawn_blocking(move || {
        let result = generate_sync(&state, &prompt, max_tokens, temperature, top_p);
        let _ = tx.send(result);
    });

    rx.await.map_err(|_| anyhow::anyhow!("generation task panicked"))?
}

fn generate_sync(
    state: &ServerState,
    prompt: &str,
    max_tokens: usize,
    temperature: f64,
    top_p: f64,
) -> Result<(String, usize, usize)> {
    let encoding = state
        .tokenizer
        .encode(prompt, true)
        .map_err(|e| anyhow::anyhow!("encode: {e}"))?;
    let prompt_tokens = encoding.get_ids();
    let device = &state.device;

    let mut weights = state
        .weights
        .lock()
        .map_err(|e| anyhow::anyhow!("lock: {e}"))?;

    // Prefill.
    let input = Tensor::new(prompt_tokens, device)
        .map_err(|e| anyhow::anyhow!("{e}"))?
        .unsqueeze(0)
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    let logits = weights
        .forward(&input, 0)
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    let mut prev = sample_token(&logits, temperature, top_p)?;
    let mut generated_ids = vec![prev];

    // Decode.
    for i in 1..max_tokens {
        let pos = prompt_tokens.len() + i - 1;
        let input = Tensor::new(&[prev], device)
            .map_err(|e| anyhow::anyhow!("{e}"))?
            .unsqueeze(0)
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        let logits = weights
            .forward(&input, pos)
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        let token = sample_token(&logits, temperature, top_p)?;

        if is_eos(&state.tokenizer, token) {
            break;
        }

        generated_ids.push(token);
        prev = token;
    }

    let text = state
        .tokenizer
        .decode(&generated_ids, true)
        .map_err(|e| anyhow::anyhow!("decode: {e}"))?;

    Ok((text, prompt_tokens.len(), generated_ids.len()))
}

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

/// Format chat messages into a single prompt string.
fn format_chat_prompt(messages: &[ChatMessage]) -> String {
    // Use a simple ChatML-style template.
    let mut prompt = String::new();
    for msg in messages {
        prompt.push_str(&format!("<|im_start|>{}\n{}<|im_end|>\n", msg.role, msg.content));
    }
    prompt.push_str("<|im_start|>assistant\n");
    prompt
}

/// Sample a token from logits.
fn sample_token(logits: &Tensor, temperature: f64, top_p: f64) -> Result<u32> {
    let logits = logits
        .squeeze(0)
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    let logits = if logits.dims().len() == 2 {
        logits
            .get(logits.dim(0).map_err(|e| anyhow::anyhow!("{e}"))? - 1)
            .map_err(|e| anyhow::anyhow!("{e}"))?
    } else {
        logits
    };
    let logits = logits
        .to_dtype(DType::F32)
        .map_err(|e| anyhow::anyhow!("{e}"))?;

    if temperature < 1e-7 {
        let token = logits
            .argmax(0)
            .map_err(|e| anyhow::anyhow!("{e}"))?
            .to_scalar::<u32>()
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        return Ok(token);
    }

    let logits = (&logits / temperature).map_err(|e| anyhow::anyhow!("{e}"))?;
    let probs = candle_nn::ops::softmax(&logits, 0).map_err(|e| anyhow::anyhow!("{e}"))?;
    let probs_vec: Vec<f32> = probs.to_vec1().map_err(|e| anyhow::anyhow!("{e}"))?;

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

fn is_eos(tokenizer: &Tokenizer, token: u32) -> bool {
    for special in &["</s>", "<|endoftext|>", "<|im_end|>"] {
        if let Some(id) = tokenizer.get_added_vocabulary().get_vocab().get(*special) {
            if token == *id {
                return true;
            }
        }
    }
    false
}

fn rand_float() -> f32 {
    use std::time::SystemTime;
    let nanos = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap_or_default()
        .subsec_nanos();
    (nanos % 10000) as f32 / 10000.0
}
