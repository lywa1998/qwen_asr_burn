//! OpenAI-compatible API server for Hy-MT (Hunyuan Machine Translation).
//!
//! Endpoints:
//!   GET  /health              — health check
//!   POST /v1/chat/completions — OpenAI-compatible chat completion

use std::sync::Arc;

use axum::{
    extract::State,
    http::StatusCode,
    routing::{get, post},
    Json, Router,
};
use burn::tensor::backend::Backend;
use crate::pipeline::Bf16ToF32Adapter;
use burn_store::{ChainAdapter, ModuleSnapshot, PyTorchToBurnAdapter, SafetensorsStore};
use serde::{Deserialize, Serialize};
use tracing::info;

use crate::hy_mt::chat::{format_chat_prompt, ChatMessage};
use crate::hy_mt::config::ModelConfig;
use crate::hy_mt::generate::{generate, GenerationConfig};
use crate::hy_mt::model::{HunYuanModel, RotaryEmbedding};

// --- App State ---

struct AppState<B: Backend> {
    model: HunYuanModel<B>,
    rope: RotaryEmbedding,
    tokenizer: tokenizers::Tokenizer,
    gen_config: GenerationConfig,
    device: B::Device,
}

// --- OpenAI-compatible types ---

#[derive(Debug, Deserialize)]
struct ChatCompletionRequest {
    #[allow(dead_code)]
    model: String,
    messages: Vec<ChatMessage>,
    #[serde(default = "default_max_tokens")]
    max_tokens: usize,
    #[serde(default = "default_temperature")]
    temperature: f64,
    #[serde(default = "default_top_p")]
    top_p: f64,
    #[serde(default = "default_top_k")]
    top_k: usize,
}

fn default_max_tokens() -> usize {
    512
}
fn default_temperature() -> f64 {
    0.7
}
fn default_top_p() -> f64 {
    0.8
}
fn default_top_k() -> usize {
    20
}

#[derive(Debug, Serialize)]
struct ChatCompletionResponse {
    id: String,
    object: String,
    created: u64,
    model: String,
    choices: Vec<Choice>,
    usage: Usage,
}

#[derive(Debug, Serialize)]
struct Choice {
    index: u32,
    message: ChoiceMessage,
    finish_reason: String,
}

#[derive(Debug, Serialize)]
struct ChoiceMessage {
    role: String,
    content: String,
}

#[derive(Debug, Serialize)]
struct Usage {
    prompt_tokens: u32,
    completion_tokens: u32,
    total_tokens: u32,
}

// --- Handlers ---

async fn health() -> &'static str {
    "ok"
}

async fn chat_completions<B: Backend + 'static>(
    State(state): State<Arc<AppState<B>>>,
    Json(req): Json<ChatCompletionRequest>,
) -> Result<Json<ChatCompletionResponse>, (StatusCode, Json<serde_json::Value>)> {
    let prompt = format_chat_prompt(&req.messages);

    let mut gen_config = state.gen_config.clone();
    gen_config.max_new_tokens = req.max_tokens;
    gen_config.temperature = req.temperature;
    gen_config.top_p = req.top_p;
    gen_config.top_k = req.top_k;

    let generated = generate(
        &state.model,
        &state.rope,
        &state.tokenizer,
        &prompt,
        &gen_config,
        &state.device,
    )
    .map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({"error": e.to_string()})),
        )
    })?;

    let response = ChatCompletionResponse {
        id: format!("chatcmpl-{}", uuid::Uuid::new_v4()),
        object: "chat.completion".to_string(),
        created: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs(),
        model: req.model,
        choices: vec![Choice {
            index: 0,
            message: ChoiceMessage {
                role: "assistant".to_string(),
                content: generated,
            },
            finish_reason: "stop".to_string(),
        }],
        usage: Usage {
            prompt_tokens: 0,
            completion_tokens: 0,
            total_tokens: 0,
        },
    };

    Ok(Json(response))
}

// --- Public entry point ---

/// Start the Hy-MT API server on the given address.
///
/// `model_dir` must contain `config.json`, `tokenizer.json`, and
/// `model.safetensors` (or `model_f32.safetensors` for converted weights).
pub async fn run<B: Backend + 'static>(
    model_dir: &str,
    host: &str,
    port: u16,
    device: B::Device,
) -> anyhow::Result<()> {
    // Load config
    let config_path = format!("{}/config.json", model_dir);
    let model_config = ModelConfig::from_file(&config_path)?;
    info!("Loaded Hy-MT config: hidden_size={}", model_config.hidden_size);

    // Load tokenizer
    let tokenizer_path = format!("{}/tokenizer.json", model_dir);
    let mut tokenizer = tokenizers::Tokenizer::from_file(&tokenizer_path)
        .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?;
    info!("Loaded Hy-MT tokenizer");
    let pad_token_id = model_config.pad_token_id as u32;
    let _ = tokenizer.with_padding(Some(tokenizers::PaddingParams {
        strategy: tokenizers::PaddingStrategy::Fixed(0),
        pad_id: pad_token_id,
        ..Default::default()
    }));

    // Init model
    let mut model = HunYuanModel::<B>::new(&model_config, &device);
    info!("Initialized Hy-MT model structure");

    // Try model_f32.safetensors first (converted), then model.safetensors
    let safetensors_path = {
        let f32_path = format!("{}/model_f32.safetensors", model_dir);
        if std::path::Path::new(&f32_path).exists() {
            f32_path
        } else {
            format!("{}/model.safetensors", model_dir)
        }
    };
    info!("Loading Hy-MT weights from {}...", safetensors_path);
    let adapter = ChainAdapter::new(PyTorchToBurnAdapter, Bf16ToF32Adapter);
    let mut store =
        SafetensorsStore::from_file(&safetensors_path).with_from_adapter(adapter);
    let result = model.load_from(&mut store)
        .map_err(|e| anyhow::anyhow!("Failed to load Hy-MT weights: {}", e))?;
    info!(
        "Hy-MT weights: {} applied, {} errors",
        result.applied.len(),
        result.errors.len()
    );
    info!("Hy-MT weights loaded successfully");

    let rope = RotaryEmbedding::new(&model_config);
    let gen_config = GenerationConfig::from_model_config(&model_config);

    let state = Arc::new(AppState {
        model,
        rope,
        tokenizer,
        gen_config,
        device,
    });

    let app = Router::new()
        .route("/health", get(health))
        .route("/v1/chat/completions", post(chat_completions))
        .with_state(state);

    let addr = format!("{host}:{port}");
    info!("Hy-MT server starting on {addr}");
    let addr: std::net::SocketAddr = addr.parse()?;
    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}
