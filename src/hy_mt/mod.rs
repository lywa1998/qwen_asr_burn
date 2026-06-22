//! Hy-MT (Hunyuan Machine Translation) model module.
//!
//! Provides model architecture, inference, and an OpenAI-compatible API server.
//!
//! ## Usage
//!
//! ```bash
//! cargo run --features cuda -- serve -m Hy-MT2-1.8B
//! ```
//!
//! Then POST to `http://localhost:3000/v1/chat/completions` with:
//! ```json
//! {"model":"hy-mt","messages":[{"role":"user","content":"Hello!"}],"max_tokens":512}
//! ```

pub mod chat;
pub mod config;
pub mod generate;
pub mod model;
pub mod server;

pub use chat::{format_chat_prompt, ChatMessage};
pub use config::ModelConfig;
pub use generate::{generate, GenerationConfig};
pub use model::{HunYuanModel, RotaryEmbedding};
