use anyhow::Context;
use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
#[allow(dead_code)]
pub struct HYV3Config {
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub intermediate_size: usize,
    pub vocab_size: usize,
    pub max_position_embeddings: usize,
    #[serde(default = "default_rms_norm_eps")]
    pub rms_norm_eps: f64,
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f64,
    #[serde(default)]
    pub rope_scaling: Option<RopeScaling>,
    pub bos_token_id: usize,
    pub eos_token_id: usize,
    pub pad_token_id: usize,
    #[serde(default)]
    pub tie_word_embeddings: bool,
    #[serde(default)]
    pub use_qk_norm: bool,
}

#[derive(Debug, Clone, Deserialize)]
#[allow(dead_code)]
pub struct RopeScaling {
    #[serde(rename = "type")]
    pub scaling_type: String,
    pub alpha: f64,
    pub beta_fast: f64,
    pub beta_slow: f64,
    pub factor: f64,
    #[serde(default = "default_one")]
    pub mscale: f64,
    #[serde(default = "default_one")]
    pub mscale_all_dim: f64,
}

fn default_rms_norm_eps() -> f64 {
    1e-5
}
fn default_rope_theta() -> f64 {
    10000.0
}
fn default_one() -> f64 {
    1.0
}

impl HYV3Config {
    pub fn from_file(path: &str) -> anyhow::Result<Self> {
        let content =
            std::fs::read_to_string(path).with_context(|| format!("Failed to read {path}"))?;
        serde_json::from_str(&content).with_context(|| format!("Failed to parse {path}"))
    }
}
