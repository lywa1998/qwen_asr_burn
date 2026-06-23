use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
pub struct ModelConfig {
    pub thinker_config: ThinkerConfigRaw,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ThinkerConfigRaw {
    pub audio_config: AudioEncoderConfig,
    pub text_config: TextConfig,
    pub audio_start_token_id: u32,
    pub audio_end_token_id: u32,
    pub audio_token_id: u32,
}

#[derive(Debug, Clone, Deserialize)]
#[allow(dead_code)]
pub struct AudioEncoderConfig {
    pub d_model: usize,
    pub encoder_attention_heads: usize,
    pub encoder_ffn_dim: usize,
    pub encoder_layers: usize,
    pub downsample_hidden_size: usize,
    pub num_mel_bins: usize,
    pub output_dim: usize,
    pub max_source_positions: usize,
    pub activation_function: String,
    #[serde(default = "default_false")]
    pub scale_embedding: bool,
    #[serde(default = "default_n_window")]
    pub n_window: usize,
    #[serde(default = "default_n_window_infer")]
    pub n_window_infer: usize,
    #[serde(default = "default_conv_chunksize")]
    pub conv_chunksize: usize,
}

fn default_false() -> bool { false }
fn default_n_window() -> usize { 50 }
fn default_n_window_infer() -> usize { 800 }
fn default_conv_chunksize() -> usize { 500 }

#[derive(Debug, Clone, Deserialize)]
#[allow(dead_code)]
pub struct TextConfig {
    pub hidden_size: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub vocab_size: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f64,
    pub hidden_act: String,
    pub max_position_embeddings: usize,
    pub use_cache: bool,
    pub tie_word_embeddings: bool,
    #[serde(default)]
    pub attention_bias: bool,
    pub rope_scaling: Option<RopeScalingConfig>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct RopeScalingConfig {
    pub interleaved: Option<bool>,
    pub mrope_interleaved: Option<bool>,
    #[serde(default)]
    pub mrope_section: Vec<usize>,
}

#[derive(Debug, Clone, Deserialize)]
#[allow(dead_code)]
pub struct PreprocessorConfig {
    pub feature_size: usize,
    pub n_fft: usize,
    pub hop_length: usize,
    pub n_samples: usize,
    pub nb_max_frames: usize,
    pub chunk_length: f32,
    pub padding_value: f64,
    pub return_attention_mask: bool,
}

#[derive(Debug, Clone, Deserialize)]
#[allow(dead_code)]
pub struct GenerationConfig {
    pub eos_token_id: Vec<u32>,
    pub pad_token_id: u32,
    pub do_sample: bool,
    pub temperature: f64,
}

impl TextConfig {
    pub fn mrope_section(&self) -> Vec<usize> {
        self.rope_scaling
            .as_ref()
            .and_then(|rope| (!rope.mrope_section.is_empty()).then(|| rope.mrope_section.clone()))
            .unwrap_or_else(|| vec![24, 20, 20])
    }

    pub fn mrope_interleaved(&self) -> bool {
        self.rope_scaling
            .as_ref()
            .map(|rope| rope.mrope_interleaved.unwrap_or(false) || rope.interleaved.unwrap_or(false))
            .unwrap_or(true)
    }
}

impl ModelConfig {
    pub fn from_dir(model_dir: &str) -> anyhow::Result<Self> {
        let path = format!("{}/config.json", model_dir);
        let content = std::fs::read_to_string(&path)?;
        Ok(serde_json::from_str(&content)?)
    }
}

// --- ForcedAligner config types ---

#[derive(Debug, Clone, Deserialize)]
pub struct ForcedAlignerConfig {
    pub thinker_config: ForcedAlignerThinkerConfig,
    #[serde(default)]
    pub timestamp_segment_time: u32,
}

#[derive(Debug, Clone, Deserialize)]
#[allow(dead_code)]
pub struct ForcedAlignerThinkerConfig {
    pub audio_config: AudioEncoderConfig,
    pub text_config: TextConfig,
    #[serde(default)]
    pub classify_num: usize,
    #[serde(default)]
    pub audio_start_token_id: u32,
    #[serde(default)]
    pub audio_end_token_id: u32,
    #[serde(default)]
    pub audio_token_id: u32,
}

impl ForcedAlignerConfig {
    pub fn from_dir(model_dir: &str) -> anyhow::Result<Self> {
        let path = format!("{}/config.json", model_dir);
        let content = std::fs::read_to_string(&path)?;
        Ok(serde_json::from_str(&content)?)
    }
}

impl PreprocessorConfig {
    pub fn from_dir(model_dir: &str) -> anyhow::Result<Self> {
        let path = format!("{}/preprocessor_config.json", model_dir);
        let content = std::fs::read_to_string(&path)?;
        Ok(serde_json::from_str(&content)?)
    }
}

impl GenerationConfig {
    pub fn from_dir(model_dir: &str) -> anyhow::Result<Self> {
        let path = format!("{}/generation_config.json", model_dir);
        let content = std::fs::read_to_string(&path)?;
        Ok(serde_json::from_str(&content)?)
    }
}
