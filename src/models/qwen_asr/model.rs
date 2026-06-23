use burn::config::Config;
use burn::module::Module;
use burn::nn::LinearConfig;
use burn::nn::Linear;

use super::config::{AudioEncoderConfig, TextConfig};
use super::decoder::{Qwen3ASRThinkerTextModel, Qwen3ASRTextConfig};
use super::encoder::{Qwen3ASRAudioEncoder, Qwen3ASRAudioEncoderConfig};

#[derive(Module, Debug)]
pub struct Qwen3ASRThinkerForConditionalGeneration {
    pub audio_tower: Qwen3ASRAudioEncoder,
    pub model: Qwen3ASRThinkerTextModel,
    pub lm_head: Linear,
}

#[derive(Config, Debug)]
pub struct Qwen3ASRThinkerConfig {
    audio_tower: Qwen3ASRAudioEncoderConfig,
    model: Qwen3ASRTextConfig,
    output_dim: usize,
}

impl Qwen3ASRThinkerConfig {
    pub fn new_with_output_dim(
        audio_tower: Qwen3ASRAudioEncoderConfig,
        model: Qwen3ASRTextConfig,
        output_dim: usize,
    ) -> Self {
        Self {
            audio_tower,
            model,
            output_dim,
        }
    }

    pub fn init(
        &self,
        device: &burn::tensor::Device,
    ) -> Qwen3ASRThinkerForConditionalGeneration {
        Qwen3ASRThinkerForConditionalGeneration {
            audio_tower: self.audio_tower.init(device),
            model: self.model.init(device),
            lm_head: LinearConfig::new(self.model.hidden_size, self.output_dim)
                .with_bias(false)
                .init(device),
        }
    }
}

#[derive(Module, Debug)]
pub struct Qwen3ASR {
    pub thinker: Qwen3ASRThinkerForConditionalGeneration,
}

#[derive(Config, Debug)]
pub struct Qwen3ASRConfig {
    thinker: Qwen3ASRThinkerConfig,
}

impl Qwen3ASRConfig {
    pub fn from_configs(audio_config: AudioEncoderConfig, text_config: TextConfig) -> Self {
        let eps = text_config.rms_norm_eps;
        let output_dim = text_config.vocab_size;
        Self {
            thinker: Qwen3ASRThinkerConfig::new(
                Qwen3ASRAudioEncoderConfig::new(
                    audio_config.d_model,
                    audio_config.encoder_ffn_dim,
                    audio_config.encoder_layers,
                    audio_config.encoder_attention_heads,
                    audio_config.downsample_hidden_size,
                    audio_config.num_mel_bins,
                    audio_config.output_dim,
                    eps,
                    audio_config.n_window,
                    audio_config.n_window_infer,
                    audio_config.conv_chunksize,
                ),
                Qwen3ASRTextConfig::new(
                    text_config.vocab_size,
                    text_config.hidden_size,
                    text_config.intermediate_size,
                    text_config.num_hidden_layers,
                    text_config.num_attention_heads,
                    text_config.num_key_value_heads,
                    text_config.head_dim,
                    eps,
                ),
                output_dim,
            ),
        }
    }

    pub fn from_aligner_configs(
        audio_config: AudioEncoderConfig,
        text_config: TextConfig,
        classify_num: usize,
    ) -> Self {
        let eps = text_config.rms_norm_eps;
        let audio_tower = Qwen3ASRAudioEncoderConfig::new(
            audio_config.d_model,
            audio_config.encoder_ffn_dim,
            audio_config.encoder_layers,
            audio_config.encoder_attention_heads,
            audio_config.downsample_hidden_size,
            audio_config.num_mel_bins,
            audio_config.output_dim,
            eps,
            audio_config.n_window,
            audio_config.n_window_infer,
            audio_config.conv_chunksize,
        );
        let model = Qwen3ASRTextConfig::new(
            text_config.vocab_size,
            text_config.hidden_size,
            text_config.intermediate_size,
            text_config.num_hidden_layers,
            text_config.num_attention_heads,
            text_config.num_key_value_heads,
            text_config.head_dim,
            eps,
        );
        Self {
            thinker: Qwen3ASRThinkerConfig::new_with_output_dim(
                audio_tower,
                model,
                classify_num,
            ),
        }
    }

    pub fn init(&self, device: &burn::tensor::Device) -> Qwen3ASR {
        Qwen3ASR {
            thinker: self.thinker.init(device),
        }
    }
}
