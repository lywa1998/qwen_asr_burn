use burn::config::Config;
use burn::module::Module;
use burn::nn::EmbeddingConfig;
use burn::nn::Embedding;
use burn::tensor::{Bool, Tensor};

use super::attention::{KvCacheEntry, Qwen3ASRAttention, Qwen3ASRAttentionConfig};
use super::mlp::{Qwen3ASRMLP, Qwen3ASRMLPConfig};
use super::norm::{Qwen3ASRRmsNorm, Qwen3ASRRmsNormConfig};

#[derive(Module, Debug)]
pub struct Qwen3ASRDecoderLayer {
    pub input_layernorm: Qwen3ASRRmsNorm,
    pub self_attn: Qwen3ASRAttention,
    pub post_attention_layernorm: Qwen3ASRRmsNorm,
    pub mlp: Qwen3ASRMLP,
}

#[derive(Config, Debug)]
pub struct Qwen3ASRDecoderLayerConfig {
    hidden_size: usize,
    intermediate_size: usize,
    num_q_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    epsilon: f64,
}

impl Qwen3ASRDecoderLayerConfig {
    pub fn init(&self, device: &burn::tensor::Device) -> Qwen3ASRDecoderLayer {
        Qwen3ASRDecoderLayer {
            input_layernorm: Qwen3ASRRmsNormConfig::new(self.hidden_size, self.epsilon).init(device),
            self_attn: Qwen3ASRAttentionConfig::new(
                self.hidden_size,
                self.num_q_heads,
                self.num_kv_heads,
                self.head_dim,
                self.epsilon,
            )
            .init(device),
            post_attention_layernorm: Qwen3ASRRmsNormConfig::new(self.hidden_size, self.epsilon)
                .init(device),
            mlp: Qwen3ASRMLPConfig::new(self.hidden_size, self.intermediate_size).init(device),
        }
    }
}

impl Qwen3ASRDecoderLayer {
    pub fn forward(
        &self,
        hidden_states: Tensor<3>,
        cos: &Tensor<4>,
        sin: &Tensor<4>,
        causal_mask: Option<Tensor<4, Bool>>,
        kv_cache: Option<&KvCacheEntry>,
    ) -> (Tensor<3>, KvCacheEntry) {
        let residual = hidden_states.clone();
        let hidden_states = self.input_layernorm.forward(hidden_states);
        let (hidden_states, new_cache) =
            self.self_attn
                .forward(hidden_states, cos, sin, causal_mask, kv_cache);
        let hidden_states = hidden_states.add(residual);

        let residual = hidden_states.clone();
        let hidden_states = self.post_attention_layernorm.forward(hidden_states);
        let hidden_states = self.mlp.forward(hidden_states);
        (hidden_states.add(residual), new_cache)
    }
}

#[derive(Module, Debug)]
pub struct Qwen3ASRThinkerTextModel {
    pub embed_tokens: Embedding,
    pub layers: Vec<Qwen3ASRDecoderLayer>,
    pub norm: Qwen3ASRRmsNorm,
}

#[derive(Config, Debug)]
pub struct Qwen3ASRTextConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_q_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub epsilon: f64,
}

impl Qwen3ASRTextConfig {
    pub fn init(&self, device: &burn::tensor::Device) -> Qwen3ASRThinkerTextModel {
        let layer_config = Qwen3ASRDecoderLayerConfig::new(
            self.hidden_size,
            self.intermediate_size,
            self.num_q_heads,
            self.num_kv_heads,
            self.head_dim,
            self.epsilon,
        );
        let layers = (0..self.num_hidden_layers)
            .map(|_| layer_config.init(device))
            .collect();
        Qwen3ASRThinkerTextModel {
            embed_tokens: EmbeddingConfig::new(self.vocab_size, self.hidden_size).init(device),
            layers,
            norm: Qwen3ASRRmsNormConfig::new(self.hidden_size, self.epsilon).init(device),
        }
    }
}

impl Qwen3ASRThinkerTextModel {
    pub fn forward_embeds(
        &self,
        hidden_states: Tensor<3>,
        cos: &Tensor<4>,
        sin: &Tensor<4>,
        causal_mask: Option<Tensor<4, Bool>>,
        kv_cache: Option<&mut super::attention::KvCache>,
    ) -> Tensor<3> {
        let mut hidden_states = hidden_states;
        match kv_cache {
            Some(cache) => {
                for (index, layer) in self.layers.iter().enumerate() {
                    let cached = cache.layer(index);
                    let (next_hidden, new_cache) =
                        layer.forward(hidden_states, cos, sin, causal_mask.clone(), cached);
                    cache.set_layer(index, new_cache);
                    hidden_states = next_hidden;
                }
            }
            None => {
                for layer in &self.layers {
                    let (next_hidden, _) =
                        layer.forward(hidden_states, cos, sin, causal_mask.clone(), None);
                    hidden_states = next_hidden;
                }
            }
        }
        self.norm.forward(hidden_states)
    }
}
