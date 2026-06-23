use burn::module::Module;
use burn::tensor::Tensor;

use super::attention::HYV3Attention;
use super::mlp::HYV3MLP;
use super::norm::HYV3RMSNorm;

use super::config::HYV3Config;

#[derive(Module, Debug)]
pub struct HYV3DecoderLayer {
    pub input_layernorm: HYV3RMSNorm,
    pub self_attn: HYV3Attention,
    pub post_attention_layernorm: HYV3RMSNorm,
    pub mlp: HYV3MLP,
}

impl HYV3DecoderLayer {
    pub fn new(config: &HYV3Config, device: &burn::tensor::Device) -> Self {
        Self {
            input_layernorm: HYV3RMSNorm::new(config.hidden_size, device),
            self_attn: HYV3Attention::new(config, device),
            post_attention_layernorm: HYV3RMSNorm::new(config.hidden_size, device),
            mlp: HYV3MLP::new(config.hidden_size, config.intermediate_size, device),
        }
    }

    pub fn forward(
        &self,
        x: Tensor<2>,
        batch: usize,
        seq_len: usize,
        rope_cos: Tensor<4>,
        rope_sin: Tensor<4>,
    ) -> Tensor<2> {
        let residual = x.clone();
        let x = self.input_layernorm.forward(x);
        let x = self.self_attn.forward(x, batch, seq_len, rope_cos, rope_sin);
        let x = x + residual;

        let residual = x.clone();
        let x = self.post_attention_layernorm.forward(x);
        let x = self.mlp.forward(x);
        x + residual
    }
}
