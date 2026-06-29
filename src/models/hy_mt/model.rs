use burn::module::Module;
use burn::module::Param;
use burn::nn;
use burn::tensor::{Int, Tensor};

use super::config::HYV3Config;
use super::decoder::HYV3DecoderLayer;
use super::norm::HYV3RMSNorm;
use super::rope::HYV3RotaryEmbedding;

#[derive(Module, Debug)]
pub struct HYV3Model {
    pub embed_tokens: nn::Embedding,
    pub layers: Vec<HYV3DecoderLayer>,
    pub norm: HYV3RMSNorm,
}

#[derive(Module, Debug)]
pub struct HYV3ForCausalLM {
    pub model: HYV3Model,
    pub lm_head: nn::Linear,
}

impl HYV3ForCausalLM {
    pub fn new(config: &HYV3Config, device: &burn::tensor::Device) -> Self {
        let embed_tokens =
            nn::EmbeddingConfig::new(config.vocab_size, config.hidden_size).init(device);

        let layers: Vec<HYV3DecoderLayer> = (0..config.num_hidden_layers)
            .map(|_| HYV3DecoderLayer::new(config, device))
            .collect();

        let norm = HYV3RMSNorm::new(config.hidden_size, device);

        let lm_head = nn::LinearConfig::new(config.hidden_size, config.vocab_size)
            .with_bias(false)
            .init(device);

        Self {
            model: HYV3Model {
                embed_tokens,
                layers,
                norm,
            },
            lm_head,
        }
    }

    /// Copy `embed_tokens.weight` into `lm_head.weight` (transposed) to handle
    /// `tie_word_embeddings: true` checkpoints, where `lm_head.weight` is
    /// absent from safetensors. Embedding stores `[vocab, hidden]`; burn's
    /// Linear stores `[d_input, d_output]` = `[hidden, vocab]`, so a transpose
    /// is required.
    pub fn tie_lm_head_to_embeddings(&mut self) {
        let embed = self.model.embed_tokens.weight.val();
        let tied = embed.swap_dims(0, 1);
        self.lm_head.weight = Param::from_tensor(tied);
    }

    /// input_ids: [batch, seq_len] Int
    /// returns: [batch, seq_len, vocab_size]
    pub fn forward(
        &self,
        input_ids: Tensor<2, Int>,
        rope: &HYV3RotaryEmbedding,
    ) -> Tensor<3> {
        let dims = input_ids.dims();
        let batch = dims[0];
        let seq_len = dims[1];
        let device = input_ids.device();

        let x = self.model.embed_tokens.forward(input_ids);
        let hidden = x.dims()[2];

        let mut x = x.reshape([batch * seq_len, hidden]);

        let (cos, sin) = rope.compute(seq_len, &device);

        for layer in &self.model.layers {
            x = layer.forward(x, batch, seq_len, cos.clone(), sin.clone());
        }

        x = self.model.norm.forward(x);
        x = self.lm_head.forward(x);
        let vocab = x.dims()[1];
        x.reshape([batch, seq_len, vocab])
    }
}
