use burn::module::Module;
use burn::module::Param;
use burn::nn;
use burn::tensor::activation::{silu, softmax};
use burn::tensor::backend::Backend;
use burn::tensor::{Int, Tensor};

use crate::hy_mt::config::ModelConfig;

const RMS_EPS: f64 = 1e-5_f64;

// ============================================================================
// RMSNorm
// ============================================================================

#[derive(Module, Debug)]
pub struct RMSNorm<B: Backend> {
    pub weight: Param<Tensor<B, 1>>,
}

impl<B: Backend> RMSNorm<B> {
    pub fn new(num_features: usize, device: &B::Device) -> Self {
        let weight = Tensor::ones([num_features], device);
        Self {
            weight: Param::from_tensor(weight),
        }
    }

    /// x: [N, D] — normalize along last dim (D), where N = batch*seq
    pub fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        let rms = x
            .clone()
            .powf_scalar(2.0)
            .mean_dim(1)
            .add_scalar(RMS_EPS)
            .powf_scalar(0.5);
        let x = x / rms;
        x * self.weight.val().unsqueeze_dim::<2>(0)
    }

    /// x: [B, S, H, D] — normalize along last dim (per-head QK norm)
    pub fn forward_4d(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let rms = x
            .clone()
            .powf_scalar(2.0)
            .mean_dim(3)
            .add_scalar(RMS_EPS)
            .powf_scalar(0.5);
        let x = x / rms;
        x * self
            .weight
            .val()
            .unsqueeze_dim::<2>(0)
            .unsqueeze_dim::<3>(0)
            .unsqueeze_dim::<4>(0)
    }
}

// ============================================================================
// Rotary Position Embedding (Dynamic NTK)
// ============================================================================

pub struct RotaryEmbedding {
    inv_freq: Vec<f32>,
    alpha: f64,
    head_dim: usize,
    max_seq_len: usize,
    theta: f64,
}

impl RotaryEmbedding {
    pub fn new(config: &ModelConfig) -> Self {
        let head_dim = config.head_dim;
        let theta = config.rope_theta;
        let mut inv_freq = Vec::with_capacity(head_dim / 2);
        for i in (0..head_dim).step_by(2) {
            inv_freq.push(1.0 / (theta.powf(i as f64 / head_dim as f64)) as f32);
        }

        let alpha = config
            .rope_scaling
            .as_ref()
            .map(|s| s.alpha)
            .unwrap_or(1.0);

        Self {
            inv_freq,
            alpha,
            head_dim,
            max_seq_len: config.max_position_embeddings,
            theta,
        }
    }

    /// Returns (cos, sin) of shape [1, 1, seq_len, head_dim].
    pub fn compute<B: Backend>(
        &self,
        seq_len: usize,
        device: &B::Device,
    ) -> (Tensor<B, 4>, Tensor<B, 4>) {
        let factor = if seq_len > self.max_seq_len {
            (seq_len as f64 / self.max_seq_len as f64).max(1.0)
        } else {
            1.0
        };

        let scaled_inv_freq: Vec<f32> = if (factor - 1.0).abs() > 1e-6 {
            let base = self.theta * factor.powf(self.alpha);
            (0..self.head_dim / 2)
                .map(|i| 1.0 / (base.powf(i as f64 * 2.0 / self.head_dim as f64)) as f32)
                .collect()
        } else {
            self.inv_freq.clone()
        };

        let half = self.head_dim / 2;
        let position_ids: Vec<f32> = (0..seq_len).map(|i| i as f32).collect();
        let mut freqs = Vec::with_capacity(seq_len * half);
        for pos in &position_ids {
            for inv_f in &scaled_inv_freq {
                freqs.push(pos * inv_f);
            }
        }

        let freqs_t =
            Tensor::<B, 1>::from_floats(freqs.as_slice(), device).reshape([seq_len, half]);
        let cos = freqs_t.clone().cos().unsqueeze_dim::<3>(0).unsqueeze_dim::<4>(1);
        let sin = freqs_t.sin().unsqueeze_dim::<3>(0).unsqueeze_dim::<4>(1);

        let cos = Tensor::cat(vec![cos.clone(), cos], 3);
        let sin = Tensor::cat(vec![sin.clone(), sin], 3);
        (cos, sin)
    }
}

// ============================================================================
// Attention (GQA + QK LayerNorm + RoPE)
// ============================================================================

#[derive(Module, Debug)]
pub struct Attention<B: Backend> {
    pub q_proj: nn::Linear<B>,
    pub k_proj: nn::Linear<B>,
    pub v_proj: nn::Linear<B>,
    pub o_proj: nn::Linear<B>,
    pub query_layernorm: RMSNorm<B>,
    pub key_layernorm: RMSNorm<B>,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    scale: f64,
}

impl<B: Backend> Attention<B> {
    pub fn new(config: &ModelConfig, device: &B::Device) -> Self {
        let hidden = config.hidden_size;
        let q_out = config.num_attention_heads * config.head_dim;
        let kv_out = config.num_key_value_heads * config.head_dim;

        Self {
            q_proj: nn::LinearConfig::new(hidden, q_out)
                .with_bias(false)
                .init(device),
            k_proj: nn::LinearConfig::new(hidden, kv_out)
                .with_bias(false)
                .init(device),
            v_proj: nn::LinearConfig::new(hidden, kv_out)
                .with_bias(false)
                .init(device),
            o_proj: nn::LinearConfig::new(q_out, hidden)
                .with_bias(false)
                .init(device),
            query_layernorm: RMSNorm::new(config.head_dim, device),
            key_layernorm: RMSNorm::new(config.head_dim, device),
            num_heads: config.num_attention_heads,
            num_kv_heads: config.num_key_value_heads,
            head_dim: config.head_dim,
            scale: 1.0 / (config.head_dim as f64).sqrt(),
        }
    }

    /// x: [batch*seq_len, hidden]
    /// rope_cos/rope_sin: [1, 1, seq_len, head_dim]
    pub fn forward(
        &self,
        x: Tensor<B, 2>,
        batch: usize,
        seq_len: usize,
        rope_cos: Tensor<B, 4>,
        rope_sin: Tensor<B, 4>,
    ) -> Tensor<B, 2> {
        let q = self.q_proj.forward(x.clone());
        let k = self.k_proj.forward(x.clone());
        let v = self.v_proj.forward(x);

        let q = q.reshape([batch, seq_len, self.num_heads, self.head_dim]);
        let k = k.reshape([batch, seq_len, self.num_kv_heads, self.head_dim]);
        let v = v.reshape([batch, seq_len, self.num_kv_heads, self.head_dim]);

        let q = self.query_layernorm.forward_4d(q);
        let k = self.key_layernorm.forward_4d(k);

        let q = q.swap_dims(1, 2);
        let k = k.swap_dims(1, 2);
        let v = v.swap_dims(1, 2);

        let q = apply_rotary_pos_emb(q, rope_cos.clone(), rope_sin.clone());
        let k = apply_rotary_pos_emb(k, rope_cos, rope_sin);

        let kv_ratio = self.num_heads / self.num_kv_heads;
        let k = repeat_kv(k, kv_ratio);
        let v = repeat_kv(v, kv_ratio);

        let attn_weights = q.matmul(k.swap_dims(2, 3)).mul_scalar(self.scale);
        let mask = causal_mask::<B>(seq_len, attn_weights.device());
        let attn_weights = softmax(attn_weights + mask, 3);

        let attn_out = attn_weights.matmul(v);
        let attn_out = attn_out.swap_dims(1, 2);
        let attn_out = attn_out.reshape([batch * seq_len, self.num_heads * self.head_dim]);

        self.o_proj.forward(attn_out)
    }
}

fn apply_rotary_pos_emb<B: Backend>(
    x: Tensor<B, 4>,
    cos: Tensor<B, 4>,
    sin: Tensor<B, 4>,
) -> Tensor<B, 4> {
    let dims = x.dims();
    let half = dims[3] / 2;
    let x1 = x.clone().slice([0..dims[0], 0..dims[1], 0..dims[2], 0..half]);
    let x2 = x.slice([0..dims[0], 0..dims[1], 0..dims[2], half..dims[3]]);

    let cos_half = cos.clone().slice([0..dims[0], 0..dims[1], 0..dims[2], 0..half]);
    let sin_half = sin.clone().slice([0..dims[0], 0..dims[1], 0..dims[2], 0..half]);
    let rot_x = x1.clone() * cos_half.clone() - x2.clone() * sin_half.clone();
    let rot_y = x1 * sin_half + x2 * cos_half;
    Tensor::cat(vec![rot_x, rot_y], 3)
}

fn repeat_kv<B: Backend>(x: Tensor<B, 4>, n_rep: usize) -> Tensor<B, 4> {
    if n_rep == 1 {
        return x;
    }
    let d = x.dims();
    let x = x.unsqueeze_dim::<5>(2);
    let x = x.expand([d[0], d[1], n_rep, d[2], d[3]]);
    x.reshape([d[0], d[1] * n_rep, d[2], d[3]])
}

fn causal_mask<B: Backend>(seq_len: usize, device: B::Device) -> Tensor<B, 4> {
    let mut data = Vec::with_capacity(seq_len * seq_len);
    for i in 0..seq_len {
        for j in 0..seq_len {
            data.push(if j > i {
                f32::NEG_INFINITY
            } else {
                0.0f32
            });
        }
    }
    Tensor::<B, 1>::from_floats(data.as_slice(), &device).reshape([1, 1, seq_len, seq_len])
}

// ============================================================================
// MLP (SwiGLU)
// ============================================================================

#[derive(Module, Debug)]
pub struct MLP<B: Backend> {
    pub gate_proj: nn::Linear<B>,
    pub up_proj: nn::Linear<B>,
    pub down_proj: nn::Linear<B>,
}

impl<B: Backend> MLP<B> {
    pub fn new(hidden: usize, intermediate: usize, device: &B::Device) -> Self {
        Self {
            gate_proj: nn::LinearConfig::new(hidden, intermediate)
                .with_bias(false)
                .init(device),
            up_proj: nn::LinearConfig::new(hidden, intermediate)
                .with_bias(false)
                .init(device),
            down_proj: nn::LinearConfig::new(intermediate, hidden)
                .with_bias(false)
                .init(device),
        }
    }

    pub fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        let gate = silu(self.gate_proj.forward(x.clone()));
        let up = self.up_proj.forward(x);
        self.down_proj.forward(gate * up)
    }
}

// ============================================================================
// Decoder Layer
// ============================================================================

#[derive(Module, Debug)]
pub struct DecoderLayer<B: Backend> {
    pub input_layernorm: RMSNorm<B>,
    pub self_attn: Attention<B>,
    pub post_attention_layernorm: RMSNorm<B>,
    pub mlp: MLP<B>,
}

impl<B: Backend> DecoderLayer<B> {
    pub fn new(config: &ModelConfig, device: &B::Device) -> Self {
        Self {
            input_layernorm: RMSNorm::new(config.hidden_size, device),
            self_attn: Attention::new(config, device),
            post_attention_layernorm: RMSNorm::new(config.hidden_size, device),
            mlp: MLP::new(config.hidden_size, config.intermediate_size, device),
        }
    }

    pub fn forward(
        &self,
        x: Tensor<B, 2>,
        batch: usize,
        seq_len: usize,
        rope_cos: Tensor<B, 4>,
        rope_sin: Tensor<B, 4>,
    ) -> Tensor<B, 2> {
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

// ============================================================================
// InnerModel + HunYuanModel
// ============================================================================

#[derive(Module, Debug)]
pub struct InnerModel<B: Backend> {
    pub embed_tokens: nn::Embedding<B>,
    pub layers: Vec<DecoderLayer<B>>,
    pub norm: RMSNorm<B>,
}

#[derive(Module, Debug)]
pub struct HunYuanModel<B: Backend> {
    pub model: InnerModel<B>,
    pub lm_head: nn::Linear<B>,
}

impl<B: Backend> HunYuanModel<B> {
    pub fn new(config: &ModelConfig, device: &B::Device) -> Self {
        let embed_tokens =
            nn::EmbeddingConfig::new(config.vocab_size, config.hidden_size).init(device);

        let layers: Vec<DecoderLayer<B>> = (0..config.num_hidden_layers)
            .map(|_| DecoderLayer::new(config, device))
            .collect();

        let norm = RMSNorm::new(config.hidden_size, device);

        let lm_head = nn::LinearConfig::new(config.hidden_size, config.vocab_size)
            .with_bias(false)
            .init(device);

        Self {
            model: InnerModel {
                embed_tokens,
                layers,
                norm,
            },
            lm_head,
        }
    }

    /// input_ids: [batch, seq_len] Int
    /// returns: [batch, seq_len, vocab_size]
    pub fn forward(
        &self,
        input_ids: Tensor<B, 2, Int>,
        rope: &RotaryEmbedding,
    ) -> Tensor<B, 3> {
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
