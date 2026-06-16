use burn::config::Config;
use burn::module::{Module, Param};
use burn::nn::conv::Conv2d;
use burn::nn::PaddingConfig2d;
use burn::nn::{conv::Conv2dConfig, EmbeddingConfig, LinearConfig};
use burn::nn::{Embedding, Linear};
use burn::tensor::backend::Backend;
use burn::tensor::{Bool, Int, Tensor};

use crate::config::{AudioEncoderConfig, TextConfig};

#[derive(Module, Debug)]
pub struct MyRmsNorm<B: Backend> {
    pub weight: Param<Tensor<B, 1>>,
    #[module(skip)]
    epsilon: f64,
}

#[derive(Config, Debug)]
pub struct MyRmsNormConfig {
    d_model: usize,
    epsilon: f64,
}

impl MyRmsNormConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> MyRmsNorm<B> {
        MyRmsNorm {
            weight: Param::from_tensor(Tensor::ones([self.d_model], device)),
            epsilon: self.epsilon,
        }
    }
}

impl<B: Backend> MyRmsNorm<B> {
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let rms = x
            .clone()
            .powf_scalar(2.0)
            .mean_dim(2)
            .add_scalar(self.epsilon)
            .sqrt();
        let w = self
            .weight
            .val()
            .unsqueeze_dim::<2>(0)
            .unsqueeze_dim::<3>(1);
        x.div(rms).mul(w)
    }
}

#[derive(Module, Debug)]
pub struct MyLayerNorm<B: Backend> {
    pub weight: Param<Tensor<B, 1>>,
    pub bias: Param<Tensor<B, 1>>,
    #[module(skip)]
    epsilon: f64,
}

#[derive(Config, Debug)]
pub struct MyLayerNormConfig {
    d_model: usize,
    epsilon: f64,
}

impl MyLayerNormConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> MyLayerNorm<B> {
        MyLayerNorm {
            weight: Param::from_tensor(Tensor::ones([self.d_model], device)),
            bias: Param::from_tensor(Tensor::zeros([self.d_model], device)),
            epsilon: self.epsilon,
        }
    }
}

impl<B: Backend> MyLayerNorm<B> {
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let ndim = 3;
        let mean = x.clone().mean_dim(ndim - 1);
        let var = x
            .clone()
            .sub(mean.clone())
            .powf_scalar(2.0)
            .mean_dim(ndim - 1);
        let x_norm = x.sub(mean).div(var.add_scalar(self.epsilon).sqrt());

        let w = self
            .weight
            .val()
            .unsqueeze_dim::<2>(0)
            .unsqueeze_dim::<3>(1);
        let b = self.bias.val().unsqueeze_dim::<2>(0).unsqueeze_dim::<3>(1);
        x_norm.mul(w).add(b)
    }
}

pub struct MRoPE {
    inv_freq: Vec<f32>,
    head_dim: usize,
}

impl MRoPE {
    pub fn new(
        head_dim: usize,
        rope_theta: f64,
        _mrope_section: &[usize],
        _interleaved: bool,
    ) -> Self {
        let half_dim = head_dim / 2;
        let inv_freq = (0..half_dim)
            .map(|i| (1.0 / rope_theta.powf(2.0 * i as f64 / head_dim as f64)) as f32)
            .collect();
        Self { inv_freq, head_dim }
    }

    pub fn compute_cos_sin<B: Backend>(
        &self,
        position_ids: Tensor<B, 2, Int>,
    ) -> (Tensor<B, 4>, Tensor<B, 4>) {
        let [batch, seq_len] = position_ids.dims();
        let device = position_ids.device();
        let half_dim = self.head_dim / 2;
        let pos_data = position_ids.to_data();
        let mut pos_vals = Vec::with_capacity(batch * seq_len);
        for chunk in pos_data.bytes.chunks_exact(4) {
            pos_vals.push(i32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
        }

        let mut cos_vals = Vec::with_capacity(batch * seq_len * self.head_dim);
        let mut sin_vals = Vec::with_capacity(batch * seq_len * self.head_dim);
        for pos in pos_vals {
            for j in 0..half_dim {
                let mapped_pos = pos as f32;
                let angle = mapped_pos * self.inv_freq[j];
                cos_vals.push(angle.cos());
                sin_vals.push(angle.sin());
            }
            for j in 0..half_dim {
                let mapped_pos = pos as f32;
                let angle = mapped_pos * self.inv_freq[j];
                cos_vals.push(angle.cos());
                sin_vals.push(angle.sin());
            }
        }

        let cos = Tensor::<B, 1>::from_floats(cos_vals.as_slice(), &device)
            .reshape([batch, seq_len, self.head_dim])
            .unsqueeze_dim::<4>(1);
        let sin = Tensor::<B, 1>::from_floats(sin_vals.as_slice(), &device)
            .reshape([batch, seq_len, self.head_dim])
            .unsqueeze_dim::<4>(1);
        (cos, sin)
    }

    pub fn compute_cos_sin_from_positions<B: Backend>(
        &self,
        positions: &[usize],
        device: &B::Device,
    ) -> (Tensor<B, 4>, Tensor<B, 4>) {
        let ids: Vec<i32> = positions.iter().map(|&pos| pos as i32).collect();
        let position_ids =
            Tensor::<B, 1, Int>::from_ints(ids.as_slice(), device).unsqueeze_dim::<2>(0);
        self.compute_cos_sin(position_ids)
    }
}

fn apply_mrope_simple<B: Backend>(
    x: Tensor<B, 4>,
    cos: &Tensor<B, 4>,
    sin: &Tensor<B, 4>,
) -> Tensor<B, 4> {
    let head_dim = x.dims()[3];
    let half = head_dim / 2;
    let x_clone = x.clone();
    let x1 = x_clone.clone().narrow(3, 0, half);
    let x2 = x_clone.narrow(3, half, half);
    let rotate_half = Tensor::cat(vec![x2.neg(), x1], 3);
    x * cos.clone() + rotate_half * sin.clone()
}

#[derive(Module, Debug)]
pub struct QKNorm<B: Backend> {
    pub weight: Param<Tensor<B, 1>>,
    #[module(skip)]
    epsilon: f64,
}

#[derive(Config, Debug)]
pub struct QKNormConfig {
    head_dim: usize,
    epsilon: f64,
}

impl QKNormConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> QKNorm<B> {
        QKNorm {
            weight: Param::from_tensor(Tensor::ones([self.head_dim], device)),
            epsilon: self.epsilon,
        }
    }
}

impl<B: Backend> QKNorm<B> {
    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let rms = x
            .clone()
            .powf_scalar(2.0)
            .mean_dim(3)
            .add_scalar(self.epsilon)
            .sqrt();
        let w = self
            .weight
            .val()
            .unsqueeze_dim::<2>(0)
            .unsqueeze_dim::<3>(1)
            .unsqueeze_dim::<4>(2);
        x.div(rms).mul(w)
    }
}

#[derive(Debug)]
pub struct KvCacheEntry<B: Backend> {
    pub k: Tensor<B, 4>,
    pub v: Tensor<B, 4>,
}

#[derive(Debug)]
pub struct KvCache<B: Backend> {
    layers: Vec<Option<KvCacheEntry<B>>>,
}

impl<B: Backend> KvCache<B> {
    pub fn new(num_layers: usize) -> Self {
        Self {
            layers: (0..num_layers).map(|_| None).collect(),
        }
    }

    pub fn layer(&self, index: usize) -> Option<&KvCacheEntry<B>> {
        self.layers.get(index).and_then(|entry| entry.as_ref())
    }

    pub fn set_layer(&mut self, index: usize, entry: KvCacheEntry<B>) {
        if let Some(slot) = self.layers.get_mut(index) {
            *slot = Some(entry);
        }
    }

    pub fn seq_len(&self) -> usize {
        self.layers
            .iter()
            .find_map(|entry| entry.as_ref().map(|cache| cache.k.dims()[2]))
            .unwrap_or(0)
    }
}

#[derive(Module, Debug)]
pub struct Qwen3Attention<B: Backend> {
    pub q_proj: Linear<B>,
    pub k_proj: Linear<B>,
    pub v_proj: Linear<B>,
    pub o_proj: Linear<B>,
    pub q_norm: QKNorm<B>,
    pub k_norm: QKNorm<B>,
    #[module(skip)]
    num_q_heads: usize,
    #[module(skip)]
    num_kv_heads: usize,
    #[module(skip)]
    head_dim: usize,
}

#[derive(Config, Debug)]
pub struct Qwen3AttentionConfig {
    hidden_size: usize,
    num_q_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    epsilon: f64,
}

impl Qwen3AttentionConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> Qwen3Attention<B> {
        Qwen3Attention {
            q_proj: LinearConfig::new(self.hidden_size, self.num_q_heads * self.head_dim)
                .with_bias(false)
                .init(device),
            k_proj: LinearConfig::new(self.hidden_size, self.num_kv_heads * self.head_dim)
                .with_bias(false)
                .init(device),
            v_proj: LinearConfig::new(self.hidden_size, self.num_kv_heads * self.head_dim)
                .with_bias(false)
                .init(device),
            o_proj: LinearConfig::new(self.num_q_heads * self.head_dim, self.hidden_size)
                .with_bias(false)
                .init(device),
            q_norm: QKNormConfig::new(self.head_dim, self.epsilon).init(device),
            k_norm: QKNormConfig::new(self.head_dim, self.epsilon).init(device),
            num_q_heads: self.num_q_heads,
            num_kv_heads: self.num_kv_heads,
            head_dim: self.head_dim,
        }
    }
}

impl<B: Backend> Qwen3Attention<B> {
    pub fn forward(
        &self,
        hidden_states: Tensor<B, 3>,
        cos: &Tensor<B, 4>,
        sin: &Tensor<B, 4>,
        causal_mask: Option<Tensor<B, 4, Bool>>,
        kv_cache: Option<&KvCacheEntry<B>>,
    ) -> (Tensor<B, 3>, KvCacheEntry<B>) {
        let [batch, seq_len, _hidden] = hidden_states.dims();
        let num_q_heads = self.num_q_heads;
        let num_kv_heads = self.num_kv_heads;
        let head_dim = self.head_dim;

        let q = self.q_proj.forward(hidden_states.clone());
        let k = self.k_proj.forward(hidden_states.clone());
        let v = self.v_proj.forward(hidden_states);

        let q = q
            .reshape([batch, seq_len, num_q_heads, head_dim])
            .swap_dims(1, 2);
        let k = k
            .reshape([batch, seq_len, num_kv_heads, head_dim])
            .swap_dims(1, 2);
        let v = v
            .reshape([batch, seq_len, num_kv_heads, head_dim])
            .swap_dims(1, 2);

        let q = self.q_norm.forward(q);
        let k = self.k_norm.forward(k);

        let q = apply_mrope_simple(q, cos, sin);
        let k_rot = apply_mrope_simple(k, cos, sin);

        let (k_full, v_full) = if let Some(cache) = kv_cache {
            (
                Tensor::cat(vec![cache.k.clone(), k_rot.clone()], 2),
                Tensor::cat(vec![cache.v.clone(), v.clone()], 2),
            )
        } else {
            (k_rot.clone(), v.clone())
        };

        let n_rep = num_q_heads / num_kv_heads;
        let new_cache = KvCacheEntry {
            k: k_full.clone(),
            v: v_full.clone(),
        };
        let k = repeat_kv(k_full, n_rep);
        let v = repeat_kv(v_full, n_rep);

        let scale = (head_dim as f64).sqrt();
        let attn_weights = q.matmul(k.swap_dims(2, 3)).div_scalar(scale);

        let attn_weights = if let Some(mask) = causal_mask {
            attn_weights.mask_fill(mask, f32::NEG_INFINITY)
        } else {
            attn_weights
        };

        let attn_weights = burn::tensor::activation::softmax(attn_weights, 3);
        let attn_output = attn_weights.matmul(v);

        let attn_output =
            attn_output
                .swap_dims(1, 2)
                .reshape([batch, seq_len, num_q_heads * head_dim]);
        (self.o_proj.forward(attn_output), new_cache)
    }
}

pub fn repeat_kv<B: Backend>(x: Tensor<B, 4>, n_rep: usize) -> Tensor<B, 4> {
    if n_rep == 1 {
        return x;
    }
    let [batch, num_kv_heads, seq_len, head_dim] = x.dims();
    x.unsqueeze_dim::<5>(2).repeat_dim(2, n_rep).reshape([
        batch,
        num_kv_heads * n_rep,
        seq_len,
        head_dim,
    ])
}

#[derive(Module, Debug)]
pub struct Qwen3MLP<B: Backend> {
    pub gate_proj: Linear<B>,
    pub up_proj: Linear<B>,
    pub down_proj: Linear<B>,
}

#[derive(Config, Debug)]
pub struct Qwen3MLPConfig {
    hidden_size: usize,
    intermediate_size: usize,
}

impl Qwen3MLPConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> Qwen3MLP<B> {
        Qwen3MLP {
            gate_proj: LinearConfig::new(self.hidden_size, self.intermediate_size)
                .with_bias(false)
                .init(device),
            up_proj: LinearConfig::new(self.hidden_size, self.intermediate_size)
                .with_bias(false)
                .init(device),
            down_proj: LinearConfig::new(self.intermediate_size, self.hidden_size)
                .with_bias(false)
                .init(device),
        }
    }
}

impl<B: Backend> Qwen3MLP<B> {
    pub fn forward<const D: usize>(&self, x: Tensor<B, D>) -> Tensor<B, D> {
        let gate = self.gate_proj.forward(x.clone());
        let gate = gate.clone().mul(burn::tensor::activation::sigmoid(gate));
        let up = self.up_proj.forward(x);
        self.down_proj.forward(gate.mul(up))
    }
}

#[derive(Module, Debug)]
pub struct Qwen3DecoderLayer<B: Backend> {
    pub input_layernorm: MyRmsNorm<B>,
    pub self_attn: Qwen3Attention<B>,
    pub post_attention_layernorm: MyRmsNorm<B>,
    pub mlp: Qwen3MLP<B>,
}

#[derive(Config, Debug)]
pub struct Qwen3DecoderLayerConfig {
    hidden_size: usize,
    intermediate_size: usize,
    num_q_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    epsilon: f64,
}

impl Qwen3DecoderLayerConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> Qwen3DecoderLayer<B> {
        Qwen3DecoderLayer {
            input_layernorm: MyRmsNormConfig::new(self.hidden_size, self.epsilon).init(device),
            self_attn: Qwen3AttentionConfig::new(
                self.hidden_size,
                self.num_q_heads,
                self.num_kv_heads,
                self.head_dim,
                self.epsilon,
            )
            .init(device),
            post_attention_layernorm: MyRmsNormConfig::new(self.hidden_size, self.epsilon).init(device),
            mlp: Qwen3MLPConfig::new(self.hidden_size, self.intermediate_size).init(device),
        }
    }
}

impl<B: Backend> Qwen3DecoderLayer<B> {
    pub fn forward(
        &self,
        hidden_states: Tensor<B, 3>,
        cos: &Tensor<B, 4>,
        sin: &Tensor<B, 4>,
        causal_mask: Option<Tensor<B, 4, Bool>>,
        kv_cache: Option<&KvCacheEntry<B>>,
    ) -> (Tensor<B, 3>, KvCacheEntry<B>) {
        let residual = hidden_states.clone();
        let hidden_states = self.input_layernorm.forward(hidden_states);
        let (hidden_states, new_cache) =
            self.self_attn.forward(hidden_states, cos, sin, causal_mask, kv_cache);
        let hidden_states = hidden_states.add(residual);

        let residual = hidden_states.clone();
        let hidden_states = self.post_attention_layernorm.forward(hidden_states);
        let hidden_states = self.mlp.forward(hidden_states);
        (hidden_states.add(residual), new_cache)
    }
}

#[derive(Module, Debug)]
pub struct Qwen3ASRThinkerTextModel<B: Backend> {
    pub embed_tokens: Embedding<B>,
    pub layers: Vec<Qwen3DecoderLayer<B>>,
    pub norm: MyRmsNorm<B>,
}

#[derive(Config, Debug)]
pub struct Qwen3ASRThinkerTextModelConfig {
    vocab_size: usize,
    hidden_size: usize,
    intermediate_size: usize,
    num_hidden_layers: usize,
    num_q_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    epsilon: f64,
}

impl Qwen3ASRThinkerTextModelConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> Qwen3ASRThinkerTextModel<B> {
        let layer_config = Qwen3DecoderLayerConfig::new(
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
            norm: MyRmsNormConfig::new(self.hidden_size, self.epsilon).init(device),
        }
    }
}

impl<B: Backend> Qwen3ASRThinkerTextModel<B> {
    pub fn forward_embeds(
        &self,
        hidden_states: Tensor<B, 3>,
        cos: &Tensor<B, 4>,
        sin: &Tensor<B, 4>,
        causal_mask: Option<Tensor<B, 4, Bool>>,
        kv_cache: Option<&mut KvCache<B>>,
    ) -> Tensor<B, 3> {
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

#[derive(Module, Debug)]
pub struct EncoderAttention<B: Backend> {
    pub q_proj: Linear<B>,
    pub k_proj: Linear<B>,
    pub v_proj: Linear<B>,
    pub out_proj: Linear<B>,
    #[module(skip)]
    num_heads: usize,
}

#[derive(Config, Debug)]
pub struct EncoderAttentionConfig {
    d_model: usize,
    num_heads: usize,
}

impl EncoderAttentionConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> EncoderAttention<B> {
        EncoderAttention {
            q_proj: LinearConfig::new(self.d_model, self.d_model).init(device),
            k_proj: LinearConfig::new(self.d_model, self.d_model).init(device),
            v_proj: LinearConfig::new(self.d_model, self.d_model).init(device),
            out_proj: LinearConfig::new(self.d_model, self.d_model).init(device),
            num_heads: self.num_heads,
        }
    }
}

impl<B: Backend> EncoderAttention<B> {
    pub fn forward(&self, hidden_states: Tensor<B, 3>) -> Tensor<B, 3> {
        let [batch, seq_len, d_model] = hidden_states.dims();
        let num_heads = self.num_heads;
        let head_dim = d_model / num_heads;

        let q = self.q_proj.forward(hidden_states.clone());
        let k = self.k_proj.forward(hidden_states.clone());
        let v = self.v_proj.forward(hidden_states);

        let q = q
            .reshape([batch, seq_len, num_heads, head_dim])
            .swap_dims(1, 2);
        let k = k
            .reshape([batch, seq_len, num_heads, head_dim])
            .swap_dims(1, 2);
        let v = v
            .reshape([batch, seq_len, num_heads, head_dim])
            .swap_dims(1, 2);

        let scale = (head_dim as f64).sqrt();
        let attn_weights = q.matmul(k.swap_dims(2, 3)).div_scalar(scale);
        let attn_weights = burn::tensor::activation::softmax(attn_weights, 3);
        let attn_output = attn_weights.matmul(v);

        let attn_output = attn_output
            .swap_dims(1, 2)
            .reshape([batch, seq_len, d_model]);
        self.out_proj.forward(attn_output)
    }
}

#[derive(Module, Debug)]
pub struct AudioEncoderLayer<B: Backend> {
    pub self_attn_layer_norm: MyLayerNorm<B>,
    pub self_attn: EncoderAttention<B>,
    pub fc1: Linear<B>,
    pub fc2: Linear<B>,
    pub final_layer_norm: MyLayerNorm<B>,
}

#[derive(Config, Debug)]
pub struct AudioEncoderLayerModuleConfig {
    d_model: usize,
    encoder_ffn_dim: usize,
    encoder_attention_heads: usize,
    epsilon: f64,
}

impl AudioEncoderLayerModuleConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> AudioEncoderLayer<B> {
        AudioEncoderLayer {
            self_attn_layer_norm: MyLayerNormConfig::new(self.d_model, self.epsilon).init(device),
            self_attn: EncoderAttentionConfig::new(self.d_model, self.encoder_attention_heads)
                .init(device),
            fc1: LinearConfig::new(self.d_model, self.encoder_ffn_dim).init(device),
            fc2: LinearConfig::new(self.encoder_ffn_dim, self.d_model).init(device),
            final_layer_norm: MyLayerNormConfig::new(self.d_model, self.epsilon).init(device),
        }
    }
}

impl<B: Backend> AudioEncoderLayer<B> {
    pub fn forward(&self, hidden_states: Tensor<B, 3>) -> Tensor<B, 3> {
        let residual = hidden_states.clone();
        let hidden_states = self.self_attn_layer_norm.forward(hidden_states);
        let hidden_states = self.self_attn.forward(hidden_states);
        let hidden_states = hidden_states.add(residual);

        let residual = hidden_states.clone();
        let hidden_states = self.final_layer_norm.forward(hidden_states);
        let hidden_states = self.fc1.forward(hidden_states);
        let hidden_states = burn::tensor::activation::gelu(hidden_states);
        let hidden_states = self.fc2.forward(hidden_states);
        hidden_states.add(residual)
    }
}

#[derive(Module, Debug)]
pub struct Qwen3ASRAudioEncoder<B: Backend> {
    pub conv2d1: Conv2d<B>,
    pub conv2d2: Conv2d<B>,
    pub conv2d3: Conv2d<B>,
    pub conv_out: Linear<B>,
    pub layers: Vec<AudioEncoderLayer<B>>,
    pub ln_post: MyLayerNorm<B>,
    pub proj1: Linear<B>,
    pub proj2: Linear<B>,
}

#[derive(Config, Debug)]
pub struct Qwen3ASRAudioEncoderModuleConfig {
    d_model: usize,
    encoder_ffn_dim: usize,
    encoder_layers: usize,
    encoder_attention_heads: usize,
    downsample_hidden_size: usize,
    num_mel_bins: usize,
    output_dim: usize,
    epsilon: f64,
}

impl Qwen3ASRAudioEncoderModuleConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> Qwen3ASRAudioEncoder<B> {
        let conv_freq_bins = (((self.num_mel_bins + 1) / 2 + 1) / 2 + 1) / 2;
        let conv_out_dim = self.downsample_hidden_size * conv_freq_bins;
        let layer_config = AudioEncoderLayerModuleConfig::new(
            self.d_model,
            self.encoder_ffn_dim,
            self.encoder_attention_heads,
            self.epsilon,
        );
        let layers = (0..self.encoder_layers)
            .map(|_| layer_config.init(device))
            .collect();
        Qwen3ASRAudioEncoder {
            conv2d1: Conv2dConfig::new([1, self.downsample_hidden_size], [3, 3])
                .with_stride([2, 2])
                .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
                .init(device),
            conv2d2: Conv2dConfig::new(
                [self.downsample_hidden_size, self.downsample_hidden_size],
                [3, 3],
            )
            .with_stride([2, 2])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .init(device),
            conv2d3: Conv2dConfig::new(
                [self.downsample_hidden_size, self.downsample_hidden_size],
                [3, 3],
            )
            .with_stride([2, 2])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .init(device),
            conv_out: LinearConfig::new(conv_out_dim, self.d_model)
                .with_bias(false)
                .init(device),
            layers,
            ln_post: MyLayerNormConfig::new(self.d_model, self.epsilon).init(device),
            proj1: LinearConfig::new(self.d_model, self.d_model).init(device),
            proj2: LinearConfig::new(self.d_model, self.output_dim).init(device),
        }
    }
}

impl<B: Backend> Qwen3ASRAudioEncoder<B> {
    pub fn forward(&self, mel: Tensor<B, 3>) -> Tensor<B, 3> {
        let [_batch, _n_mels, mel_time] = mel.dims();
        let chunk_size: usize = 100;
        let num_full_chunks = mel_time / chunk_size;
        let remainder = mel_time % chunk_size;

        let device = mel.device();
        let mut conv_outputs: Vec<Tensor<B, 3>> = Vec::new();

        for i in 0..num_full_chunks {
            let chunk = mel.clone().narrow(2, i * chunk_size, chunk_size);
            let out = self.conv_chunk(chunk);
            conv_outputs.push(out);
        }
        if remainder > 0 {
            let chunk = mel.narrow(2, num_full_chunks * chunk_size, remainder);
            let out = self.conv_chunk(chunk);
            conv_outputs.push(out);
        }

        let mut x = if conv_outputs.len() == 1 {
            conv_outputs.remove(0)
        } else {
            Tensor::cat(conv_outputs, 1)
        };

        let time = x.dims()[1];
        let d_model = x.dims()[2];
        let pos_emb = sinusoidal_position_embedding::<B>(time, d_model, &device);
        x = x + pos_emb;

        for layer in &self.layers {
            x = layer.forward(x);
        }

        x = self.ln_post.forward(x);
        x = burn::tensor::activation::gelu(self.proj1.forward(x));
        self.proj2.forward(x)
    }

    fn conv_chunk(&self, chunk: Tensor<B, 3>) -> Tensor<B, 3> {
        let x = chunk.unsqueeze_dim::<4>(1);
        let x = burn::tensor::activation::gelu(self.conv2d1.forward(x));
        let x = burn::tensor::activation::gelu(self.conv2d2.forward(x));
        let x = burn::tensor::activation::gelu(self.conv2d3.forward(x));

        let [batch, channels, freq_bins, time] = x.dims();
        let x = x
            .swap_dims(1, 3)
            .swap_dims(2, 3)
            .reshape([batch, time, channels * freq_bins]);
        self.conv_out.forward(x)
    }
}

fn sinusoidal_position_embedding<B: Backend>(
    length: usize,
    channels: usize,
    device: &B::Device,
) -> Tensor<B, 3> {
    let log_timescale_increment = (10000.0f64).ln() / (channels as f64 / 2.0 - 1.0);
    let half_channels = channels / 2;
    let mut emb = Vec::with_capacity(length * channels);
    for pos in 0..length {
        for i in 0..half_channels {
            let inv_timescale = (-log_timescale_increment * i as f64).exp();
            let scaled_time = pos as f64 * inv_timescale;
            emb.push(scaled_time.sin() as f32);
        }
        for i in 0..half_channels {
            let inv_timescale = (-log_timescale_increment * i as f64).exp();
            let scaled_time = pos as f64 * inv_timescale;
            emb.push(scaled_time.cos() as f32);
        }
    }
    Tensor::<B, 1>::from_floats(emb.as_slice(), device)
        .reshape([length, channels])
        .unsqueeze_dim::<3>(0)
}

#[derive(Module, Debug)]
pub struct Qwen3ASRThinkerForConditionalGeneration<B: Backend> {
    pub audio_tower: Qwen3ASRAudioEncoder<B>,
    pub model: Qwen3ASRThinkerTextModel<B>,
    pub lm_head: Linear<B>,
}

#[derive(Config, Debug)]
pub struct Qwen3ASRThinkerForConditionalGenerationConfig {
    audio_tower: Qwen3ASRAudioEncoderModuleConfig,
    model: Qwen3ASRThinkerTextModelConfig,
}

impl Qwen3ASRThinkerForConditionalGenerationConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> Qwen3ASRThinkerForConditionalGeneration<B> {
        Qwen3ASRThinkerForConditionalGeneration {
            audio_tower: self.audio_tower.init(device),
            model: self.model.init(device),
            lm_head: LinearConfig::new(self.model.hidden_size, self.model.vocab_size)
                .with_bias(false)
                .init(device),
        }
    }
}

#[derive(Module, Debug)]
pub struct Qwen3ASR<B: Backend> {
    pub thinker: Qwen3ASRThinkerForConditionalGeneration<B>,
}

#[derive(Config, Debug)]
pub struct Qwen3ASRConfig {
    thinker: Qwen3ASRThinkerForConditionalGenerationConfig,
}

impl Qwen3ASRConfig {
    pub fn from_configs(audio_config: AudioEncoderConfig, text_config: TextConfig) -> Self {
        let eps = text_config.rms_norm_eps;
        Self {
            thinker: Qwen3ASRThinkerForConditionalGenerationConfig::new(
                Qwen3ASRAudioEncoderModuleConfig::new(
                    audio_config.d_model,
                    audio_config.encoder_ffn_dim,
                    audio_config.encoder_layers,
                    audio_config.encoder_attention_heads,
                    audio_config.downsample_hidden_size,
                    audio_config.num_mel_bins,
                    audio_config.output_dim,
                    eps,
                ),
                Qwen3ASRThinkerTextModelConfig::new(
                    text_config.vocab_size,
                    text_config.hidden_size,
                    text_config.intermediate_size,
                    text_config.num_hidden_layers,
                    text_config.num_attention_heads,
                    text_config.num_key_value_heads,
                    text_config.head_dim,
                    eps,
                ),
            ),
        }
    }

    pub fn init<B: Backend>(&self, device: &B::Device) -> Qwen3ASR<B> {
        Qwen3ASR {
            thinker: self.thinker.init(device),
        }
    }
}

pub fn create_mrope(text_config: &TextConfig) -> MRoPE {
    MRoPE::new(
        text_config.head_dim,
        text_config.rope_theta,
        &text_config.mrope_section(),
        text_config.mrope_interleaved(),
    )
}

pub fn create_causal_mask<B: Backend>(
    seq_len: usize,
    past_len: usize,
    device: &B::Device,
) -> Tensor<B, 4, Bool> {
    let total_len = past_len + seq_len;
    let mut values = Vec::with_capacity(seq_len * total_len);
    for row in 0..seq_len {
        let current_pos = past_len + row;
        for col in 0..total_len {
            values.push(col > current_pos);
        }
    }
    let mask = Tensor::<B, 1, Bool>::from_bool(values.as_slice().into(), device)
        .reshape([seq_len, total_len]);
    mask.unsqueeze_dim::<3>(0).unsqueeze_dim::<4>(0)
}
