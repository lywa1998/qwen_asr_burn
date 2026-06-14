use burn::module::{Module, Param};
use burn::nn::{
    conv::Conv2dConfig, EmbeddingConfig, LinearConfig,
};
use burn::nn::conv::Conv2d;
use burn::nn::PaddingConfig2d;
use burn::nn::{Embedding, Linear};
use burn::tensor::backend::Backend;
use burn::tensor::{Bool, Int, Tensor};

use crate::config::{AudioEncoderConfig, TextConfig};

// ============================================================
// Custom RMS Norm (uses "weight" field to match PyTorch naming)
// ============================================================

#[derive(Module, Debug)]
pub struct MyRmsNorm<B: Backend> {
    pub weight: Param<Tensor<B, 1>>,
    #[module(skip)]
    epsilon: f64,
}

impl<B: Backend> MyRmsNorm<B> {
    pub fn new(d_model: usize, epsilon: f64, device: &B::Device) -> Self {
        Self { weight: Param::from_tensor(Tensor::ones([d_model], device)), epsilon }
    }

    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let rms = x.clone().powf_scalar(2.0).mean_dim(2).add_scalar(self.epsilon).sqrt();
        let w = self.weight.val().unsqueeze_dim::<2>(0).unsqueeze_dim::<3>(1);
        x.div(rms).mul(w)
    }
}

// ============================================================
// Custom Layer Norm (uses "gamma"/"beta" to match PyTorch Burn convention)
// ============================================================

#[derive(Module, Debug)]
pub struct MyLayerNorm<B: Backend> {
    pub weight: Param<Tensor<B, 1>>,
    pub bias: Param<Tensor<B, 1>>,
    #[module(skip)]
    epsilon: f64,
}

impl<B: Backend> MyLayerNorm<B> {
    pub fn new(d_model: usize, epsilon: f64, device: &B::Device) -> Self {
        Self {
            weight: Param::from_tensor(Tensor::ones([d_model], device)),
            bias: Param::from_tensor(Tensor::zeros([d_model], device)),
            epsilon,
        }
    }

    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let ndim = 3;
        let mean = x.clone().mean_dim(ndim - 1);
        let var = x.clone().sub(mean.clone()).powf_scalar(2.0).mean_dim(ndim - 1);
        let x_norm = x.sub(mean).div(var.add_scalar(self.epsilon).sqrt());

        let w = self.weight.val().unsqueeze_dim::<2>(0).unsqueeze_dim::<3>(1);
        let b = self.bias.val().unsqueeze_dim::<2>(0).unsqueeze_dim::<3>(1);
        x_norm.mul(w).add(b)
    }
}

// ============================================================
// MRoPE
// ============================================================

/// MRoPE matching Qwen3ASRThinkerTextRotaryEmbedding in Python reference
pub struct MRoPE {
    inv_freq: Vec<f32>,         // head_dim/2 frequencies, same denominator for all sections
    pub total_rotary_dim: usize,
}

impl MRoPE {
    pub fn new(head_dim: usize, rope_theta: f64, _mrope_section: &[usize]) -> Self {
        // Single inv_freq using head_dim as denominator (matches Python rope_init_fn default)
        let head_dim = head_dim; // 128 for Qwen3-0.6B
        let mut inv_freq = Vec::new();
        for i in (0..head_dim).step_by(2) {
            let freq = 1.0 / (rope_theta.powf(i as f64 / head_dim as f64));
            inv_freq.push(freq as f32);
        }
        let total_rotary_dim = head_dim / 2 * 2; // 64 for head_dim=128
        Self { inv_freq, total_rotary_dim }
    }

    pub fn compute_cos_sin<B: Backend>(
        &self,
        position_ids: Tensor<B, 2, Int>,
    ) -> (Tensor<B, 4>, Tensor<B, 4>) {
        let [_batch, seq_len] = position_ids.dims();
        let device = position_ids.device();

        // position_ids: [B, seq_len] → [B, seq_len, 1]
        let pos = position_ids
            .unsqueeze_dim::<3>(2)
            .float();

        // inv_freq: [num_pairs] → [1, 1, num_pairs]
        let inv_freq_t = Tensor::<B, 1>::from_floats(self.inv_freq.as_slice(), &device)
            .unsqueeze_dim::<2>(0)
            .unsqueeze_dim::<3>(1);

        // freqs: [B, seq_len, num_pairs=64]
        let freqs = pos.mul(inv_freq_t);

        // Duplicate freqs: matching Python torch.cat((freqs, freqs), dim=-1)
        let emb = Tensor::cat(vec![freqs.clone(), freqs.clone()], 2);
        let cos = emb.clone().cos().unsqueeze_dim::<4>(1);
        let sin = emb.sin().unsqueeze_dim::<4>(1);

        (cos, sin)
    }
}

/// Apply RoPE rotation. Matches Python's apply_rotary_pos_emb:
/// - q_embed = (q * cos) + (rotate_half(q) * sin)
/// - rotate_half splits at head_dim/2 and swaps halves with negation
/// - cos/sin should cover the full head_dim
fn apply_mrope_simple<B: Backend>(
    x: Tensor<B, 4>,
    cos: &Tensor<B, 4>,
    sin: &Tensor<B, 4>,
) -> Tensor<B, 4> {
    // Python's rotate_half: split at half, cat((-x2, x1))
    let head_dim = x.dims()[3];
    let half = head_dim / 2;
    let x_clone = x.clone();
    let x1 = x_clone.clone().narrow(3, 0, half);
    let x2 = x_clone.narrow(3, half, half);
    let rotate_half = Tensor::cat(vec![x2.neg(), x1], 3);
    // q_embed = (q * cos) + (rotate_half(q) * sin)
    x * cos.clone() + rotate_half * sin.clone()
}

// ============================================================
// QK Normalization
// ============================================================

#[derive(Module, Debug)]
pub struct QKNorm<B: Backend> {
    pub weight: Param<Tensor<B, 1>>,
    #[module(skip)]
    epsilon: f64,
}

impl<B: Backend> QKNorm<B> {
    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let rms = x.clone().powf_scalar(2.0).mean_dim(3).add_scalar(self.epsilon).sqrt();
        let w = self.weight.val().unsqueeze_dim::<2>(0).unsqueeze_dim::<3>(1).unsqueeze_dim::<4>(2);
        x.div(rms).mul(w)
    }
}

// ============================================================
// Qwen3 Attention
// ============================================================

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

impl<B: Backend> Qwen3Attention<B> {
    pub fn forward(
        &self,
        hidden_states: Tensor<B, 3>,
        mrope: &MRoPE,
        causal_mask: Option<Tensor<B, 4, Bool>>,
    ) -> Tensor<B, 3> {
        let [batch, seq_len, _hidden] = hidden_states.dims();
        let num_q_heads = self.num_q_heads;
        let num_kv_heads = self.num_kv_heads;
        let head_dim = self.head_dim;

        let pos_ids = make_positions::<B>(seq_len, &hidden_states.device());
        let (cos, sin) = mrope.compute_cos_sin(pos_ids);

        let q = self.q_proj.forward(hidden_states.clone());
        let k = self.k_proj.forward(hidden_states.clone());
        let v = self.v_proj.forward(hidden_states);

        let q = q.reshape([batch, seq_len, num_q_heads, head_dim]).swap_dims(1, 2);
        let k = k.reshape([batch, seq_len, num_kv_heads, head_dim]).swap_dims(1, 2);
        let v = v.reshape([batch, seq_len, num_kv_heads, head_dim]).swap_dims(1, 2);

        let q = self.q_norm.forward(q);
        let k = self.k_norm.forward(k);

        let q = apply_mrope_simple(q, &cos, &sin);
        let k = apply_mrope_simple(k, &cos, &sin);

        let k = repeat_kv(k, 2);
        let v = repeat_kv(v, 2);

        let scale = (head_dim as f64).sqrt();
        let attn_weights = q.matmul(k.swap_dims(2, 3)).div_scalar(scale);

        let attn_weights = if let Some(mask) = causal_mask {
            attn_weights.mask_fill(mask, f32::NEG_INFINITY)
        } else {
            attn_weights
        };

        let attn_weights = burn::tensor::activation::softmax(attn_weights, 3);
        let attn_output = attn_weights.matmul(v);

        let attn_output = attn_output.swap_dims(1, 2).reshape([batch, seq_len, num_q_heads * head_dim]);
        self.o_proj.forward(attn_output)
    }
}

fn repeat_kv<B: Backend>(x: Tensor<B, 4>, n_rep: usize) -> Tensor<B, 4> {
    if n_rep == 1 { return x; }
    let [batch, num_kv_heads, seq_len, head_dim] = x.dims();
    x.unsqueeze_dim::<5>(2).repeat_dim(2, n_rep)
        .reshape([batch, num_kv_heads * n_rep, seq_len, head_dim])
}

fn make_positions<B: Backend>(seq_len: usize, device: &B::Device) -> Tensor<B, 2, Int> {
    let vals: Vec<i32> = (0..seq_len as i32).collect();
    Tensor::<B, 1, Int>::from_ints(vals.as_slice(), device).unsqueeze_dim::<2>(0)
}

// ============================================================
// Qwen3 SwiGLU MLP (no biases in decoder Linear layers)
// ============================================================

#[derive(Module, Debug)]
pub struct Qwen3MLP<B: Backend> {
    pub gate_proj: Linear<B>,
    pub up_proj: Linear<B>,
    pub down_proj: Linear<B>,
}

impl<B: Backend> Qwen3MLP<B> {
    pub fn forward<const D: usize>(&self, x: Tensor<B, D>) -> Tensor<B, D> {
        let gate = self.gate_proj.forward(x.clone());
        let gate = gate.clone().mul(burn::tensor::activation::sigmoid(gate));
        let up = self.up_proj.forward(x);
        self.down_proj.forward(gate.mul(up))
    }
}

// ============================================================
// Qwen3 Decoder Layer
// ============================================================

#[derive(Module, Debug)]
pub struct Qwen3DecoderLayer<B: Backend> {
    pub input_layernorm: MyRmsNorm<B>,
    pub self_attn: Qwen3Attention<B>,
    pub post_attention_layernorm: MyRmsNorm<B>,
    pub mlp: Qwen3MLP<B>,
}

impl<B: Backend> Qwen3DecoderLayer<B> {
    pub fn forward(
        &self,
        hidden_states: Tensor<B, 3>,
        mrope: &MRoPE,
        causal_mask: Option<Tensor<B, 4, Bool>>,
    ) -> Tensor<B, 3> {
        let residual = hidden_states.clone();
        let hidden_states = self.input_layernorm.forward(hidden_states);
        let hidden_states = self.self_attn.forward(hidden_states, mrope, causal_mask);
        let hidden_states = hidden_states.add(residual);

        let residual = hidden_states.clone();
        let hidden_states = self.post_attention_layernorm.forward(hidden_states);
        let hidden_states = self.mlp.forward(hidden_states);
        hidden_states.add(residual)
    }
}

// ============================================================
// Qwen3 Decoder
// ============================================================

#[derive(Module, Debug)]
pub struct Qwen3Model<B: Backend> {
    pub embed_tokens: Embedding<B>,
    pub layers: Vec<Qwen3DecoderLayer<B>>,
    pub norm: MyRmsNorm<B>,
}

impl<B: Backend> Qwen3Model<B> {
    pub fn forward_embeds(
        &self,
        hidden_states: Tensor<B, 3>,
        mrope: &MRoPE,
        causal_mask: Option<Tensor<B, 4, Bool>>,
    ) -> Tensor<B, 3> {
        let mut hidden_states = hidden_states;
        for layer in &self.layers {
            hidden_states = layer.forward(hidden_states, mrope, causal_mask.clone());
        }
        self.norm.forward(hidden_states)
    }
}

// ============================================================
// Audio Encoder Attention
// ============================================================

#[derive(Module, Debug)]
pub struct EncoderAttention<B: Backend> {
    pub q_proj: Linear<B>,
    pub k_proj: Linear<B>,
    pub v_proj: Linear<B>,
    pub out_proj: Linear<B>,
    #[module(skip)]
    num_heads: usize,
}

impl<B: Backend> EncoderAttention<B> {
    pub fn forward(&self, hidden_states: Tensor<B, 3>) -> Tensor<B, 3> {
        let [batch, seq_len, d_model] = hidden_states.dims();
        let num_heads = self.num_heads;
        let head_dim = d_model / num_heads;

        let q = self.q_proj.forward(hidden_states.clone());
        let k = self.k_proj.forward(hidden_states.clone());
        let v = self.v_proj.forward(hidden_states);

        let q = q.reshape([batch, seq_len, num_heads, head_dim]).swap_dims(1, 2);
        let k = k.reshape([batch, seq_len, num_heads, head_dim]).swap_dims(1, 2);
        let v = v.reshape([batch, seq_len, num_heads, head_dim]).swap_dims(1, 2);

        let scale = (head_dim as f64).sqrt();
        let attn_weights = q.matmul(k.swap_dims(2, 3)).div_scalar(scale);
        let attn_weights = burn::tensor::activation::softmax(attn_weights, 3);
        let attn_output = attn_weights.matmul(v);

        let attn_output = attn_output.swap_dims(1, 2).reshape([batch, seq_len, d_model]);
        self.out_proj.forward(attn_output)
    }
}

// ============================================================
// Audio Encoder Layer
// ============================================================

#[derive(Module, Debug)]
pub struct AudioEncoderLayer<B: Backend> {
    pub self_attn_layer_norm: MyLayerNorm<B>,
    pub self_attn: EncoderAttention<B>,
    pub fc1: Linear<B>,
    pub fc2: Linear<B>,
    pub final_layer_norm: MyLayerNorm<B>,
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

// ============================================================
// Audio Tower (layers directly, no encoder wrapper)
// ============================================================

#[derive(Module, Debug)]
pub struct AudioTower<B: Backend> {
    pub conv2d1: Conv2d<B>,
    pub conv2d2: Conv2d<B>,
    pub conv2d3: Conv2d<B>,
    pub conv_out: Linear<B>,
    pub layers: Vec<AudioEncoderLayer<B>>,
    pub ln_post: MyLayerNorm<B>,
    pub proj1: Linear<B>,
    pub proj2: Linear<B>,
}

impl<B: Backend> AudioTower<B> {
    pub fn forward(&self, mel: Tensor<B, 3>) -> Tensor<B, 3> {
        // Chunked convolution matching Python Qwen3ASRAudioEncoder:
        // Split mel into chunks of n_window*2 = 100 frames, process each chunk
        // independently through conv, then concatenate. This matches the Python
        // chunking behavior which produces different output lengths due to per-chunk padding.
        let [batch, _n_mels, mel_time] = mel.dims();
        let chunk_size: usize = 100; // n_window * 2
        let num_full_chunks = mel_time / chunk_size;
        let remainder = mel_time % chunk_size;

        let device = mel.device();

        let mut conv_outputs: Vec<Tensor<B, 3>> = Vec::new();

        // Process full chunks
        for i in 0..num_full_chunks {
            let chunk = mel.clone().narrow(2, i * chunk_size, chunk_size);
            let out = self.conv_chunk(chunk);
            conv_outputs.push(out);
        }
        // Process remainder chunk
        if remainder > 0 {
            let chunk = mel.narrow(2, num_full_chunks * chunk_size, remainder);
            let out = self.conv_chunk(chunk);
            conv_outputs.push(out);
        }

        // Concatenate all chunk outputs along time dimension
        let mut x = if conv_outputs.len() == 1 {
            conv_outputs.remove(0)
        } else {
            Tensor::cat(conv_outputs, 1)
        };

        // Add sinusoidal positional embedding
        let time = x.dims()[1];
        let d_model = x.dims()[2];
        let pos_emb = sinusoidal_position_embedding::<B>(time, d_model, &device);
        x = x + pos_emb;

        // Encoder layers with full self-attention (no chunk masking needed
        // since our audio is short enough to fit in one attention group)
        for layer in &self.layers {
            x = layer.forward(x);
        }

        x = self.ln_post.forward(x);
        x = burn::tensor::activation::gelu(self.proj1.forward(x));
        self.proj2.forward(x)
    }

    /// Process a single mel chunk through the conv layers and conv_out projection.
    fn conv_chunk(&self, chunk: Tensor<B, 3>) -> Tensor<B, 3> {
        let x = chunk.unsqueeze_dim::<4>(1);
        let x = burn::tensor::activation::gelu(self.conv2d1.forward(x));
        let x = burn::tensor::activation::gelu(self.conv2d2.forward(x));
        let x = burn::tensor::activation::gelu(self.conv2d3.forward(x));

        let [batch, channels, freq_bins, time] = x.dims();
        let x = x.swap_dims(1, 3).swap_dims(2, 3).reshape([batch, time, channels * freq_bins]);
        self.conv_out.forward(x)
    }
}

fn sinusoidal_position_embedding<B: Backend>(length: usize, channels: usize, device: &B::Device) -> Tensor<B, 3> {
    let log_timescale_increment = (10000.0f64).ln() / (channels as f64 / 2.0 - 1.0);
    let half_channels = channels / 2;
    let mut emb = Vec::with_capacity(length * channels);
    for pos in 0..length {
        // First half: all sin values (matches Python torch.cat([sin, cos]))
        for i in 0..half_channels {
            let inv_timescale = (-log_timescale_increment * i as f64).exp();
            let scaled_time = pos as f64 * inv_timescale;
            emb.push(scaled_time.sin() as f32);
        }
        // Second half: all cos values
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

// ============================================================
// Thinker + Top-level wrapper
// ============================================================

#[derive(Module, Debug)]
pub struct Thinker<B: Backend> {
    pub audio_tower: AudioTower<B>,
    pub model: Qwen3Model<B>,
    pub lm_head: Linear<B>,
}

#[derive(Module, Debug)]
pub struct Qwen3ASR<B: Backend> {
    pub thinker: Thinker<B>,
}

// ============================================================
// Config
// ============================================================

pub struct Qwen3ASRConfig {
    pub audio_config: AudioEncoderConfig,
    pub text_config: TextConfig,
}

impl Qwen3ASRConfig {
    pub fn new(audio_config: AudioEncoderConfig, text_config: TextConfig) -> Self {
        Self { audio_config, text_config }
    }

    pub fn init<B: Backend>(&self, device: &B::Device) -> Qwen3ASR<B> {
        let d = self.audio_config.d_model;
        let ffn = self.audio_config.encoder_ffn_dim;
        let ds = self.audio_config.downsample_hidden_size;
        let num_mel_bins = self.audio_config.num_mel_bins;
        let conv_freq_bins = ((((num_mel_bins + 1) / 2 + 1) / 2 + 1) / 2);
        let conv_out_dim = ds * conv_freq_bins;
        let eps = self.text_config.rms_norm_eps;

        let encoder_layers: Vec<AudioEncoderLayer<B>> = (0..self.audio_config.encoder_layers)
            .map(|_| AudioEncoderLayer {
                self_attn_layer_norm: MyLayerNorm::new(d, eps, device),
                self_attn: EncoderAttention {
                    q_proj: LinearConfig::new(d, d).init(device),
                    k_proj: LinearConfig::new(d, d).init(device),
                    v_proj: LinearConfig::new(d, d).init(device),
                    out_proj: LinearConfig::new(d, d).init(device),
                    num_heads: self.audio_config.encoder_attention_heads,
                },
                fc1: LinearConfig::new(d, ffn).init(device),
                fc2: LinearConfig::new(ffn, d).init(device),
                final_layer_norm: MyLayerNorm::new(d, eps, device),
            })
            .collect();

        let hidden = self.text_config.hidden_size;
        let intermediate = self.text_config.intermediate_size;
        let n_q = self.text_config.num_attention_heads;
        let n_kv = self.text_config.num_key_value_heads;
        let hd = self.text_config.head_dim;

        let decoder_layers: Vec<Qwen3DecoderLayer<B>> =
            (0..self.text_config.num_hidden_layers)
                .map(|_| Qwen3DecoderLayer {
                    input_layernorm: MyRmsNorm::new(hidden, eps, device),
                    self_attn: Qwen3Attention {
                        q_proj: LinearConfig::new(hidden, n_q * hd).with_bias(false).init(device),
                        k_proj: LinearConfig::new(hidden, n_kv * hd).with_bias(false).init(device),
                        v_proj: LinearConfig::new(hidden, n_kv * hd).with_bias(false).init(device),
                        o_proj: LinearConfig::new(n_q * hd, hidden).with_bias(false).init(device),
                        q_norm: QKNorm { weight: Param::from_tensor(Tensor::ones([hd], device)), epsilon: eps },
                        k_norm: QKNorm { weight: Param::from_tensor(Tensor::ones([hd], device)), epsilon: eps },
                        num_q_heads: n_q,
                        num_kv_heads: n_kv,
                        head_dim: hd,
                    },
                    post_attention_layernorm: MyRmsNorm::new(hidden, eps, device),
                    mlp: Qwen3MLP {
                        gate_proj: LinearConfig::new(hidden, intermediate).with_bias(false).init(device),
                        up_proj: LinearConfig::new(hidden, intermediate).with_bias(false).init(device),
                        down_proj: LinearConfig::new(intermediate, hidden).with_bias(false).init(device),
                    },
                })
                .collect();

        Qwen3ASR {
            thinker: Thinker {
                audio_tower: AudioTower {
                    conv2d1: Conv2dConfig::new([1, ds], [3, 3])
                        .with_stride([2, 2])
                        .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
                        .init(device),
                    conv2d2: Conv2dConfig::new([ds, ds], [3, 3])
                        .with_stride([2, 2])
                        .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
                        .init(device),
                    conv2d3: Conv2dConfig::new([ds, ds], [3, 3])
                        .with_stride([2, 2])
                        .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
                        .init(device),
                    conv_out: LinearConfig::new(conv_out_dim, d).with_bias(false).init(device),
                    layers: encoder_layers,
                    ln_post: MyLayerNorm::new(d, eps, device),
                    proj1: LinearConfig::new(d, d).init(device),
                    proj2: LinearConfig::new(d, self.audio_config.output_dim).init(device),
                },
                model: Qwen3Model {
                    embed_tokens: EmbeddingConfig::new(self.text_config.vocab_size, hidden).init(device),
                    layers: decoder_layers,
                    norm: MyRmsNorm::new(hidden, eps, device),
                },
                lm_head: LinearConfig::new(hidden, self.text_config.vocab_size).with_bias(false).init(device),
            },
        }
    }
}

// ============================================================
// Helpers
// ============================================================

pub fn create_mrope(text_config: &TextConfig) -> MRoPE {
    let mrope_section = text_config.rope_scaling.as_ref()
        .and_then(|rs| if rs.mrope_section.is_empty() { None } else { Some(rs.mrope_section.clone()) })
        .unwrap_or_else(|| vec![24, 20, 20]);
    MRoPE::new(text_config.head_dim, text_config.rope_theta, &mrope_section)
}

pub fn create_causal_mask<B: Backend>(seq_len: usize, device: &B::Device) -> Tensor<B, 4, Bool> {
    // Create upper triangular mask (True = masked/blocked, i.e., j > i)
    // Start with ones, then triu(1) keeps ones above diagonal, zeros below
    let mask: Tensor<B, 2> = Tensor::ones([seq_len, seq_len], device);
    let mask = mask.triu(1); // keep elements above diagonal (j > i)
    let mask: Tensor<B, 2, Bool> = mask.equal_elem(1.0f64);
    let mask: Tensor<B, 3, Bool> = mask.unsqueeze_dim::<3>(0);
    mask.unsqueeze_dim::<4>(0)
}
