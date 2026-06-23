use burn::config::Config;
use burn::module::Module;
use burn::nn::conv::Conv2dConfig;
use burn::nn::conv::Conv2d;
use burn::nn::PaddingConfig2d;
use burn::nn::LinearConfig;
use burn::nn::Linear;
use burn::tensor::Tensor;

use super::norm::{Qwen3ASRLayerNorm, Qwen3ASRLayerNormConfig};

#[derive(Module, Debug)]
pub struct Qwen3ASRAudioAttention {
    pub q_proj: Linear,
    pub k_proj: Linear,
    pub v_proj: Linear,
    pub out_proj: Linear,
    #[module(skip)]
    num_heads: usize,
}

#[derive(Config, Debug)]
pub struct Qwen3ASRAudioAttentionConfig {
    d_model: usize,
    num_heads: usize,
}

impl Qwen3ASRAudioAttentionConfig {
    pub fn init(&self, device: &burn::tensor::Device) -> Qwen3ASRAudioAttention {
        Qwen3ASRAudioAttention {
            q_proj: LinearConfig::new(self.d_model, self.d_model).init(device),
            k_proj: LinearConfig::new(self.d_model, self.d_model).init(device),
            v_proj: LinearConfig::new(self.d_model, self.d_model).init(device),
            out_proj: LinearConfig::new(self.d_model, self.d_model).init(device),
            num_heads: self.num_heads,
        }
    }
}

impl Qwen3ASRAudioAttention {
    pub fn forward(&self, hidden_states: Tensor<3>) -> Tensor<3> {
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
pub struct Qwen3ASRAudioEncoderLayer {
    pub self_attn_layer_norm: Qwen3ASRLayerNorm,
    pub self_attn: Qwen3ASRAudioAttention,
    pub fc1: Linear,
    pub fc2: Linear,
    pub final_layer_norm: Qwen3ASRLayerNorm,
}

#[derive(Config, Debug)]
pub struct Qwen3ASRQwen3ASRAudioEncoderLayerConfig {
    d_model: usize,
    encoder_ffn_dim: usize,
    encoder_attention_heads: usize,
    epsilon: f64,
}

impl Qwen3ASRQwen3ASRAudioEncoderLayerConfig {
    pub fn init(&self, device: &burn::tensor::Device) -> Qwen3ASRAudioEncoderLayer {
        Qwen3ASRAudioEncoderLayer {
            self_attn_layer_norm: Qwen3ASRLayerNormConfig::new(self.d_model, self.epsilon).init(device),
            self_attn: Qwen3ASRAudioAttentionConfig::new(self.d_model, self.encoder_attention_heads)
                .init(device),
            fc1: LinearConfig::new(self.d_model, self.encoder_ffn_dim).init(device),
            fc2: LinearConfig::new(self.encoder_ffn_dim, self.d_model).init(device),
            final_layer_norm: Qwen3ASRLayerNormConfig::new(self.d_model, self.epsilon).init(device),
        }
    }
}

impl Qwen3ASRAudioEncoderLayer {
    pub fn forward(&self, hidden_states: Tensor<3>) -> Tensor<3> {
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
pub struct Qwen3ASRAudioEncoder {
    pub conv2d1: Conv2d,
    pub conv2d2: Conv2d,
    pub conv2d3: Conv2d,
    pub conv_out: Linear,
    pub layers: Vec<Qwen3ASRAudioEncoderLayer>,
    pub ln_post: Qwen3ASRLayerNorm,
    pub proj1: Linear,
    pub proj2: Linear,
}

#[derive(Config, Debug)]
pub struct Qwen3ASRAudioEncoderConfig {
    d_model: usize,
    encoder_ffn_dim: usize,
    encoder_layers: usize,
    encoder_attention_heads: usize,
    downsample_hidden_size: usize,
    num_mel_bins: usize,
    output_dim: usize,
    epsilon: f64,
}

impl Qwen3ASRAudioEncoderConfig {
    pub fn init(&self, device: &burn::tensor::Device) -> Qwen3ASRAudioEncoder {
        let conv_freq_bins = (((self.num_mel_bins + 1) / 2 + 1) / 2 + 1) / 2;
        let conv_out_dim = self.downsample_hidden_size * conv_freq_bins;
        let layer_config = Qwen3ASRQwen3ASRAudioEncoderLayerConfig::new(
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
                .with_bias(true)
                .init(device),
            conv2d2: Conv2dConfig::new(
                [self.downsample_hidden_size, self.downsample_hidden_size],
                [3, 3],
            )
            .with_stride([2, 2])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_bias(true)
            .init(device),
            conv2d3: Conv2dConfig::new(
                [self.downsample_hidden_size, self.downsample_hidden_size],
                [3, 3],
            )
            .with_stride([2, 2])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_bias(true)
            .init(device),
            conv_out: LinearConfig::new(conv_out_dim, self.d_model)
                .with_bias(false)
                .init(device),
            layers,
            ln_post: Qwen3ASRLayerNormConfig::new(self.d_model, self.epsilon).init(device),
            proj1: LinearConfig::new(self.d_model, self.d_model).init(device),
            proj2: LinearConfig::new(self.d_model, self.output_dim).init(device),
        }
    }
}

impl Qwen3ASRAudioEncoder {
    pub fn forward(&self, mel: Tensor<3>) -> Tensor<3> {
        let [_batch, _n_mels, mel_time] = mel.dims();
        let chunk_size: usize = 100;
        let num_full_chunks = mel_time / chunk_size;
        let remainder = mel_time % chunk_size;

        let device = mel.device();
        let mut conv_outputs: Vec<Tensor<3>> = Vec::new();

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
        let pos_emb = sinusoidal_position_embedding(time, d_model, &device);
        x = x + pos_emb;

        for layer in &self.layers {
            x = layer.forward(x);
        }

        x = self.ln_post.forward(x);
        x = burn::tensor::activation::gelu(self.proj1.forward(x));
        self.proj2.forward(x)
    }

    fn conv_chunk(&self, chunk: Tensor<3>) -> Tensor<3> {
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

fn sinusoidal_position_embedding(
    length: usize,
    channels: usize,
    device: &burn::tensor::Device,
) -> Tensor<3> {
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
    Tensor::<1>::from_floats(emb.as_slice(), device)
        .reshape([length, channels])
        .unsqueeze_dim::<3>(0)
}
