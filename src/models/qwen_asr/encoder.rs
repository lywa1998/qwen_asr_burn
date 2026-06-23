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
    pub fn forward(&self, hidden_states: Tensor<3>, mask: Option<Tensor<4>>) -> Tensor<3> {
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
        let mut attn_weights = q.matmul(k.swap_dims(2, 3)).div_scalar(scale);
        if let Some(m) = mask {
            attn_weights = attn_weights + m;
        }
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
pub struct Qwen3ASRAudioEncoderLayerConfig {
    d_model: usize,
    encoder_ffn_dim: usize,
    encoder_attention_heads: usize,
    epsilon: f64,
}

impl Qwen3ASRAudioEncoderLayerConfig {
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
    pub fn forward(&self, hidden_states: Tensor<3>, mask: Option<Tensor<4>>) -> Tensor<3> {
        let residual = hidden_states.clone();
        let hidden_states = self.self_attn_layer_norm.forward(hidden_states);
        let hidden_states = self.self_attn.forward(hidden_states, mask);
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
    #[module(skip)]
    n_window: usize,
    #[module(skip)]
    n_window_infer: usize,
    #[module(skip)]
    conv_chunksize: usize,
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
    n_window: usize,
    n_window_infer: usize,
    conv_chunksize: usize,
}

impl Qwen3ASRAudioEncoderConfig {
    pub fn init(&self, device: &burn::tensor::Device) -> Qwen3ASRAudioEncoder {
        let conv_freq_bins = (((self.num_mel_bins + 1) / 2 + 1) / 2 + 1) / 2;
        let conv_out_dim = self.downsample_hidden_size * conv_freq_bins;
        let layer_config = Qwen3ASRAudioEncoderLayerConfig::new(
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
            n_window: self.n_window,
            n_window_infer: self.n_window_infer,
            conv_chunksize: self.conv_chunksize,
        }
    }
}

/// Output length of one Conv2d layer: kernel=3, stride=2, padding=1.
fn conv_out_len(input_len: usize) -> usize {
    (input_len - 1) / 2 + 1
}

/// Output length after 3 stacked Conv2d layers.
fn feat_extract_output_length(input_len: usize) -> usize {
    conv_out_len(conv_out_len(conv_out_len(input_len)))
}

/// Build a block-diagonal attention mask from cumulative segment lengths.
/// Each block gets zero attention (bidirectional within block); between blocks
/// the mask value is -inf.  mirrors Python's `_prepare_attention_mask` in
/// `modeling_qwen3_asr.py`.
fn build_block_diagonal_mask(
    seq_len: usize,
    cu_seqlens: &[usize],
    device: &burn::tensor::Device,
) -> Tensor<4> {
    let mut values = vec![f32::NEG_INFINITY; seq_len * seq_len];
    for w in 1..cu_seqlens.len() {
        let start = cu_seqlens[w - 1];
        let end = cu_seqlens[w];
        for i in start..end {
            for j in start..end {
                values[i * seq_len + j] = 0.0;
            }
        }
    }
    Tensor::<1>::from_floats(values.as_slice(), device)
        .reshape([1, 1, seq_len, seq_len])
}

impl Qwen3ASRAudioEncoder {
    pub fn forward(&self, mel: Tensor<3>) -> Tensor<3> {
        let [_batch, n_mels, mel_time] = mel.dims();
        let device = mel.device();
        let chunk_size = self.n_window * 2; // 100 mel frames per chunk
        let num_chunks = mel_time.div_ceil(chunk_size);

        // 1. Split mel into chunks, padding the last to chunk_size.
        let mut chunk_lengths = Vec::with_capacity(num_chunks);
        let mut padded_chunks = Vec::with_capacity(num_chunks);
        for i in 0..num_chunks {
            let start = i * chunk_size;
            let end = (start + chunk_size).min(mel_time);
            let len = end - start;
            chunk_lengths.push(len);
            let chunk = mel.clone().narrow(2, start, len);
            if len < chunk_size {
                let pad = Tensor::zeros([1, n_mels, chunk_size - len], &device);
                padded_chunks.push(Tensor::cat(vec![chunk, pad], 2));
            } else {
                padded_chunks.push(chunk);
            }
        }
        // [num_chunks, n_mels, chunk_size] → [num_chunks, 1, n_mels, chunk_size]
        let padded = Tensor::cat(padded_chunks, 0).unsqueeze_dim::<4>(1);

        // 2. Run conv on padded batch (split into conv_chunksize groups).
        let mut conv_outputs = Vec::new();
        for i in (0..num_chunks).step_by(self.conv_chunksize) {
            let end = (i + self.conv_chunksize).min(num_chunks);
            let group = padded.clone().narrow(0, i, end - i);
            conv_outputs.push(self.conv_chunk(group));
        }
        // [num_chunks, padded_time_after_cnn, d_model]
        let padded_embed = Tensor::cat(conv_outputs, 0);

        // 3. Add sinusoidal position embedding – positions 0..padded_time,
        //    shared across all chunks (Python reference: same for every chunk).
        let [num_chunks_actual, padded_time, d_model] = padded_embed.dims();
        let pos_emb = sinusoidal_position_embedding(padded_time, d_model, &device);
        let padded_embed = padded_embed + pos_emb;

        // 4. Compute per-chunk after-CNN lengths, flatten only valid positions.
        let after_cnn_lens: Vec<usize> = chunk_lengths
            .iter()
            .map(|&l| feat_extract_output_length(l))
            .collect();
        let mut hidden_vec = Vec::with_capacity(num_chunks_actual);
        let mut total_valid = 0usize;
        for i in 0..num_chunks_actual {
            let valid_len = after_cnn_lens[i];
            let chunk = padded_embed.clone().narrow(0, i, 1).narrow(1, 0, valid_len);
            hidden_vec.push(chunk.reshape([valid_len, d_model]));
            total_valid += valid_len;
        }
        let total_valid = total_valid; // no longer mutable
        let hidden_states = Tensor::cat(hidden_vec, 0).unsqueeze_dim::<3>(0); // [1, total_valid, d_model]

        // 5. Build cu_seqlens for block-diagonal attention.
        let window_aftercnn = padded_time * (self.n_window_infer / chunk_size);
        let mut cu_chunk_lens: Vec<usize> = Vec::new();
        for &cnn_len in &after_cnn_lens {
            let num_windows = cnn_len / window_aftercnn;
            for _ in 0..num_windows {
                cu_chunk_lens.push(window_aftercnn);
            }
            let rem = cnn_len % window_aftercnn;
            if rem > 0 {
                cu_chunk_lens.push(rem);
            }
        }
        let mut cu_seqlens = vec![0usize];
        let mut cum = 0usize;
        for &l in &cu_chunk_lens {
            cum += l;
            cu_seqlens.push(cum);
        }
        assert_eq!(cum, total_valid, "cu_seqlens cum mismatch");

        // 6. Build 4-D block-diagonal mask and run encoder layers.
        let mask = build_block_diagonal_mask(total_valid, &cu_seqlens, &device);
        let mut hidden_states = hidden_states;
        for layer in &self.layers {
            hidden_states = layer.forward(hidden_states, Some(mask.clone()));
        }

        // 7. Output projection.
        hidden_states = self.ln_post.forward(hidden_states);
        hidden_states = burn::tensor::activation::gelu(self.proj1.forward(hidden_states));
        self.proj2.forward(hidden_states)
    }

    fn conv_chunk(&self, x: Tensor<4>) -> Tensor<3> {
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