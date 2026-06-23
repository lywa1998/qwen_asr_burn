use burn::config::Config;
use burn::module::Module;
use burn::nn::LinearConfig;
use burn::nn::Linear;
use burn::tensor::{Bool, Tensor};

use super::norm::{Qwen3ASRQKNorm, Qwen3ASRQKNormConfig};
use super::rope::apply_mrope_simple;

#[derive(Debug)]
pub struct KvCacheEntry {
    pub k: Tensor<4>,
    pub v: Tensor<4>,
}

#[derive(Debug)]
pub struct KvCache {
    layers: Vec<Option<KvCacheEntry>>,
}

impl KvCache {
    pub fn new(num_layers: usize) -> Self {
        Self {
            layers: (0..num_layers).map(|_| None).collect(),
        }
    }

    pub fn layer(&self, index: usize) -> Option<&KvCacheEntry> {
        self.layers.get(index).and_then(|entry| entry.as_ref())
    }

    pub fn set_layer(&mut self, index: usize, entry: KvCacheEntry) {
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
pub struct Qwen3ASRAttention {
    pub q_proj: Linear,
    pub k_proj: Linear,
    pub v_proj: Linear,
    pub o_proj: Linear,
    pub q_norm: Qwen3ASRQKNorm,
    pub k_norm: Qwen3ASRQKNorm,
    #[module(skip)]
    num_q_heads: usize,
    #[module(skip)]
    num_kv_heads: usize,
    #[module(skip)]
    head_dim: usize,
}

#[derive(Config, Debug)]
pub struct Qwen3ASRAttentionConfig {
    hidden_size: usize,
    num_q_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    epsilon: f64,
}

impl Qwen3ASRAttentionConfig {
    pub fn init(&self, device: &burn::tensor::Device) -> Qwen3ASRAttention {
        Qwen3ASRAttention {
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
            q_norm: Qwen3ASRQKNormConfig::new(self.head_dim, self.epsilon).init(device),
            k_norm: Qwen3ASRQKNormConfig::new(self.head_dim, self.epsilon).init(device),
            num_q_heads: self.num_q_heads,
            num_kv_heads: self.num_kv_heads,
            head_dim: self.head_dim,
        }
    }
}

impl Qwen3ASRAttention {
    pub fn forward(
        &self,
        hidden_states: Tensor<3>,
        cos: &Tensor<4>,
        sin: &Tensor<4>,
        causal_mask: Option<Tensor<4, Bool>>,
        kv_cache: Option<&KvCacheEntry>,
    ) -> (Tensor<3>, KvCacheEntry) {
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

pub fn repeat_kv(x: Tensor<4>, n_rep: usize) -> Tensor<4> {
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

pub fn create_causal_mask(
    seq_len: usize,
    past_len: usize,
    device: &burn::tensor::Device,
) -> Tensor<4, Bool> {
    let total_len = past_len + seq_len;
    let mut values = Vec::with_capacity(seq_len * total_len);
    for row in 0..seq_len {
        let current_pos = past_len + row;
        for col in 0..total_len {
            values.push(col > current_pos);
        }
    }
    let data: burn::tensor::TensorData = values.as_slice().into();
    let mask = Tensor::<1, Bool>::from_bool(data, device)
        .reshape([seq_len, total_len]);
    mask.unsqueeze_dim::<3>(0).unsqueeze_dim::<4>(0)
}
