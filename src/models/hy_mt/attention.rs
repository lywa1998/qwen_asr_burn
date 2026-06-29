use burn::module::Module;
use burn::nn;
use burn::tensor::activation::softmax;
use burn::tensor::Tensor;

use super::config::HYV3Config;
use super::norm::HYV3RMSNorm;

#[derive(Module, Debug)]
pub struct HYV3Attention {
    pub q_proj: nn::Linear,
    pub k_proj: nn::Linear,
    pub v_proj: nn::Linear,
    pub o_proj: nn::Linear,
    pub q_norm: HYV3RMSNorm,
    pub k_norm: HYV3RMSNorm,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    scale: f64,
}

impl HYV3Attention {
    pub fn new(config: &HYV3Config, device: &burn::tensor::Device) -> Self {
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
            q_norm: HYV3RMSNorm::new(config.head_dim, device),
            k_norm: HYV3RMSNorm::new(config.head_dim, device),
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
        x: Tensor<2>,
        batch: usize,
        seq_len: usize,
        rope_cos: Tensor<4>,
        rope_sin: Tensor<4>,
    ) -> Tensor<2> {
        let q = self.q_proj.forward(x.clone());
        let k = self.k_proj.forward(x.clone());
        let v = self.v_proj.forward(x);

        let q = q.reshape([batch, seq_len, self.num_heads, self.head_dim]);
        let k = k.reshape([batch, seq_len, self.num_kv_heads, self.head_dim]);
        let v = v.reshape([batch, seq_len, self.num_kv_heads, self.head_dim]);

        let q = self.q_norm.forward_4d(q);
        let k = self.k_norm.forward_4d(k);

        let q = q.swap_dims(1, 2);
        let k = k.swap_dims(1, 2);
        let v = v.swap_dims(1, 2);

        let q = apply_rotary_pos_emb(q, rope_cos.clone(), rope_sin.clone());
        let k = apply_rotary_pos_emb(k, rope_cos, rope_sin);

        let kv_ratio = self.num_heads / self.num_kv_heads;
        let k = repeat_kv(k, kv_ratio);
        let v = repeat_kv(v, kv_ratio);

        let attn_weights = q.matmul(k.swap_dims(2, 3)).mul_scalar(self.scale);
        let mask = causal_mask(seq_len, attn_weights.device());
        let attn_weights = softmax(attn_weights + mask, 3);

        let attn_out = attn_weights.matmul(v);
        let attn_out = attn_out.swap_dims(1, 2);
        let attn_out = attn_out.reshape([batch * seq_len, self.num_heads * self.head_dim]);

        self.o_proj.forward(attn_out)
    }
}

fn apply_rotary_pos_emb(
    x: Tensor<4>,
    cos: Tensor<4>,
    sin: Tensor<4>,
) -> Tensor<4> {
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

fn repeat_kv(x: Tensor<4>, n_rep: usize) -> Tensor<4> {
    if n_rep == 1 {
        return x;
    }
    let d = x.dims();
    let x = x.unsqueeze_dim::<5>(2);
    let x = x.expand([d[0], d[1], n_rep, d[2], d[3]]);
    x.reshape([d[0], d[1] * n_rep, d[2], d[3]])
}

fn causal_mask(seq_len: usize, device: burn::tensor::Device) -> Tensor<4> {
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
    Tensor::<1>::from_floats(data.as_slice(), &device).reshape([1, 1, seq_len, seq_len])
}
