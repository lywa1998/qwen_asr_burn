use burn::tensor::{Tensor, Device};

use super::config::HYV3Config;

pub struct HYV3RotaryEmbedding {
    inv_freq: Vec<f32>,
    alpha: f64,
    head_dim: usize,
    max_seq_len: usize,
    theta: f64,
}

impl HYV3RotaryEmbedding {
    pub fn new(config: &HYV3Config) -> Self {
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
    pub fn compute(
        &self,
        seq_len: usize,
        device: &Device,
    ) -> (Tensor<4>, Tensor<4>) {
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
            Tensor::<1>::from_floats(freqs.as_slice(), device).reshape([seq_len, half]);
        let cos = freqs_t.clone().cos().unsqueeze_dim::<3>(0).unsqueeze_dim::<4>(1);
        let sin = freqs_t.sin().unsqueeze_dim::<3>(0).unsqueeze_dim::<4>(1);

        let cos = Tensor::cat(vec![cos.clone(), cos], 3);
        let sin = Tensor::cat(vec![sin.clone(), sin], 3);
        (cos, sin)
    }
}
