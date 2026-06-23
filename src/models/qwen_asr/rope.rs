use burn::tensor::{Int, Tensor};

use super::config::TextConfig;

pub struct Qwen3ASRMRoPE {
    inv_freq: Vec<f32>,
    head_dim: usize,
}

impl Qwen3ASRMRoPE {
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

    pub fn compute_cos_sin(
        &self,
        position_ids: Tensor<2, Int>,
    ) -> (Tensor<4>, Tensor<4>) {
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

        let cos = Tensor::<1>::from_floats(cos_vals.as_slice(), &device)
            .reshape([batch, seq_len, self.head_dim])
            .unsqueeze_dim::<4>(1);
        let sin = Tensor::<1>::from_floats(sin_vals.as_slice(), &device)
            .reshape([batch, seq_len, self.head_dim])
            .unsqueeze_dim::<4>(1);
        (cos, sin)
    }

    pub fn compute_cos_sin_from_positions(
        &self,
        positions: &[usize],
        device: &burn::tensor::Device,
    ) -> (Tensor<4>, Tensor<4>) {
        let ids: Vec<i32> = positions.iter().map(|&pos| pos as i32).collect();
        let position_ids =
            Tensor::<1, Int>::from_ints(ids.as_slice(), device).unsqueeze_dim::<2>(0);
        self.compute_cos_sin(position_ids)
    }
}

pub fn apply_mrope_simple(
    x: Tensor<4>,
    cos: &Tensor<4>,
    sin: &Tensor<4>,
) -> Tensor<4> {
    let head_dim = x.dims()[3];
    let half = head_dim / 2;
    let x_clone = x.clone();
    let x1 = x_clone.clone().narrow(3, 0, half);
    let x2 = x_clone.narrow(3, half, half);
    let rotate_half = Tensor::cat(vec![x2.neg(), x1], 3);
    x * cos.clone() + rotate_half * sin.clone()
}

pub fn create_mrope(text_config: &TextConfig) -> Qwen3ASRMRoPE {
    Qwen3ASRMRoPE::new(
        text_config.head_dim,
        text_config.rope_theta,
        &text_config.mrope_section(),
        text_config.mrope_interleaved(),
    )
}
