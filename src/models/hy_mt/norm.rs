use burn::module::Module;
use burn::module::Param;
use burn::tensor::Tensor;

const RMS_EPS: f64 = 1e-5_f64;

#[derive(Module, Debug)]
pub struct HYV3RMSNorm {
    pub weight: Param<Tensor<1>>,
}

impl HYV3RMSNorm {
    pub fn new(num_features: usize, device: &burn::tensor::Device) -> Self {
        let weight = Tensor::ones([num_features], device);
        Self {
            weight: Param::from_tensor(weight),
        }
    }

    /// x: [N, D] — normalize along last dim (D), where N = batch*seq
    pub fn forward(&self, x: Tensor<2>) -> Tensor<2> {
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
    pub fn forward_4d(&self, x: Tensor<4>) -> Tensor<4> {
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
