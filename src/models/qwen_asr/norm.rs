use burn::config::Config;
use burn::module::{Module, Param};
use burn::tensor::Tensor;

#[derive(Module, Debug)]
pub struct Qwen3ASRRmsNorm {
    pub weight: Param<Tensor<1>>,
    #[module(skip)]
    epsilon: f64,
}

#[derive(Config, Debug)]
pub struct Qwen3ASRRmsNormConfig {
    d_model: usize,
    epsilon: f64,
}

impl Qwen3ASRRmsNormConfig {
    pub fn init(&self, device: &burn::tensor::Device) -> Qwen3ASRRmsNorm {
        Qwen3ASRRmsNorm {
            weight: Param::from_tensor(Tensor::ones([self.d_model], device)),
            epsilon: self.epsilon,
        }
    }
}

impl Qwen3ASRRmsNorm {
    pub fn forward(&self, x: Tensor<3>) -> Tensor<3> {
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
pub struct Qwen3ASRLayerNorm {
    pub weight: Param<Tensor<1>>,
    pub bias: Param<Tensor<1>>,
    #[module(skip)]
    epsilon: f64,
}

#[derive(Config, Debug)]
pub struct Qwen3ASRLayerNormConfig {
    d_model: usize,
    epsilon: f64,
}

impl Qwen3ASRLayerNormConfig {
    pub fn init(&self, device: &burn::tensor::Device) -> Qwen3ASRLayerNorm {
        Qwen3ASRLayerNorm {
            weight: Param::from_tensor(Tensor::ones([self.d_model], device)),
            bias: Param::from_tensor(Tensor::zeros([self.d_model], device)),
            epsilon: self.epsilon,
        }
    }
}

impl Qwen3ASRLayerNorm {
    pub fn forward(&self, x: Tensor<3>) -> Tensor<3> {
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

#[derive(Module, Debug)]
pub struct Qwen3ASRQKNorm {
    pub weight: Param<Tensor<1>>,
    #[module(skip)]
    epsilon: f64,
}

#[derive(Config, Debug)]
pub struct Qwen3ASRQKNormConfig {
    head_dim: usize,
    epsilon: f64,
}

impl Qwen3ASRQKNormConfig {
    pub fn init(&self, device: &burn::tensor::Device) -> Qwen3ASRQKNorm {
        Qwen3ASRQKNorm {
            weight: Param::from_tensor(Tensor::ones([self.head_dim], device)),
            epsilon: self.epsilon,
        }
    }
}

impl Qwen3ASRQKNorm {
    pub fn forward(&self, x: Tensor<4>) -> Tensor<4> {
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
