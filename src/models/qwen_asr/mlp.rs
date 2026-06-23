use burn::config::Config;
use burn::module::Module;
use burn::nn::LinearConfig;
use burn::nn::Linear;
use burn::tensor::Tensor;

#[derive(Module, Debug)]
pub struct Qwen3ASRMLP {
    pub gate_proj: Linear,
    pub up_proj: Linear,
    pub down_proj: Linear,
}

#[derive(Config, Debug)]
pub struct Qwen3ASRMLPConfig {
    hidden_size: usize,
    intermediate_size: usize,
}

impl Qwen3ASRMLPConfig {
    pub fn init(&self, device: &burn::tensor::Device) -> Qwen3ASRMLP {
        Qwen3ASRMLP {
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

impl Qwen3ASRMLP {
    pub fn forward<const D: usize>(&self, x: Tensor<D>) -> Tensor<D> {
        let gate = self.gate_proj.forward(x.clone());
        let gate = gate.clone().mul(burn::tensor::activation::sigmoid(gate));
        let up = self.up_proj.forward(x);
        self.down_proj.forward(gate.mul(up))
    }
}
