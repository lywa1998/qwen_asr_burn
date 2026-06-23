use burn::module::Module;
use burn::nn;
use burn::tensor::activation::silu;
use burn::tensor::Tensor;

#[derive(Module, Debug)]
pub struct HYV3MLP {
    pub gate_proj: nn::Linear,
    pub up_proj: nn::Linear,
    pub down_proj: nn::Linear,
}

impl HYV3MLP {
    pub fn new(hidden: usize, intermediate: usize, device: &burn::tensor::Device) -> Self {
        Self {
            gate_proj: nn::LinearConfig::new(hidden, intermediate)
                .with_bias(false)
                .init(device),
            up_proj: nn::LinearConfig::new(hidden, intermediate)
                .with_bias(false)
                .init(device),
            down_proj: nn::LinearConfig::new(intermediate, hidden)
                .with_bias(false)
                .init(device),
        }
    }

    pub fn forward(&self, x: Tensor<2>) -> Tensor<2> {
        let gate = silu(self.gate_proj.forward(x.clone()));
        let up = self.up_proj.forward(x);
        self.down_proj.forward(gate * up)
    }
}
