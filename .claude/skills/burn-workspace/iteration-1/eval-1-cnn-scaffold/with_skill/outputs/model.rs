use burn::{
    nn::{
        conv::{Conv2d, Conv2dConfig},
        pool::{AdaptiveAvgPool2d, AdaptiveAvgPool2dConfig, MaxPool2d, MaxPool2dConfig},
        Linear, LinearConfig, Relu,
    },
    prelude::*,
};

/// A simple CNN classifier for CIFAR-10 (3x32x32 RGB, 10 classes).
///
/// Architecture:
///   Conv2d(3, 32, 3) → ReLU → MaxPool2d(2) →
///   Conv2d(32, 64, 3) → ReLU → MaxPool2d(2) →
///   Conv2d(64, 64, 3) → ReLU → AdaptiveAvgPool2d(4) →
///   Flatten → Linear(64*4*4, 128) → ReLU → Linear(128, 10)
#[derive(Module, Debug)]
pub struct Model {
    conv1: Conv2d,
    conv2: Conv2d,
    conv3: Conv2d,
    pool: MaxPool2d,
    avg_pool: AdaptiveAvgPool2d,
    linear1: Linear,
    linear2: Linear,
    activation: Relu,
}

#[derive(Config, Debug)]
pub struct ModelConfig {
    /// Number of output classes (10 for CIFAR-10).
    num_classes: usize,
    /// Hidden dimension for the first linear layer.
    #[config(default = 128)]
    hidden: usize,
}

impl ModelConfig {
    /// Build the model, allocating all parameters on `device`.
    pub fn init(&self, device: &Device) -> Model {
        let conv1 = Conv2dConfig::new([3, 32], [3, 3])
            .with_padding(1)
            .init(device);
        let conv2 = Conv2dConfig::new([32, 64], [3, 3])
            .with_padding(1)
            .init(device);
        let conv3 = Conv2dConfig::new([64, 64], [3, 3])
            .with_padding(1)
            .init(device);

        let pool = MaxPool2dConfig::new([2, 2]).with_strides([2, 2]).init();
        let avg_pool = AdaptiveAvgPool2dConfig::new([4, 4]).init();

        // After three 2×2 pools on a 32×32 input: 32→16→8→4 (adaptive avg pool).
        // So the flattened feature size is 64 channels * 4 * 4 = 1024.
        let linear1 = LinearConfig::new(64 * 4 * 4, self.hidden).init(device);
        let linear2 = LinearConfig::new(self.hidden, self.num_classes).init(device);

        Model {
            conv1,
            conv2,
            conv3,
            pool,
            avg_pool,
            linear1,
            linear2,
            activation: Relu::new(),
        }
    }
}

impl Model {
    /// Forward pass.
    ///
    /// Takes images of shape `[batch_size, 3, 32, 32]` (BCHW) and returns
    /// logits of shape `[batch_size, num_classes]`.
    pub fn forward(&self, images: Tensor<4>) -> Tensor<2> {
        // conv1 + relu + pool
        let x = self.conv1.forward(images);
        let x = self.activation.forward(x);
        let x = self.pool.forward(x);

        // conv2 + relu + pool
        let x = self.conv2.forward(x);
        let x = self.activation.forward(x);
        let x = self.pool.forward(x);

        // conv3 + relu + adaptive avg pool
        let x = self.conv3.forward(x);
        let x = self.activation.forward(x);
        let x = self.avg_pool.forward(x);

        // flatten [B, 64, 4, 4] → [B, 1024]
        let x = x.flatten(1, 3);

        // linear1 + relu + linear2
        let x = self.linear1.forward(x);
        let x = self.activation.forward(x);
        self.linear2.forward(x)
    }
}
