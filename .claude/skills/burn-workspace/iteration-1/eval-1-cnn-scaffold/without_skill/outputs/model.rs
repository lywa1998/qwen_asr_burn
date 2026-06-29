use burn::{
    nn::{
        Dropout, DropoutConfig, Linear, LinearConfig, PaddingConfig2d, Relu,
        conv::{Conv2d, Conv2dConfig},
        pool::{MaxPool2d, MaxPool2dConfig, AdaptiveAvgPool2d, AdaptiveAvgPool2dConfig},
    },
    prelude::*,
};

/// Simple CNN classifier for CIFAR-10 (3x32x32 RGB images, 10 classes).
///
/// Architecture:
///   Conv2d(3→32, 3x3, same) → ReLU → MaxPool(2)          # 32x16x16
///   Conv2d(32→64, 3x3, same) → ReLU → MaxPool(2)          # 64x8x8
///   Conv2d(64→128, 3x3, same) → ReLU → AdaptiveAvgPool(4) # 128x4x4
///   Flatten → Linear(2048→256) → ReLU → Dropout(0.5)
///   Linear(256→10)
#[derive(Module, Debug)]
pub struct Cifar10Cnn {
    conv1: Conv2d,
    conv2: Conv2d,
    conv3: Conv2d,
    pool: MaxPool2d,
    avg_pool: AdaptiveAvgPool2d,
    fc1: Linear,
    fc2: Linear,
    dropout: Dropout,
    activation: Relu,
}

impl Cifar10Cnn {
    pub fn new(device: &Device) -> Self {
        let conv1 = Conv2dConfig::new([3, 32], [3, 3])
            .with_padding(PaddingConfig2d::Same)
            .init(device);

        let conv2 = Conv2dConfig::new([32, 64], [3, 3])
            .with_padding(PaddingConfig2d::Same)
            .init(device);

        let conv3 = Conv2dConfig::new([64, 128], [3, 3])
            .with_padding(PaddingConfig2d::Same)
            .init(device);

        // 2x2 max-pooling with stride 2 halves spatial dims each time.
        let pool = MaxPool2dConfig::new([2, 2]).with_strides([2, 2]).init();

        // Adaptive avg-pool collapses each feature map to 4x4 regardless of input size.
        let avg_pool = AdaptiveAvgPool2dConfig::new([4, 4]).init();

        // After two 2x2 pools: 32 → 16 → 8 spatial; after AdaptiveAvgPool: 4x4.
        // So FC input is 128 channels × 4 × 4 = 2048.
        let fc1 = LinearConfig::new(2048, 256).init(device);
        let fc2 = LinearConfig::new(256, 10).init(device);

        let dropout = DropoutConfig::new(0.5).init();

        Self {
            conv1,
            conv2,
            conv3,
            pool,
            avg_pool,
            fc1,
            fc2,
            dropout,
            activation: Relu::new(),
        }
    }

    /// Forward pass.
    ///
    /// # Shapes
    ///   - input:  [batch_size, 3, 32, 32]
    ///   - output: [batch_size, 10]  (raw logits for each CIFAR-10 class)
    pub fn forward(&self, x: Tensor<4>) -> Tensor<2> {
        // Block 1: Conv → ReLU → Pool
        let x = self.conv1.forward(x);
        let x = self.activation.forward(x);
        let x = self.pool.forward(x); // [B, 32, 16, 16]

        // Block 2: Conv → ReLU → Pool
        let x = self.conv2.forward(x);
        let x = self.activation.forward(x);
        let x = self.pool.forward(x); // [B, 64, 8, 8]

        // Block 3: Conv → ReLU → AdaptiveAvgPool
        let x = self.conv3.forward(x);
        let x = self.activation.forward(x);
        let x = self.avg_pool.forward(x); // [B, 128, 4, 4]

        // Flatten spatial dims (1..3) into a single feature vector.
        let x = x.flatten(1, 3); // [B, 2048]

        // Fully-connected head.
        let x = self.fc1.forward(x);
        let x = self.activation.forward(x);
        let x = self.dropout.forward(x);

        self.fc2.forward(x) // [B, 10]
    }
}
