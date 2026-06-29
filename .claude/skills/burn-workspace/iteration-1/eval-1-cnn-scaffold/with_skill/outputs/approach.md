## Approach

### API style: Burn 0.21 (post-dispatch)

This code uses the **current Burn 0.21 API** where the `B: Backend` generic is gone
from user-facing code. Backend selection is now done at the `Device` level via
runtime dispatch (`DispatchDevice`). Tensors are `Tensor<D>` (no `B`), modules
derive `Module` without a `B` parameter, and forward methods take plain `Tensor<D>`.

Key differences from pre-0.21 code shown in older blog posts and the burn-book:

| Old (pre-0.21) | Current (0.21+) |
|---|---|
| `Tensor<B, 2>` | `Tensor<2>` |
| `struct Model<B: Backend>` | `struct Model` |
| `type B = Wgpu;` | `let device = Device::wgpu(...);` |
| `Autodiff<B>` decorator | `device.clone().autodiff()` |
| `B::seed(42)` | `device.seed(42)` |

### Model architecture

A three-layer convolutional stack followed by two linear layers, designed for
CIFAR-10 (32x32 RGB, 10 classes):

```
Input [B, 3, 32, 32]
  → Conv2d(3, 32, kernel 3, pad 1) → ReLU → MaxPool2d(2)  → [B, 32, 16, 16]
  → Conv2d(32, 64, kernel 3, pad 1) → ReLU → MaxPool2d(2) → [B, 64, 8, 8]
  → Conv2d(64, 64, kernel 3, pad 1) → ReLU → AdaptiveAvgPool2d(4) → [B, 64, 4, 4]
  → Flatten → [B, 1024]
  → Linear(1024, 128) → ReLU → Linear(128, 10) → [B, 10]
```

### Conventions followed

- **Config + init pattern**: `ModelConfig` (with `#[derive(Config)]`) holds
  hyperparameters; `init(&Device)` allocates the module. The number of classes
  is the only required field; hidden dim defaults to 128.
- **No `B: Backend` on modules**: `#[derive(Module, Debug)] struct Model` has
  no generic parameter.
- **No backend generic on `forward`**: `fn forward(&self, images: Tensor<4>) -> Tensor<2>`.
- **No `Ignored<T>`**: Not needed in this design.
- **`Relu::new()`** directly (no config for parameter-free modules).
- **`recursion_limit = "256"`**: Required at the crate root for the WGPU backend
  (CubeCL's associated types exceed the default 128).
- **`device.clone().autodiff()`**: Enables gradient tracking for training.
- **Ownership**: Tensors are consumed by each op; this is fine because each
  intermediate is used exactly once in this feed-forward stack.

### What's in each file

- **`Cargo.toml`**: Minimal dependency — `burn = { version = "0.21", features =
  ["wgpu", "train", "vision"] }`. The `wgpu` feature brings in the CubeCL WGPU
  backend; `train` pulls in autodiff+optim+dataset; `vision` activates built-in
  vision datasets (CifarDataset, MnistDataset).
- **`model.rs`**: The CNN module definition, its config, and the `forward` method.
- **`main.rs`**: Device setup (WGPU default GPU), model initialization on an
  autodiff device, a smoke-test forward pass with random data, and commented-out
  training loop scaffolding showing how to wire up `SupervisedTraining` +
  `Learner` with the built-in CIFAR-10 dataset.
