## Approach

**Goal**: Scaffold a Burn 0.21 CNN for CIFAR-10 using the current (non-generic) API style.

### Key Burn 0.21 API decisions

1. **No `B: Backend` on the struct.** In Burn 0.21, model structs are plain (e.g., `struct Cifar10Cnn { conv1: Conv2d, ... }`) — the `#[derive(Module)]` macro handles everything. This replaces the old pattern of `struct Model<B: Backend> { ... }`.

2. **`Device` is a first-class type** imported from `burn::prelude::*`. No more `B::Device` associated type gymnastics.

3. **`Tensor<D>` and `Tensor<D, Kind>`** carry only dimension/kind information — the backend is tracked internally at runtime, not in the type signature.

4. **`Device::wgpu(DeviceKind::DefaultDevice)`** is the standard WGPU entry point. For training you would call `device.autodiff()` to wrap it for gradient tracking.

5. **Config→init pattern**: each layer is built via a `*Config::new(...).init(&device)` call, consistent across conv, linear, pooling, dropout, and batchnorm layers.

6. **`flatten(start, end)`** replaces manual reshape for collapsing spatial dims before the FC head.

### Architecture rationale

- **Three conv blocks** (3→32, 32→64, 64→128) with `PaddingConfig2d::Same` so spatial dims are preserved through convolutions.
- **MaxPool2d(2×2, stride 2)** after the first two blocks halves spatial resolution from 32→16→8.
- **AdaptiveAvgPool2d(4×4)** after the third block ensures a fixed 128×4×4 feature map regardless of input resolution, then flattens to 2048.
- **FC head**: 2048 → 256 (ReLU + Dropout 0.5) → 10 (raw logits for CIFAR-10 classes).
- No BatchNorm is used here to keep the scaffold minimal; it can be added for better convergence during actual training.

### Files produced

| File | Purpose |
|---|---|
| `Cargo.toml` | Burn 0.21 dependency with `wgpu`, `train`, and `vision` features |
| `model.rs` | `Cifar10Cnn` struct + `forward()` using the current Burn 0.21 API |
| `main.rs` | Creates a WGPU device, instantiates the model, runs a dummy forward pass |
