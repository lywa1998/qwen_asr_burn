# Backends

Burn separates the model code from the backend. The same `Tensor<D>` runs on any backend; you switch by passing a different `Device`. Backend choice is now made **at the device level**, not via type generics (this is the major API change from pre-0.21 Burn).

## CPU backend story (changed in 0.21)

Three CPU backends coexist, with different roles:

| Backend | Crate | When to pick it |
| ------- | ----- | --------------- |
| **Flex** | `burn-flex` | Pure-Rust eager CPU. **New in 0.21, replaces `burn-ndarray`** for embedded, WebAssembly, and small models. Eager only — no fusion, no autotune. Smallest binary, fewest dependencies. |
| **CubeCL CPU** | `burn-cpu` (via `cpu` feature) | Production CPU workloads. Goes through the CubeCL JIT, gets fusion + autotune. Use this for serving on CPU servers. |
| **ndarray** | `burn-ndarray` | **Deprecated.** Still works in 0.21 as a transition shim, will be removed in 1–2 more releases. Was the BLAS-backed CPU path; either move to Flex (small / no_std) or CubeCL CPU (perf). |

When picking a CPU backend now: pick **Flex** for embedded/WASM/small, **CubeCL CPU** for performance, **never ndarray for new projects**.

## Picking a backend

```rust
use burn::prelude::*;

let device = Device::wgpu(DeviceKind::DefaultDevice);   // GPU, cross-platform
let device = Device::cuda(0);                          // NVIDIA, fastest on NVIDIA
let device = Device::vulkan(DeviceKind::DefaultDevice);
let device = Device::metal(DeviceKind::DefaultDevice);
let device = Device::webgpu(DeviceKind::DefaultDevice);
let device = Device::rocm(0);                          // AMD
let device = Device::libtorch();
let device = Device::libtorch_cuda(0);
let device = Device::libtorch_mps();
let device = Device::libtorch_vulkan();
let device = Device::flex();                           // pure-Rust CPU (default)
let device = Device::ndarray();                        // CPU via ndarray + BLAS
let device = Device::cpu();                            // best available CPU backend
let device = Device::remote("tcp://host:port", 0);     // remote execution
```

Each variant requires a matching cargo feature in `burn`.

## Cargo feature matrix

| Feature | Brings in | Use when |
| ------- | --------- | -------- |
| `wgpu` | CubeCL on top of wgpu — works on Vulkan, Metal, DX12, WebGPU, with one binary | Cross-platform GPU, no driver-specific setup |
| `vulkan` | wgpu pinned to Vulkan | Linux/Windows, need Vulkan specifically |
| `metal` | wgpu pinned to Metal | Apple Silicon, no Vulkan SDK |
| `webgpu` | wgpu pinned to WebGPU | Browser / WASM |
| `cuda` | CubeCL CUDA | NVIDIA, fastest |
| `rocm` | CubeCL ROCm | AMD GPUs |
| `cpu` | CubeCL CPU | CPU, autotuned kernels |
| `tch` | LibTorch | Reuse PyTorch's kernels, need CUDA+cuDNN already installed |
| `candle` | Candle | Lightweight inference |
| `candle-cuda`, `candle-metal` | Candle on GPU | Lightweight inference on GPU |
| `flex` | Pure-Rust CPU (`burn-flex`) — **new in 0.21** | `no_std`, WASM, embedded, small models |
| `ndarray` | ndarray-backed CPU — **deprecated in 0.21** | Don't use for new projects. Still compiles, will be removed in 1–2 releases |
| `fusion` | Kernel fusion decorator | Almost always — large speedup on CubeCL backends |
| `autodiff` | Autodiff support | Always for training |
| `train` | Implies `autodiff`, `optim`, `dataset` | Anything that trains |

`burn`'s default features pull in `rl`, `std`, `optim`, plus a few others, so a normal training crate just adds the backend feature:

```toml
[dependencies]
burn = { version = "0.21", features = ["wgpu", "train", "vision"] }
```

For a leaner setup (e.g. inference-only):

```toml
burn = { version = "0.21", default-features = false, features = ["wgpu", "std", "store"] }
```

## Device decorators

Composed on the device, not on the backend type:

```rust
let device = Device::wgpu(DeviceKind::DefaultDevice);
let autodiff_device = device.clone().autodiff();           // for training
let with_checkpointing = device.clone().gradient_checkpointing();
let inner = autodiff_device.inner();                       // strip autodiff
```

You typically:
- Use the autodiff device when calling `model.init(&device)` for training.
- Use the plain device for inference and validation paths.
- Call `.inner()` if you have an autodiff device and want the underlying one.

## Cross-platform gotchas

**WGPU recursion limit.** Any binary that touches wgpu/vulkan/metal/webgpu needs:

```rust
#![recursion_limit = "256"]
```

at the top of `main.rs` or `lib.rs`. The default 128 isn't enough for the nested associated-type chains in CubeCL. Symptom: `error[E0275]: overflow evaluating the requirement`.

**macOS + Vulkan.** wgpu uses Metal by default on macOS. To use Vulkan instead, install the Vulkan SDK and enable the `vulkan` feature. Otherwise just use `metal`.

**LibTorch.** Requires PyTorch's C++ libraries installed and `LIBTORCH` / `LIBTORCH_INCLUDE` / `LIBTORCH_LIB` environment variables pointing to them. See the `tch-rs` README. Worth it if you need cuDNN parity with PyTorch.

**Autotune cold start.** CubeCL backends benchmark kernel variants the first time they encounter a new tensor shape, then cache results. For production deployments, configure the cache path in `burn.toml` and bundle the file:

```toml
# burn.toml
[cubecl.autotune]
level = "balanced"
cache = { file = "autotune.json" }
```

Then bundle `autotune.json` with the deployed binary. See `references/burn-toml.md` for the full configuration surface (level, streaming, persistent memory, logging). The pre-0.21 `CUBECL_CONFIG` env var path still works as a fallback but `burn.toml` is the recommended interface.

**Kernel validation layer (new in 0.21).** Opt-in via `burn.toml`:

```toml
[cubecl.compilation]
check_mode = "validate"  # or "enforce" to fail on validation errors
```

The validation layer was added in 0.21 and already caught real kernels generating out-of-bounds memory accesses during framework development. Worth enabling when writing custom CubeCL kernels.

**Shape multiples of 8/16/32 are fastest.** `[1024, 1024]` autotunes better than `[1000, 1000]`. Burn won't refuse non-aligned shapes, but you'll lose vectorization and trigger bounds checks.

## Mixing backends

Different parts of a program can use different backends. A common pattern: train on GPU, run augmentation on CPU.

```rust
let train_device = Device::wgpu(DeviceKind::DefaultDevice);
let augment_device = Device::cpu();

// In your batcher:
fn batch(&self, items: Vec<Item>, device: &Device) -> Batch {
    // assemble on the augment device
    let batch_aug = build_batch(items, augment_device);
    // move to training device in one go (one sync)
    Tensor::from_data(batch_aug.into_data(), &train_device)
}
```

Avoid bouncing per-tensor between devices in a hot loop — each move can be a sync.

## Default device

`Default::default()` returns `Device::flex()` (the pure-Rust CPU backend) unless you've explicitly compiled without `flex`. Use the explicit constructor (`Device::wgpu(...)`) for real workloads — relying on the default is fine for demos but tends to make examples ambiguous about what's actually running.

## Inspecting available devices

```rust
use burn::tensor::{DeviceFilter, DeviceType};

let devices = Device::enumerate(DeviceFilter::new()
    .with(DeviceType::Cuda)
    .with(DeviceType::Wgpu));

for d in devices {
    println!("{d:?}");
}
```

## Device configuration

Element types and other per-device settings:

```rust
use burn::tensor::{DeviceConfig, FloatDType, IntDType};

let mut device = Device::wgpu(DeviceKind::DefaultDevice);
device.configure(DeviceConfig::default()
    .float_dtype(FloatDType::F16)
    .int_dtype(IntDType::I32))?;
```

This is how you pick `f16` vs `f32` for the entire device's tensor element type.

## Backends that compose with autodiff and fusion

```
                  +---- Autodiff ----+----  fusion ----+
CubeCL (cuda)     |       yes        |       yes        |
CubeCL (rocm)     |       yes        |       yes        |
CubeCL (wgpu)     |       yes        |       yes        |
CubeCL (metal)    |       yes        |       yes        |
CubeCL (vulkan)   |       yes        |       yes        |
CubeCL (cpu)      |       yes        |       yes        |
Flex (pure CPU)   |       yes        |       no         |
ndarray           |       yes        |       no         |
LibTorch          |       yes        |       no         |
Candle            |       yes        |       no         |
```

So if you want autotune + kernel fusion (and you usually do), pick a CubeCL backend: `wgpu`, `cuda`, `rocm`, `metal`, `vulkan`, or `cpu`.

## Distributed and remote

- **Router** (`router` feature): split a model across multiple devices in the same process.
- **Remote** (`remote` feature): tensor ops execute on another machine over the network. Useful for inference servers.

For distributed training across machines, see `burn-book/src/performance/distributed-computing.md` and `crates/burn-communication/`.
