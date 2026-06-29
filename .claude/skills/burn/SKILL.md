---
name: burn
description: Expert guidance for building applications with the Burn deep learning framework in Rust (version 0.21+, including the in-progress 0.22 from git main). Covers tensors, modules, autodiff, training loops, the Learner, datasets, model saving/loading, PyTorch/SafeTensors import, and choosing backends (WGPU, CUDA, ROCm, Metal, Flex CPU, LibTorch). Use this whenever the user is writing Rust code that imports `burn`, asks about Burn APIs (`Tensor`, `Module`, `Backend`, `Device`, `DispatchDevice`, `Learner`, `Optimizer`, `Batcher`, `Dataset`, `ModuleRecord`, burnpack, `.bpk`, `burn.toml`), is porting a PyTorch model to Rust, wants to train or run inference with Burn, or asks anything that sounds like deep-learning-in-Rust. Burn 0.21 introduced `burn-dispatch` (runtime backend dispatch), `burn-flex` (new pure-Rust CPU backend replacing burn-ndarray), `burn.toml` project config, and a wave of breaking changes (Shape API, PaddingConfig, Gelu, Ignored→#[module(skip)], DType::Bool, powf for Int) — apply this skill so generated code matches the current API rather than older patterns from blog posts and the burn-book. Especially apply when the user is migrating an existing `<B: Backend>` codebase, hitting `cannot find type 'B' in this scope` / `unresolved import burn::tensor::backend` errors, or mixing crates.io `burn` with the git version.
---

# Burn framework

Burn is a Rust deep-learning framework. Public release is 0.21 on crates.io; git `main` is on 0.22.0-pre.1. This skill covers the current API and the conventions that make Burn code idiomatic, fast, and portable across backends.

## crates.io vs git — pick one, not both

This is the **first** thing to clarify with the user. The two sources have different APIs:

| Source | Version | API shape |
| ------ | ------- | --------- |
| crates.io | `burn = "0.21"` | Pre-dispatch — modules still take `<B: Backend>`; `Tensor<B, D>` everywhere. `burn-dispatch` exists but the user-facing surface hasn't been switched over yet. |
| git main | `0.22.0-pre.1` | Post-dispatch — `<B: Backend>` is fully removed. `Tensor<D>`. Bare `Device`. Factory methods like `Device::cuda(0)`, `Device::metal(DeviceKind::DefaultDevice)`. |

The burn-book, blog posts, and most code on GitHub show some mix of pre-0.21, crates.io 0.21, and git-main APIs. Before writing or reviewing code, **check which source the user's `Cargo.toml` resolves to** and write code that matches.

If they're on crates.io 0.21 and want the new API: tell them it requires the git version (see migration section below) and confirm before switching them — the git version pulls a much larger build (cubecl, wgpu, etc. all from git too).

### Switching a project from crates.io 0.21 to git

A `git = "..."` line on `burn` alone is not enough — transitive `burn-*` crates will still resolve to crates.io 0.21 and you'll get trait-id mismatches at compile time. Always pair the direct deps with a `[patch.crates-io]` block that redirects **every** `burn-*` crate to git:

```toml
[dependencies]
burn = { git = "https://github.com/tracel-ai/burn.git", features = ["fusion", "std"], default-features = false }
burn-store = { git = "https://github.com/tracel-ai/burn.git" }

[patch.crates-io]
burn               = { git = "https://github.com/tracel-ai/burn.git" }
burn-autodiff      = { git = "https://github.com/tracel-ai/burn.git" }
burn-backend       = { git = "https://github.com/tracel-ai/burn.git" }
burn-core          = { git = "https://github.com/tracel-ai/burn.git" }
burn-cubecl        = { git = "https://github.com/tracel-ai/burn.git" }
burn-cubecl-fusion = { git = "https://github.com/tracel-ai/burn.git" }
burn-cuda          = { git = "https://github.com/tracel-ai/burn.git" }
burn-derive        = { git = "https://github.com/tracel-ai/burn.git" }
burn-dispatch      = { git = "https://github.com/tracel-ai/burn.git" }
burn-flex          = { git = "https://github.com/tracel-ai/burn.git" }
burn-fusion        = { git = "https://github.com/tracel-ai/burn.git" }
burn-ir            = { git = "https://github.com/tracel-ai/burn.git" }
burn-ndarray       = { git = "https://github.com/tracel-ai/burn.git" }
burn-nn            = { git = "https://github.com/tracel-ai/burn.git" }
burn-optim         = { git = "https://github.com/tracel-ai/burn.git" }
burn-router        = { git = "https://github.com/tracel-ai/burn.git" }
burn-std           = { git = "https://github.com/tracel-ai/burn.git" }
burn-store         = { git = "https://github.com/tracel-ai/burn.git" }
burn-tensor        = { git = "https://github.com/tracel-ai/burn.git" }
burn-wgpu          = { git = "https://github.com/tracel-ai/burn.git" }
# Plus burn-tch / burn-rocm / burn-vision / burn-candle if you depend on those backends.
```

After `cargo check`, if Cargo warns `patch 'burn-X' was not used in the crate graph`, you can drop that line — it just means nothing transitively pulls `burn-X`. Better to start over-broad and trim than to under-cover and get silent version splits.

**Don't depend on `burn-dispatch` directly.** It's reachable as a transitive of `burn`, and its public surface (`DispatchDevice`, `Dispatch`) is internal plumbing. Use `Device::cuda(0)` / `Device::metal(DeviceKind::DefaultDevice)` instead.

## The most important API change to remember

Burn 0.21 introduced the `burn-dispatch` crate, which replaces the old "compile-time `Backend` generic" model with **runtime backend dispatch** at the `Device` layer. Internally `Device` wraps a `DispatchDevice` enum, but you use the `Device::*` constructors, not raw enum variants.

The pre-dispatch API was generic over a `Backend` trait — you'll see this in older blog posts, the burn-book, the crates.io 0.21 release, and any code from before that:

```rust
// OLD — pre-dispatch (also crates.io 0.21)
type B = Wgpu;
let t: Tensor<B, 2> = Tensor::from_data([[1., 2.], [3., 4.]], &device);

#[derive(Module, Debug)]
struct Model<B: Backend> {
    linear: Linear<B>,
}

impl<B: Backend> Model<B> {
    fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> { ... }
}
```

Current API (git main / 0.22-pre) — the `B` generic is gone. Backend selection lives on the `Device`:

```rust
// CURRENT — git main
use burn::prelude::*;

// Each of these picks the backend at runtime (internally via DispatchDevice):
let device = Device::cuda(0);
let device = Device::wgpu(DeviceKind::DefaultDevice);
let device = Device::metal(DeviceKind::DefaultDevice);

let t: Tensor<2> = Tensor::from_data([[1., 2.], [3., 4.]], &device);

#[derive(Module, Debug)]
struct Model {
    linear: nn::Linear,
}

impl Model {
    fn forward(&self, x: Tensor<2>) -> Tensor<2> { ... }
}
```

**No perf regression.** Tensor ops do a static enum dispatch on the user's thread; this doesn't penalize fusion or the CubeCL runtime.

**Compile times are the big win.** Early Tracel experiment: incremental release rebuild dropped from 11s to under 1s.

What's still backend-aware: `Device` picker, `device.clone().autodiff()` for gradient tracking, `device.inner()` to strip autodiff for inference.

## Migrating an existing `<B: Backend>` codebase

This is mostly mechanical. The recipe below has worked end-to-end on a multi-thousand-line audio-ML project. Apply it as a sequence of regex passes, then fix the small number of residual errors by hand.

**Step 1 — update `Cargo.toml`** to git + `[patch.crates-io]` as shown above.

**Step 2 — automated regex sweep** across `src/`. The substitutions are deterministic; running them as a script avoids slip-ups:

```python
import re, pathlib
ROOT = pathlib.Path("src")
for p in ROOT.rglob("*.rs"):
    s = p.read_text()
    s = re.sub(r"Tensor<B,\s*", "Tensor<", s)              # Tensor<B, D[, Kind]>
    s = re.sub(r"Tensor::<B,\s*", "Tensor::<", s)           # turbofish form
    s = re.sub(r"\bB::Device\b", "Device", s)               # B::Device → Device
    s = re.sub(r"<B:\s*Backend(\s*\+\s*'static)?>", "", s) # strip bound
    s = re.sub(r",\s*B:\s*Backend(\s*\+\s*'static)?", "", s)
    s = re.sub(r"<B:\s*Backend(\s*\+\s*'static)?,\s*", "<", s)
    s = re.sub(r"::<B>", "", s)                             # ::<B> turbofish
    s = re.sub(r"::<B,\s*", "::<", s)
    s = re.sub(r"(\b[A-Z]\w*)<B>", r"\1", s)                # Foo<B> → Foo
    s = re.sub(r"(\b[A-Z]\w*)<B,\s*", r"\1<", s)
    s = re.sub(r"^use burn::tensor::backend::Backend;\s*\n", "", s, flags=re.M)
    p.write_text(s)
```

**Step 3 — fix the residuals** that the regex can't catch. From a real run, these were:

- **Missing `Device` import.** Anywhere you used `B::Device`, you now need `use burn::tensor::Device;` at the top of the file. The grep `grep -rn "\bDevice\b" src` after the sweep shows the affected files.
- **`Tensor::<D, Bool>::from_bool(slice.into(), &device)` → E0283 "type annotations needed".** The `slice.into()` is ambiguous because `from_bool` takes `impl Into<TensorData>` and bools have multiple `Into` paths. Bind the data explicitly:

  ```rust
  let data: burn::tensor::TensorData = values.as_slice().into();
  let mask = Tensor::<1, Bool>::from_bool(data, &device);
  ```

- **Half-removed generics on a few call sites.** `Foo::<Backend>::new(...)` patterns where `Foo` no longer takes a generic. Drop the turbofish (`Foo::new(...)`).
- **Construct devices via the factory methods**, not `DispatchDevice::Cuda(...).into()`:

  ```rust
  #[cfg(feature = "cuda")]
  let device: Device = Device::cuda(0);
  #[cfg(feature = "metal")]
  let device: Device = Device::metal(DeviceKind::DefaultDevice);
  ```

**Step 4 — back up `src/` first** (`cp -r src src.bak` or use a git branch). The regex sweep is destructive and you'll want to diff if something behaves wrongly post-migration.

**Anti-pattern: half migration.** Don't keep `<B: Backend>` generics in library code while binding `type Backend = Dispatch;` at the binary boundary. It compiles, but you're now writing pre-dispatch code that drags `Dispatch` through every signature for no reason. Migrate fully or stay on the old API.

## What changed in Burn 0.21 (released May 2026)

Headline items:

- **`burn-dispatch`** — runtime backend dispatch. Foundation for dropping the `Backend` user-facing generic (completed on git main). Discussed above.
- **`burn-flex`** — new pure-Rust eager CPU backend, replaces deprecated `burn-ndarray`. Target: WASM, embedded, small models. No fusion, no autotune. See `references/backends.md`.
- **`burn.toml`** — project-level runtime config (autotune, fusion beam-search, kernel validation, streaming, logging). See `references/burn-toml.md`.
- **Distributed training rebuilt** — 4×CUDA: `to_device` 16–21× faster, `all_reduce` ~6× faster.
- **CubeCL overhead drop** — small-tensor eager ~3.4×, fusion ~5.4×, best 8.2×, no regressions.
- **GEMV and top-k optimized** — GEMV column-major now competitive with LibTorch; top-k up to 41× on inner axes.
- **FFT/IFFT ops** in `burn::tensor::signal` — first step toward complex tensors.
- **Off-policy RL loop** in `burn-train` — `RLTraining`, `OffPolicyConfig`, `DqnLearningAgent`. See `references/training.md`.

### Breaking changes from 0.20 → 0.21

- `TensorData::shape` is `Shape`, not `Vec<usize>`
- `Ignored<T>` → `#[module(skip)]`
- `PaddingConfig::Explicit` takes all sides
- `Gelu` → `Gelu::new()`
- `Shape` fields private, `swap`→`swapped`, `permute`→`permuted`, `ShapeError`→`MetadataError`
- `DType::Bool` → `DType::Bool(_)` (carries storage discriminator)
- `powf` removed for `Int` tensors (cast via `.float()` first)
- `BinFileRecorder` records not forward-compatible — convert before upgrading
- Full diffs for all 10 items in `references/migration-0.21.md`

## Core building blocks

These are the pieces you'll reach for in essentially every Burn app:

| Concept | Type / macro | One-liner |
| ------- | ------------ | --------- |
| Tensor | `Tensor<D>`, `Tensor<D, Int>`, `Tensor<D, Bool>` | N-D array. `D` is the rank (number of dims), known at compile time. |
| Module | `#[derive(Module, Debug)]` | Parameter container. Like `nn.Module` in PyTorch but no opinion on `forward`. |
| Config | `#[derive(Config, Debug)]` | Serializable struct with `with_*` builders and `#[config(default = ...)]`. Used to define both model hyperparameters and training config. |
| Device | `Device::wgpu(...)`, `Device::cuda(0)`, `Device::flex()`, `Device::libtorch()`, etc. | Backend + hardware index. Get autodiff with `.autodiff()`. |
| Record | `ModuleRecord`, `OptimizerRecord`, `LrSchedulerRecord` | Serialized state. Saves as `.bpk` (burnpack). Backend-independent. |
| Learner | `Learner::new(model, optim, lr)` + `SupervisedTraining` | High-level training loop with checkpointing, metrics, TUI. |
| Dataset / Batcher | `Dataset<I>` trait, `Batcher<I, O>` trait | Dataset is random-access. Batcher converts `Vec<I>` → batched tensor. |
| DataLoader | `DataLoaderBuilder::new(batcher).build(dataset)` | Multi-worker iteration. |
| Loss | `nn::loss::CrossEntropyLossConfig`, `MseLoss`, etc. | Built with a config; `.init(&device).forward(...)`. |

The `burn::prelude::*` import covers most everyday symbols:

```rust
pub use burn::prelude::*;
// brings in: Config, Module, Device, DeviceIndex, DeviceKind, Bool, Float, Int,
//            Shape, SliceArg, Tensor, TensorData, ElementConversion, s
```

## A minimal end-to-end app

Use this as the spine when sketching new projects. It compiles against the current API.

```rust
// src/model.rs
use burn::{
    nn::{conv::{Conv2d, Conv2dConfig}, Linear, LinearConfig, Relu},
    prelude::*,
};

#[derive(Module, Debug)]
pub struct Model {
    conv: Conv2d,
    linear: Linear,
    activation: Relu,
}

#[derive(Config, Debug)]
pub struct ModelConfig {
    num_classes: usize,
    #[config(default = 64)]
    hidden: usize,
}

impl ModelConfig {
    pub fn init(&self, device: &Device) -> Model {
        Model {
            conv: Conv2dConfig::new([1, 16], [3, 3]).init(device),
            linear: LinearConfig::new(16 * 26 * 26, self.num_classes).init(device),
            activation: Relu::new(),
        }
    }
}

impl Model {
    pub fn forward(&self, images: Tensor<3>) -> Tensor<2> {
        let [batch, h, w] = images.dims();
        let x = images.reshape([batch, 1, h, w]);
        let x = self.conv.forward(x);
        let x = self.activation.forward(x);
        let x = x.flatten(1, 3);
        self.linear.forward(x)
    }
}
```

```rust
// src/main.rs
#![recursion_limit = "256"]   // required for the wgpu backend
mod model;

use burn::{optim::AdamConfig, prelude::*};
use model::ModelConfig;

fn main() {
    let device = Device::wgpu(DeviceKind::DefaultDevice);
    let autodiff_device = device.clone().autodiff();

    let model = ModelConfig::new(10).init(&autodiff_device);
    // ... build dataloaders, Learner, SupervisedTraining ...
}
```

```toml
# Cargo.toml
[dependencies]
burn = { version = "0.21", features = ["wgpu", "train", "vision"] }
```

The `#![recursion_limit = "256"]` is a hard requirement for the WGPU backend — nested associated types blow past the default 128. Always set it for binaries that touch wgpu/vulkan/metal/webgpu.

## How to think about training

Burn separates three responsibilities:

1. **`forward` on the model** — pure tensor math. Knows nothing about loss or optimizers.
2. **`forward_classification` / `forward_regression`** — wraps `forward`, computes the loss, and returns a `ClassificationOutput` (or your own struct) that metrics can read. By convention this lives as an `impl` method on the model.
3. **`TrainStep` / `InferenceStep` impls** — connect the model to the `Learner`. They call `forward_classification` and, for training, run `loss.backward()`. The `Input` type is the batch struct your batcher produces (e.g. `MnistBatch`):

```rust
impl TrainStep for Model {
    type Input = MnistBatch;
    type Output = ClassificationOutput;

    fn step(&self, batch: MnistBatch) -> TrainOutput<ClassificationOutput> {
        let item = self.forward_classification(batch);
        TrainOutput::new(self, item.loss.backward(), item)
    }
}

impl InferenceStep for Model {
    type Input = MnistBatch;
    type Output = ClassificationOutput;

    fn step(&self, batch: MnistBatch) -> ClassificationOutput {
        self.forward_classification(batch)
    }
}
```

Two consequences of Burn's design that trip people up:

- **Gradients are returned, not stored on tensors.** `let grads = loss.backward(); model = optim.step(lr, model, GradientsParams::from_grads(grads, &model));`. There's no `optimizer.zero_grad()` — gradients are consumed by `step`.
- **The model is consumed and returned by each `optim.step`.** Reassign with `model = optim.step(...)`. This is what makes mixed-precision and gradient checkpointing tractable.

For full custom training loops (multi-optimizer, GAN-style, RL), see `references/training.md`. For the standard supervised case, the `Learner` + `SupervisedTraining` combo from `burn::train` does it for you and renders a TUI dashboard.

## Backends — what to recommend

User says "I want to use Burn on..." → recommend:

| Hardware | Backend | Cargo feature |
| -------- | ------- | ------------- |
| Any GPU, no setup | WGPU (Vulkan/Metal/DX12/WebGPU) | `wgpu` |
| NVIDIA GPU, fastest | CUDA via CubeCL | `cuda` |
| NVIDIA via LibTorch | LibTorch | `tch` |
| AMD GPU | ROCm | `rocm` |
| Apple Silicon | Metal | `metal` (or `wgpu`) |
| CPU, pure Rust | Flex | `flex` |
| Browser (WASM) | WGPU + WebGPU | `webgpu` |
| `no_std` / embedded | Flex | `flex`, `default-features = false` |

Backend decorators (composed via the `Device`, not via wrapper types in the new API):
- **Autodiff** — `device.clone().autodiff()`. Required for training.
- **Fusion** — turn on the `fusion` cargo feature; kernel fusion happens automatically for CubeCL backends.
- **Router** — `router` feature, for splitting work across multiple devices.
- **Remote** — `remote` feature, for running tensor ops over the network.

Backend choice does not affect what code you write — `Tensor<D>` is backend-agnostic. The same training code runs on CPU and GPU. Pick by what the user has and whether they want raw speed (CUDA), portability (WGPU), or pure-Rust (Flex).

See `references/backends.md` for cross-platform gotchas (recursion limit, macOS Vulkan SDK, etc.).

## Common operations you'll write a lot

- **Get a default device:** `let device = Default::default();` — picks `Flex` (pure-Rust CPU) unless features dictate otherwise. For real work, name the backend explicitly: `Device::wgpu(DeviceKind::DefaultDevice)`.
- **Move a tensor:** `tensor.to_device(&device)` — beware, this can trigger a sync on some backends.
- **Get a scalar out:** `tensor.into_scalar()` — **synchronizes** the backend. Use sparingly inside training loops.
- **Cast precision:** `tensor.cast(DType::F16)`.
- **Get the device:** `tensor.device()` — useful in module methods that need to construct intermediate tensors on the right device.
- **No-grad inference path:** call `model.valid()` to get an inference-only version of an autodiff-tracked model. Or, if your model is already autodiff-free, just call `forward`.
- **Seed the RNG:** `device.seed(42)`. (Not a free function on the backend type anymore.)

## Ownership and cloning

Almost every Burn tensor operation **consumes the input**. There is no in-place mutation API by design — the fusion engine relies on knowing each tensor's exact reference count to decide what can be reused. So you write code like this:

```rust
// Min-max normalization. Each tensor needs to outlive a few ops, so clone where needed.
let input = Tensor::<1>::from_floats([1., 2., 3., 4.], &device);
let min = input.clone().min();
let max = input.clone().max();
let normalized = (input - min.clone()).div(max - min);
```

Cloning a tensor is cheap — it bumps a refcount on the underlying buffer, never copies bytes. The fusion engine actively prefers code where you clone only as needed, because that lets it identify the last use and fuse the operation in place.

When users hit Rust-borrow-checker pain on tensors, the fix is almost always `.clone()`, not refactoring around ownership.

## Saving and loading weights

Three flavors, in order of common-ness:

1. **`ModuleRecord` (burnpack `.bpk`)** — the native path. Backend-independent.
   ```rust
   model.into_record().save("model")?;
   let record = ModuleRecord::load("model")?;
   let model = ModelConfig::new().init(&device).load_record(record);
   ```
2. **`burn-store` stores** — adds zero-copy mmap, PyTorch `.pt` reading, SafeTensors read/write, key remapping, partial loading. Needed any time the user wants to import HuggingFace / PyTorch weights.
   ```rust
   use burn_store::{ModuleSnapshot, PytorchStore, PyTorchToBurnAdapter};
   let mut store = PytorchStore::from_file("model.pt")
       .with_from_adapter(PyTorchToBurnAdapter);
   model.load_from(&mut store)?;
   ```
3. **ONNX** — `burn-onnx` (external crate) generates Rust source from `.onnx`. Used for static models more than for fine-tuning.

For PyTorch interop and the full `burn-store` builder API, see `references/saving-loading.md`.

## Performance — the 80/20

Three habits matter most:

1. **Don't sync inside hot loops.** `.into_scalar()`, `.to_data()`, `.into_data()` all block. Batch reads with `Transaction::default().register(...).execute()` to do one sync for many tensors.
2. **Prefer shape multiples of powers of two.** Shapes like `[1024, 1024]` autotune to faster kernels than `[1000, 1000]`. The autotune cache is on disk, so the cold start is one-time per machine.
3. **Group view operations.** `reshape`, `transpose`, `permute`, `slice`, `swap_dims` together, then call compute ops. Splitting them up breaks fusion.

For full performance notes (kernel fusion, kernel selection, async execution), see `references/performance.md`.

## When the user wants something specific

- **Custom training loop** — `references/training.md` has the manual loop with `GradientsParams`, gradient accumulation, multi-optimizer, EMA, etc. The same file covers the new off-policy RL training loop.
- **Saving/loading, PyTorch import, SafeTensors** — `references/saving-loading.md`.
- **Tensor operations cheatsheet** — `references/tensors.md` has the PyTorch → Burn op table plus the type system (`Float`/`Int`/`Bool`, `Param<Tensor<D>>`, etc.). Includes FFT/IFFT and signal-processing ops introduced in 0.21.
- **Custom modules, visitors, mappers** — `references/modules.md`.
- **Datasets and dataloaders** — `references/datasets.md`.
- **Backend selection, features, gotchas** — `references/backends.md`. Note the `burn-flex` vs `burn-ndarray` story.
- **Performance** — `references/performance.md`.
- **`no_std`, WASM, embedded** — `references/deployment.md`.
- **`burn.toml` runtime config** — `references/burn-toml.md` (autotune level, fusion knobs, kernel validation, logging).
- **Migrating from 0.20 → 0.21** — `references/migration-0.21.md` (full diffs for every breaking change).

Each of those is a thin slice of the burn-book updated for the current API. Read whichever matches the question before writing code; the framework changes fast and assumptions from training data are likely outdated.

## Things to flag to the user

- If they paste code with `Tensor<B, D>` or `impl<B: Backend>` on modules: ask which source their `Cargo.toml` resolves to. Crates.io 0.21 keeps these generics; git main has fully removed them. Convert only if they're on git.
- If their build errors say `cannot find type 'B' in this scope` or `unresolved import 'burn::tensor::backend'`: they've moved to git main but their source still uses the old generics. Run the migration recipe above.
- If they list both `burn = "0.21"` and `burn = { git = ... }`-style mixed sources, or only `git` on the top-level `burn` without `[patch.crates-io]`: warn about version splits and produce the full patch block.
- If they depend on `burn-dispatch` directly: tell them it's internal plumbing. Use `Device::cuda(0)` / `Device::metal(DeviceKind::DefaultDevice)` and remove the direct dep.
- If they hit E0283 "type annotations needed" on `Tensor::<D, Bool>::from_bool(slice.into(), &device)`: explicitly bind `let data: burn::tensor::TensorData = slice.into();` first.
- If they want to use WGPU but forget `#![recursion_limit = "256"]`: they'll hit a confusing E0275 (overflow evaluating type) — add the attribute proactively.
- If they're saving a PyTorch model and getting "missing source values": they saved `torch.save(model, ...)` instead of `torch.save(model.state_dict(), ...)`.
- If they ask for `optimizer.zero_grad()`: there's no such thing — gradients are consumed by `optim.step`.
- If they treat tensors as cheap to keep alive: explain that fusion potential drops with each "extra" live reference, and offer to reorder operations so view ops are grouped.
- If their `Cargo.toml` lists `burn-ndarray`: it still works in 0.21 but is deprecated. Suggest `burn-flex` for new projects (especially WASM/embedded) and the CubeCL CPU backend (`cpu` feature) for production CPU workloads.
- If they have an old `BinFileRecorder` checkpoint and are upgrading: warn them it's not forward-compatible. Convert via 0.20 → `NamedMpkFileRecorder<FullPrecisionSettings>` → 0.21.
- If they have `Ignored<T>` fields, `PaddingConfig::Explicit(1)`, or match on `DType::Bool` directly: 0.21 broke each of those — see `references/migration-0.21.md`.

## Verifying before writing code

Before suggesting code with a specific module, operator, or feature flag, prefer to check the source — the framework is moving fast and your training data is older than the API. Cheap checks:

- `nn` modules live under `crates/burn-nn/src/`
- Tensor operators are in `crates/burn-tensor/src/tensor/api/`
- The `Device` enum and its constructors live in `crates/burn-tensor/src/device.rs`
- Public re-exports for `burn::*` are at `crates/burn/src/lib.rs`

A 30-second grep is much cheaper than getting a feature name wrong.
