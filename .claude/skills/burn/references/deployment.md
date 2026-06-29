# Deployment: no_std, WASM, embedded

Burn supports running anywhere from a server with eight A100s down to a microcontroller — same code, different feature set. This file covers the deployment-specific knobs.

## `no_std`

`burn`'s `default-features = false` build runs without `std`. The Flex backend (pure-Rust CPU) is the recommended choice for `no_std`.

```toml
[dependencies]
burn = {
    version = "0.21",
    default-features = false,
    features = ["flex", "store"],     # add what you need
}
```

What stays:

- `Tensor` and all tensor operations
- `Module` derive, `nn::*` modules
- `ModuleRecord` / burnpack — load weights from `&[u8]` instead of files
- Autodiff (with `autodiff` feature)

What's gone:

- `DataLoader` (relies on threads)
- `Learner` / `SupervisedTraining` (training is generally `std`-only)
- `sqlite`, `vision::MnistDataset`, the `data` module
- Filesystem-based record save/load — use `ModuleRecord::from_bytes(&[u8])`

The pattern for embedded inference:

```rust
#![no_std]
extern crate alloc;

use burn::{prelude::*, store::ModuleRecord};

static MODEL_WEIGHTS: &[u8] = include_bytes!("../assets/model.bpk");

fn run_inference(input: &[f32]) -> alloc::vec::Vec<f32> {
    let device = Device::flex();
    let record = ModuleRecord::from_bytes(MODEL_WEIGHTS).unwrap();
    let model = ModelConfig::new().init(&device).load_record(record);

    let input = Tensor::<1>::from_floats(input, &device);
    let output = model.forward(input);
    output.into_data().to_vec().unwrap()
}
```

You'll need an allocator (`alloc`) on most embedded targets; Burn doesn't run on heap-less targets.

## WebAssembly

Two flavors:

1. **WASM in browser, WebGPU** — fast GPU inference in the browser.
2. **WASM in any runtime, Flex** — pure-Rust CPU, runs anywhere WASM does.

### Browser + WebGPU

```toml
[dependencies]
burn = { version = "0.21", default-features = false, features = ["webgpu", "std"] }
wasm-bindgen = "..."
wasm-bindgen-futures = "..."
```

```rust
use burn::prelude::*;

#[wasm_bindgen]
pub async fn run_model(input: Vec<f32>) -> Vec<f32> {
    let device = Device::webgpu(DeviceKind::DefaultDevice);
    // ... load model, run inference, return ...
}
```

The `examples/mnist-inference-web/` directory has a complete example. Notes:

- The `#![recursion_limit = "256"]` requirement applies (WebGPU is part of the wgpu family).
- WebGPU initialization is async — your top-level entry needs to `await` device creation.
- Bundle the model weights via `include_bytes!`. Loading from network at startup is fine but adds latency.

### WASM + Flex (any runtime)

```toml
burn = { version = "0.21", default-features = false, features = ["flex", "store"] }
```

This runs on `wasm32-unknown-unknown`, `wasm32-wasip1`, etc. Slower than WebGPU but no platform GPU requirement.

## Embedded (microcontrollers)

Burn runs on ARM Cortex-M / RISC-V with `no_std` + Flex backend. Tested setups:
- ARM Cortex-M with `cortex-m-rt`
- RISC-V with `riscv-rt`
- `nrf52`, `stm32` family via their respective HALs

Memory profile depends entirely on your model — a small CNN can fit in 256 KB of flash and ~64 KB of RAM if you keep weights F16 or quantized.

Tips:

- **Bake weights into flash** with `include_bytes!` + `ModuleRecord::from_bytes`. Don't try to load from filesystem.
- **Use F16 weights** via `HalfPrecisionAdapter` at save time.
- **Quantize** further with `tensor.quantize` if backends support it (currently limited on Flex).
- **Avoid `Dropout`** at inference — it has no effect but takes code space.

See `burn-book/src/advanced/no-std.md` for more.

## Server / production

For a production inference server, the typical setup:

```toml
burn = { version = "0.21", features = [
    "cuda",          # or your target backend
    "fusion",
    "autotune",
    "store",
    "pytorch",       # if loading PyTorch weights
    "safetensors",   # if loading HF SafeTensors
    "std",
] }
```

Pre-deployment checklist:

- **Bundle the autotune cache.** Copy `$XDG_CACHE_HOME/cubecl/` into the deployment image so cold starts don't pay autotune cost. See `burn-book/src/performance/good-practices/kernel-selection.md`.
- **Embed weights or memory-map them.** Use `BurnpackStore::from_static(...)` for compile-time embedding, `.zero_copy(true)` for runtime mmap. Both avoid copy-on-load.
- **Use `model.no_grad()`** after loading to make sure no parameter is tracked for autodiff. Saves memory.
- **Use `Device::cuda(0)` (or your backend) without `.autodiff()`** — inference doesn't need autodiff.
- **Batch requests.** Throughput scales near-linearly with batch size until you hit memory limits. A request-batcher in front of the model is usually a bigger win than micro-optimizing the kernel.

## Multi-device / multi-GPU inference

Use the **router** feature to split work:

```rust
use burn::router::RouterDevice;

let device = RouterDevice::new(vec![Device::cuda(0), Device::cuda(1)])?;
```

Each tensor op gets dispatched to one of the underlying devices automatically. For more control over placement, use multiple `Device` values and `tensor.to_device` explicitly.

## Remote execution

The `remote` feature exposes a remote-device that forwards ops over the network to a server running `burn-remote::server`. Useful for thin clients (e.g. mobile app, embedded) that offload compute to a beefier machine.

```rust
let device = Device::remote("tcp://192.168.1.10:7777", 0);
let t = Tensor::<2>::from_floats([[1.0, 2.0]], &device);
let result = model.forward(t);    // computation happens on the remote
```

Server side runs a small daemon. See `crates/burn-remote/` for the protocol.

## Binary size

For size-sensitive deployments:

- `default-features = false` strips ~30% of binary size.
- Drop `train`, `optim`, `rl`, `dataset` features if doing inference only.
- LTO + `opt-level = "z"` in `release` profile.
- Tools: `cargo-bloat`, `cargo-llvm-lines` to find what's pulling weight.
- The Flex backend has a smaller code footprint than CubeCL backends.

Typical sizes (rough):
- CPU inference (Flex, no_std, small CNN): ~500 KB stripped
- WGPU inference (browser, no model): ~2-3 MB compressed WASM
- CUDA training (full features): ~30-50 MB

## Cross-compilation

```bash
# embedded ARM
cargo build --target thumbv7em-none-eabihf --release

# wasm
wasm-pack build --target web

# Linux from macOS (cross)
cross build --target x86_64-unknown-linux-gnu --release
```

Burn doesn't have target-specific build scripts beyond what cargo/`build.rs` already gives you, so standard cross-compilation tooling works.

## Deployment checklist (server inference)

```
[ ] Built with --release (or production profile)
[ ] Backend feature matches deployment hardware
[ ] fusion feature on (massive perf win)
[ ] autotune feature on, cache bundled or warmed
[ ] Weights embedded or memory-mapped, no fs reads in hot path
[ ] model.no_grad() applied after loading
[ ] No .into_scalar() inside request handling
[ ] Recursion limit set if using any wgpu-family backend
[ ] Single sync per request, not per layer
[ ] Request batching in front of the model
```
