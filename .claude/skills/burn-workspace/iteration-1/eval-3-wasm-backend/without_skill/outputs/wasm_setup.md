# Burn Inference in WebAssembly (Browser, Pure CPU)

## Backend Choice: `burn-flex`

For WebAssembly browser inference with **no GPU**, use the **Flex** backend (`burn-flex`). It is Burn 0.21's new pure-Rust eager CPU backend, purpose-built for WASM, embedded, and small-model use cases.

Why Flex (and not the alternatives):

| Backend | WASM? | Why not |
|---------|-------|---------|
| **Flex** | Yes | The right choice. Pure Rust, no GPU dependency, smallest binary. |
| WebGPU (`webgpu` feature) | Yes, but... | Requires the browser to expose `navigator.gpu` (WebGPU). Falls back to nothing on browsers without WebGPU support. Also async init. |
| CubeCL CPU (`cpu` feature) | No | Goes through the CubeCL JIT compiler -- generates and compiles kernels at runtime. WASM cannot JIT-compile the generated kernels. |
| ndarray (`ndarray` feature) | No | Deprecated in 0.21, will be removed. Also depends on BLAS C libraries that do not cross-compile to WASM. |

Flex is eager-only (no kernel fusion, no autotune), but for WASM inference on CPU that is exactly what you want -- minimal binary size, no runtime compilation.

---

## Cargo.toml Features

```toml
[package]
name = "burn-wasm-inference"
version = "0.1.0"
edition = "2021"

[lib]
crate-type = ["cdylib"]        # produce a .wasm binary

[dependencies]
burn = {
    version = "0.21",
    default-features = false,   # strip train, optim, dataset, rl -- not needed for inference
    features = [
        "flex",                 # pure-Rust CPU backend
        "store",                # ModuleRecord serialization (needed for .bpk loading)
        "std",                  # alloc + standard library (WASM has std)
    ]
}

wasm-bindgen = "0.2"
# wasm-bindgen-futures only needed if you use async (Flex is sync, so omit for Flex)

[profile.release]
opt-level = "z"                 # optimize for size
lto = true                      # link-time optimization
```

### Feature breakdown

| Feature | Why |
|---------|-----|
| `default-features = false` | Strips training, RL, datasets, SQLite -- everything not needed for inference. Cuts binary size ~30%. |
| `flex` | The pure-Rust CPU backend. No GPU, no BLAS, no JIT. Runs anywhere WASM runs. |
| `store` | Enables `ModuleRecord::from_bytes()` to load `.bpk` weights from a `&[u8]` slice. |
| `std` | WASM targets have `std` -- this is not `no_std` mode. |

### What you do NOT need

- `autodiff`, `train`, `optim` -- inference-only, no gradients.
- `fusion`, `autotune` -- Flex does not support kernel fusion or autotune (eager-only).
- `webgpu` -- you said no GPU. Flex is pure CPU.

---

## Weight Loading from Compiled-In Bytes

### Step 1: Embed the .bpk file

Use `include_bytes!` to bake your model weights into the WASM binary at compile time:

```rust
static MODEL_WEIGHTS: &[u8] = include_bytes!("../assets/model.bpk");
```

The `.bpk` file sits in your project directory (e.g., `assets/model.bpk`). At compile time the bytes are embedded into the `.wasm` file. No filesystem reads at runtime.

### Step 2: Load the ModuleRecord from bytes

```rust
use burn::store::ModuleRecord;

let record = ModuleRecord::from_bytes(MODEL_WEIGHTS)
    .expect("Failed to deserialize model weights from embedded bytes");
```

### Step 3: Initialize model and load weights

```rust
let device = Device::flex();
let model = ModelConfig::new(/* ... */)
    .init(&device)
    .load_record(record);
```

### How to produce the .bpk file (on the training side)

```rust
// Run this in your training code (native, not WASM)
model.into_record().save("model").expect("Failed to save");
// Produces: model.bpk
```

Then copy `model.bpk` into your WASM project's `assets/` directory.

### Alternative: load from network at startup

```rust
use wasm_bindgen_futures::JsFuture;
use web_sys::{Request, RequestInit, RequestMode, Response};

pub async fn fetch_model_bytes(url: &str) -> Result<Vec<u8>, JsValue> {
    let mut opts = RequestInit::new();
    opts.method("GET");
    opts.mode(RequestMode::Cors);

    let request = Request::new_with_str_and_init(url, &opts)?;
    let window = web_sys::window().unwrap();
    let resp_value = JsFuture::from(window.fetch_with_request(&request)).await?;
    let resp: Response = resp_value.dyn_into().unwrap();
    let buffer = JsFuture::from(resp.array_buffer()?).await?;
    let array = js_sys::Uint8Array::new(&buffer);
    Ok(array.to_vec())
}
```

`include_bytes!` is preferred because it avoids startup latency and network dependency.

---

## Browser Limitations to Know

1. **Single-threaded execution.** WASM in browsers runs on the main thread by default. Flex is fully synchronous, so a large model's `forward()` call will block the UI. Mitigations:
   - Use Web Workers to move inference off the main thread.
   - Keep models small (Flex is designed for small models).
   - For larger models, consider the WebGPU backend (`webgpu` feature) which runs asynchronously.

2. **No recursion_limit needed.** The `#![recursion_limit = "256"]` attribute is **only** required for CubeCL/WGPU-family backends. Flex does not need it.

3. **Memory.** WASM has a 4 GB memory ceiling (32-bit address space). Model weights + activations must fit within that. Flex allocates eagerly, so monitor `wasm_bindgen::memory()` usage.

4. **Binary size targets (rough):**
   - Flex CPU inference (small CNN, `opt-level = "z"` + LTO): ~500 KB stripped
   - Flex + store + std (no model weights): ~1-2 MB
   - With model weights embedded: adds the size of the .bpk file

---

## Quick Reference

```rust
// Backend
let device = Device::flex();

// Weight loading
static MODEL: &[u8] = include_bytes!("../assets/model.bpk");
let record = ModuleRecord::from_bytes(MODEL).unwrap();
let model = ModelConfig::new(/* params */).init(&device).load_record(record);

// Inference
let input = Tensor::<D>::from_floats(input_data, &device);
let output = model.forward(input);
let result: Vec<f32> = output.into_data().to_vec().unwrap();
```
