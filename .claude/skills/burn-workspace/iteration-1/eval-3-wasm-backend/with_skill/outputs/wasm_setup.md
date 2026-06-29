# Burn Inference in a WebAssembly Browser App (Pure CPU)

## Backend Choice: Flex

For WASM inference in a browser with **no GPU, pure CPU**, use the **Flex backend** (`burn-flex`). This is the correct choice because:

- **Flex is Burn 0.21's new pure-Rust eager CPU backend**, replacing `burn-ndarray` (deprecated).
- Target use cases explicitly include **WebAssembly, embedded, and small models**.
- No GPU required -- runs anywhere WASM does (`wasm32-unknown-unknown`, `wasm32-wasip1`).
- Smallest binary footprint among all Burn backends.
- Eager execution only (no kernel fusion, no autotune), which is fine for inference.

### Why not WebGPU?

The `webgpu` feature requires a browser GPU (WebGPU API). Since the constraint is "no GPU, pure CPU," Flex is the only WASM-compatible choice. The WebGPU backend would fail to initialize without a GPU device.

### Comparison Table

| Backend | WASM? | GPU Required? | Binary Size | Fusion/Autotune |
|---------|-------|---------------|-------------|-----------------|
| Flex    | Yes   | No (pure CPU) | Smallest    | No              |
| WebGPU  | Yes   | Yes           | Medium      | Yes             |
| ndarray | Yes   | No (CPU+BLAS) | Medium      | No (deprecated) |

---

## Cargo.toml Features

```toml
[dependencies]
burn = {
    version = "0.21",
    default-features = false,
    features = ["flex", "store", "std"],
}
wasm-bindgen = "0.2"

[profile.release]
opt-level = "z"     # optimize for size
lto = true          # link-time optimization
codegen-units = 1   # better LTO results
```

### Feature Breakdown

| Feature | Why It's Needed |
|---------|----------------|
| `flex` | Pure-Rust CPU backend. This is the engine that runs tensor ops on CPU. |
| `store` | Required for `ModuleRecord::from_bytes()` -- loading `.bpk` from a byte slice. |
| `std` | WASM in the browser has `std` available. Needed for `ModuleRecord` serialization paths. |
| `default-features = false` | Strips `train`, `optim`, `dataset`, `rl`, and other training-only features. Saves ~30% binary size. |

### Features You Do NOT Need

- `train`, `optim`, `rl`, `dataset` -- training only, adds unnecessary binary weight.
- `wgpu`, `webgpu`, `cuda`, `metal`, `vulkan`, `rocm` -- GPU backends, irrelevant.
- `fusion`, `autotune` -- Flex is eager-only; these have no effect on Flex.
- `pytorch`, `safetensors` -- only needed for loading PyTorch/SafeTensors weights, not `.bpk`.
- `ndarray` -- deprecated in 0.21, do not use for new projects.

---

## Loading Model Weights from a Compiled-In Bytes Slice

Burn 0.21's `ModuleRecord` API supports loading directly from a `&[u8]`, which is perfect for `include_bytes!`:

```rust
use burn::store::ModuleRecord;

// Compile-time embedding -- no filesystem access at runtime
static MODEL_WEIGHTS: &[u8] = include_bytes!("../assets/model.bpk");

// Parse the bytes into a ModuleRecord (backend-independent)
let record = ModuleRecord::from_bytes(MODEL_WEIGHTS)
    .expect("Failed to parse compiled-in model weights");

// Initialize model on the Flex device, then load the weights
let device = Device::flex();
let model = ModelConfig::new(/* ... */)
    .init(&device)
    .load_record(record);
```

### Key Points

1. **`include_bytes!` embeds the file at compile time.** The `.bpk` bytes become part of the WASM binary. No `fetch()`, no filesystem, no network request at runtime.

2. **`ModuleRecord::from_bytes` is `no_std` compatible.** It does not touch the filesystem -- pure in-memory parsing.

3. **The `.bpk` format is backend-independent.** You can train on CUDA, save as `.bpk`, and load on Flex in WASM.

4. **Optional load-time controls:**
   ```rust
   let record = ModuleRecord::from_bytes(MODEL_WEIGHTS)?
       .allow_partial(true)            // ignore missing tensors (for transfer learning)
       .validate(false)                 // skip shape checks (faster, less safe)
       .cast_to_module_dtype();         // auto-cast dtypes to match module
   ```

### Half-Precision Weights (Smaller Binary)

To reduce the embedded `.bpk` size by ~50%, save weights as F16:

```rust
// At save time (on the training side):
use burn_store::{BurnpackStore, HalfPrecisionAdapter};

let adapter = HalfPrecisionAdapter::new()
    .without_module("BatchNorm");   // BatchNorm F16 can underflow
let mut store = BurnpackStore::from_file("model_f16.bpk")
    .with_to_adapter(adapter);
model.save_into(&mut store)?;
```

Then `include_bytes!("model_f16.bpk")` in the WASM build. The Flex backend will load F16 weights and cast to F32 automatically.

---

## Build and Deploy

```bash
# Build for web
wasm-pack build --target web --release

# Or with cargo directly
cargo build --target wasm32-unknown-unknown --release
```

### No `#![recursion_limit = "256"]` Required

This is a WGPU/CubeCL requirement. Flex does not use CubeCL, so the default recursion limit (128) is sufficient.

### Expected Binary Size (approximate)

- Small CNN with Flex: ~500 KB stripped
- Medium model, F16 weights embedded: ~1-3 MB compressed WASM

---

## Common Pitfalls

| Pitfall | Fix |
|---------|-----|
| Using `ndarray` feature | `ndarray` is deprecated in 0.21. Use `flex`. |
| Adding `webgpu` feature | This requires a browser GPU. For CPU-only, use only `flex`. |
| Forgetting `store` feature | Required for `ModuleRecord::from_bytes()`. Without it, you can only load from filesystem. |
| Trying `Device::default()` blindly | `Default::default()` picks Flex anyway, but be explicit: `Device::flex()`. |
| Old `Tensor<B, D>` style | Use Burn 0.21 API: `Tensor<D>` with no backend generic. |
| Thinking `ModuleRecord` needs files | `from_bytes` and `into_bytes` are pure in-memory. Filesystem is optional. |
