# Saving and loading

Burn has three layers for persisting models. Pick the smallest one that does the job.

## 0.21 forward-compatibility heads-up

Records written by `BinFileRecorder` or `BinBytesRecorder` in Burn 0.20 are **not forward-compatible** with 0.21. The on-disk format changed under those recorders.

If a user is upgrading and has old checkpoints:

```rust
// Run this code under Burn 0.20, BEFORE upgrading
use burn::record::{BinFileRecorder, NamedMpkFileRecorder, FullPrecisionSettings};

let old = BinFileRecorder::<FullPrecisionSettings>::default();
let record = old.load("old_checkpoint.bin", &device).unwrap();

let new = NamedMpkFileRecorder::<FullPrecisionSettings>::default();
new.record(record, "checkpoint.mpk".into()).unwrap();
```

Then upgrade to 0.21 and load with `NamedMpkFileRecorder<FullPrecisionSettings>`. The `NamedMpk` format is stable.

The high-level `model.into_record().save("...")` path that uses burnpack `.bpk` is unaffected — only the explicit `BinFileRecorder`/`BinBytesRecorder` paths.

## Layer 1: `ModuleRecord` + burnpack (`.bpk`)

The default path. `model.into_record()` produces a `ModuleRecord` that serializes to Burn's compact binary format. Backend-independent — save on GPU, load on CPU.

```rust
use burn::store::ModuleRecord;

// Save
model.into_record().save("model")?;     // writes "model.bpk"

// Load
let record = ModuleRecord::load("model")?;
let model = ModelConfig::new(...).init(&device).load_record(record);
```

The `.bpk` extension is added automatically when the path has no extension. Pass `"model.bpk"` or `"model"` — both work.

Burnpack format: small fixed header + CBOR metadata + tensor data section. Tensors start on 256-byte boundaries so they can be memory-mapped without copies.

### Load-time controls

```rust
let record = ModuleRecord::load("model")?
    .allow_partial(true)            // ignore missing tensors
    .validate(false)                 // skip shape-mismatch checks
    .cast_to_module_dtype();         // cast record dtypes to module dtypes on load
let model = model.load_record(record);

// Fallible variant
match model.try_load_record(record) {
    Ok(m) => m,
    Err(e) => { eprintln!("load failed: {e}"); model }
}
```

### Saving and loading via bytes (no_std friendly)

```rust
let bytes: Vec<u8> = model.into_record().into_bytes()?;
let record = ModuleRecord::from_bytes(&bytes)?;

// Compile-time embedding (great for binary-only deployment)
static MODEL_DATA: &[u8] = include_bytes!("../assets/model.bpk");
let record = ModuleRecord::from_bytes(MODEL_DATA)?;
```

### Optimizer and scheduler state

Same shape of API, used for training checkpoint/resume:

```rust
optimizer.save("optim")?;
let optimizer = optimizer.load("optim")?;

scheduler.to_record().save("scheduler")?;
let scheduler = scheduler.load_record(LrSchedulerRecord::load("scheduler")?);
```

The `Learner` + `SupervisedTraining` flow saves all three automatically when you call `.with_checkpointer()`.

## Layer 2: `burn-store` (PyTorch, SafeTensors, zero-copy)

Enable with the `pytorch`, `safetensors`, or `store` features. Adds:

- **Zero-copy memory-mapped loading** (huge models load instantly)
- **PyTorch `.pt` reading** (read-only)
- **SafeTensors read/write**
- **Key remapping** (rename tensors during load)
- **Partial loading** (load what's there, randomize the rest)
- **Filtering** (load/save only matching tensors)

### Pattern: load a HuggingFace SafeTensors checkpoint

```rust
use burn_store::{ModuleSnapshot, PyTorchToBurnAdapter, SafetensorsStore};

let device = Default::default();
let mut model = MyModel::init(&device);

let mut store = SafetensorsStore::from_file("model.safetensors")
    .with_from_adapter(PyTorchToBurnAdapter);   // handles conv weight layout differences
let result = model.load_from(&mut store)?;
```

For SafeTensors files **produced by Burn**, omit the adapter.

### Pattern: load a PyTorch `.pt` checkpoint

```rust
use burn_store::{ModuleSnapshot, PytorchStore};

let mut store = PytorchStore::from_file("pytorch_model.pt");
model.load_from(&mut store)?;
```

If the checkpoint nests the state dict under a key (e.g. `{"state_dict": {...}}`):

```rust
let mut store = PytorchStore::from_file("checkpoint.pt")
    .with_top_level_key("state_dict");
```

### Pattern: key remapping

If your Burn model's parameter names don't match the source:

```rust
let mut store = PytorchStore::from_file("model.pt")
    .with_key_remapping(r"^model\.", "")            // strip "model." prefix
    .with_key_remapping(r"^layer", "encoder.layer"); // rename
model.load_from(&mut store)?;
```

For complex remapping rules:

```rust
use burn_store::KeyRemapper;

let remapper = KeyRemapper::new()
    .add_pattern(r"^transformer\.h\.(\d+)\.", "transformer.layer$1.")?
    .add_pattern(r"\.attn\.", ".attention.")?;
let store = SafetensorsStore::from_file("model.safetensors").remap(remapper);
```

### Pattern: partial loading

```rust
let mut store = PytorchStore::from_file("pretrained.pt").allow_partial(true);
let result = model.load_from(&mut store)?;

println!("Loaded: {}", result.applied.len());
println!("Missing (kept their init): {:?}", result.missing);
println!("Errors: {:?}", result.errors);
```

**Important:** inspecting `result.missing`/`result.errors` requires `.allow_partial(true)`. Without it, a missing tensor causes a hard `Err` before you ever see an `ApplyResult`.

### Pattern: filtering

Load or save only some layers:

```rust
let mut store = SafetensorsStore::from_file("model.safetensors")
    .with_regex(r"^encoder\..*")        // include encoder
    .with_regex(r".*\.bias$")           // OR any bias
    .with_full_path("decoder.scale")    // OR specific path
    .allow_partial(true);

model.load_from(&mut store)?;
```

The `with_*` filters compose with OR.

### Pattern: zero-copy

For large files or embedded models:

```rust
// Compile-time embedded
static MODEL: &[u8] = include_bytes!("model.bpk");
let mut store = BurnpackStore::from_static(MODEL);
model.load_from(&mut store)?;

// Memory-mapped file
let mut store = BurnpackStore::from_file("large_model.bpk").zero_copy(true);
model.load_from(&mut store)?;
```

### Pattern: half-precision storage

Cut file size ~50% by saving as F16 and loading back as F32:

```rust
use burn_store::{BurnpackStore, HalfPrecisionAdapter};

let adapter = HalfPrecisionAdapter::new();

// Save F32 -> F16
let mut store = BurnpackStore::from_file("model_f16.bpk")
    .with_to_adapter(adapter.clone());
model.save_into(&mut store)?;

// Load F16 -> F32
let mut store = BurnpackStore::from_file("model_f16.bpk")
    .with_from_adapter(adapter);
model.load_from(&mut store)?;
```

By default the adapter converts weights in Linear, Embedding, Conv\*, LayerNorm, GroupNorm, InstanceNorm, RmsNorm, and PRelu. It **excludes BatchNorm** because its running variance can underflow in F16.

```rust
let adapter = HalfPrecisionAdapter::new()
    .without_module("LayerNorm")        // keep LayerNorm at F32
    .with_module("CustomLayer");        // include a custom module
```

### Pattern: model surgery / transfer learning

```rust
use burn_store::{ModuleSnapshot, PathFilter};

// Copy encoder from model1 to model2
let filter = PathFilter::new().with_regex(r"^encoder\..*");
let snapshots = model1.collect(Some(filter.clone()), None, false);
model2.apply(snapshots, Some(filter), None, false);
```

### Inspecting a file

```rust
use burn_store::ModuleStore;

let mut store = PytorchStore::from_file("model.pt");
let names = store.keys()?;
for name in names {
    if let Some(snap) = store.get_snapshot(&name)? {
        println!("{name}: shape={:?} dtype={:?}", snap.shape, snap.dtype);
    }
}
```

## Layer 3: ONNX

For static models (mostly inference), use `burn-onnx` (external crate). It generates Rust source code from an `.onnx` file, producing a `Module` that mirrors the ONNX graph.

```bash
cargo install burn-import
burn-import model.onnx --output ./src/model_generated.rs
```

The generated code is regular Burn — you can fine-tune it like any other model.

## Exporting from PyTorch for Burn

Critical: save the **state_dict**, not the model object:

```python
import torch
model = MyModel()
torch.save(model.state_dict(), "model.pt")       # correct
# torch.save(model, "model.pt")                  # wrong — Burn can't load this
```

For SafeTensors:

```python
from safetensors.torch import save_file
save_file(model.state_dict(), "model.safetensors")
```

## Common errors

| Symptom | Cause | Fix |
| ------- | ----- | --- |
| "Missing source values" | Saved entire PyTorch model, not state_dict | Re-export with `torch.save(model.state_dict(), ...)` |
| "Shape mismatch on `conv1.weight`" | Burn model architecture differs from source | Verify layer configs (in/out channels, kernel size, bias setting) |
| "Key not found" | Parameter names differ | Inspect with `store.keys()?`, then `with_key_remapping(...)` |
| "Top-level key 'state_dict' not found" | PyTorch checkpoint not nested | Remove `.with_top_level_key(...)` |
| `Tensor` deserialization fails on `f16` BatchNorm | F16 underflow in running variance | Use `HalfPrecisionAdapter::new().without_module("BatchNorm")` (the default) |

## Inspecting `.bpk` files

```bash
cargo run --example burnpack_inspect model.bpk
```

For `.pt` and `.safetensors`, [Netron](https://github.com/lutzroeder/netron) gives you a visual graph.

## Cross-framework export from Burn

To save a model for PyTorch consumption:

```rust
use burn_store::{BurnToPyTorchAdapter, SafetensorsStore};

let mut store = SafetensorsStore::from_file("for_pytorch.safetensors")
    .with_to_adapter(BurnToPyTorchAdapter)
    .skip_enum_variants(true);
model.save_into(&mut store)?;
```

## Metadata

Burnpack and SafeTensors support custom key/value metadata on the file:

```rust
let mut store = BurnpackStore::from_file("model.bpk")
    .metadata("version", "1.0")
    .metadata("git_sha", "abc1234")
    .metadata("epochs", "100");
model.save_into(&mut store)?;
```
