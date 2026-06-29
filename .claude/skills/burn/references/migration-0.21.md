# Migrating from Burn 0.20 to 0.21

This file is the consolidated migration guide. Every breaking change called out in the 0.21 release post is here with a working diff.

When migrating an existing project, plan to do this in three rounds:
1. **Recorders first** — convert any `BinFileRecorder` / `BinBytesRecorder` checkpoints under 0.20 (records are not forward-compatible).
2. **Compile against 0.21** — fix every error below.
3. **Run tests** — the changes are mostly type-level; runtime behavior is unchanged for the documented cases.

## 1. `TensorData::shape` is `Shape`, not `Vec<usize>`

The `shape` field on `TensorData` changed type. Most call sites that just reshape or read dims keep working, but anything that constructed a `Vec<usize>` for `TensorData::new(...)` needs updating.

```rust
// 0.20
let data = TensorData::new(values, vec![3, 4]);

// 0.21
let data = TensorData::new(values, Shape::new([3, 4]));
```

Records written with `BinFileRecorder` / `BinBytesRecorder` are **not forward-compatible**. Convert them under 0.20 first:

```rust
// Run this on 0.20 BEFORE upgrading
use burn::record::{BinFileRecorder, NamedMpkFileRecorder, FullPrecisionSettings};

let recorder = BinFileRecorder::<FullPrecisionSettings>::default();
let record: ModelRecord<B> = recorder.load("old_model.bin", &device).unwrap();

let new_recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::default();
new_recorder.record(record, "model.mpk".into()).unwrap();
```

Then upgrade to 0.21 and load with `NamedMpkFileRecorder`.

## 2. `Ignored<T>` → `#[module(skip)]`

```rust
// 0.20
#[derive(Module, Debug)]
pub struct Conv1d<B: Backend> {
    pub padding: Ignored<PaddingConfig1d>,
}

// 0.21
#[derive(Module, Debug)]
pub struct Conv1d<B: Backend> {
    #[module(skip)]
    pub padding: PaddingConfig1d,
}
```

`Ignored<T>` is deprecated, not removed — old code keeps compiling with a warning. Take the chance to migrate.

## 3. `PaddingConfig::Explicit` takes all sides

Both `PaddingConfig1d` and `PaddingConfig2d` now support asymmetric padding. The `Explicit` variant takes one value per side.

```rust
// 0.20
PaddingConfig1d::Explicit(1)
PaddingConfig2d::Explicit(1, 1)

// 0.21
PaddingConfig1d::Explicit(1, 1)               // (left, right)
PaddingConfig2d::Explicit(1, 1, 1, 1)         // (top, bottom, left, right)
```

`PaddingConfig3d` does not yet support asymmetric padding, so its `Explicit` variant is unchanged.

## 4. `Gelu` is now a constructed type

```rust
// 0.20
#[derive(Module)]
struct Block { activation: Gelu }
// or as a field literal:
Block { activation: Gelu, ... }

// 0.21
Block { activation: Gelu::new(), ... }
// or
Block { activation: Gelu::default(), ... }
```

The reason: `Gelu` now carries an "approximation" flag (exact vs tanh approximation). Most code wants `Gelu::new()`.

## 5. `PositionWiseFeedForward` field rename

```rust
// 0.20
#[derive(Module)]
struct PositionWiseFeedForward<B: Backend> {
    linear_inner: Linear<B>,
    linear_outer: Linear<B>,
    dropout: Dropout,
    gelu: Gelu,
}

// 0.21
#[derive(Module)]
struct PositionWiseFeedForward<B: Backend> {
    linear_inner: Linear<B>,
    linear_outer: Linear<B>,
    dropout: Dropout,
    #[module(skip)]
    activation: Activation<B>,
}
```

The activation is now configurable, not hardcoded to GELU. The `#[module(skip)]` preserves record compatibility — the activation isn't a learned parameter.

If you implemented this module yourself (with the same field name), keep doing what you were doing; this only matters if you depended on the built-in.

## 6. `Shape` API: private fields and renamed methods

```rust
// 0.20
let b = tensor.shape().dims[0];
if let Err(ShapeError::RankMismatch{..}) = lhs.broadcast(&rhs) { ... }
let shape = shape.swap(1, 2).unwrap();
let shape = shape.permute(&[0, 2, 1, 3]).unwrap();

// 0.21
let b = tensor.shape()[0];
if let Err(MetadataError::RankMismatch{..}) = lhs.broadcast(&rhs) { ... }
let shape = shape.swapped(1, 2).unwrap();
let shape = shape.permuted(&[0, 2, 1, 3]).unwrap();
```

Summary:
- **Field access** → indexing: `.dims[i]` → `[i]`
- **`swap` → `swapped`**, **`permute` → `permuted`** (the new names imply "returns a new shape", which matches the value-returning behavior).
- **`ShapeError` → `MetadataError`** in error returns.

## 7. `DType::Bool` carries a storage discriminator

```rust
// 0.20
match bool_tensor.dtype() {
    DType::Bool => todo!(),
    _ => unreachable!(),
}

// 0.21
match bool_tensor.dtype() {
    DType::Bool(BoolStore::Native) => todo!(),
    DType::Bool(BoolStore::U8) => todo!(),
    DType::Bool(BoolStore::U32) => todo!(),
    _ => unreachable!(),
}
```

Different backends store boolean tensors differently (native bit-packed, byte-per-element, or u32-per-element). Code that branches on `DType::Bool` needs to handle all three or use an ignore pattern: `DType::Bool(_)`.

## 8. `powf` and `powf_scalar` removed for `Int` tensors

Pre-0.21 had an implicit lossy truncation when raising an integer tensor to a float power. That's now a compile error.

```rust
// 0.20 — silently truncates the result back to int
let result = tensor_int.powf(tensor_float);

// 0.21 — explicit cast
let result = tensor_int.float().powf(tensor_float);
// If you actually want an int back at the end:
let result = tensor_int.float().powf(tensor_float).int();
```

Same change for `powf_scalar`.

## 9. Backend trait surface (implementor-facing)

This only matters if you're **writing** a backend, not just using one. Skip if your code says `use burn::*` and stops there.

Tensor creation/conversion ops now take an explicit output `dtype`:

```rust
// 0.20
fn bool_empty(shape: Shape, device: &Device<Self>) -> BoolTensor<Self> {
    // ...
}

fn bool_into_int(tensor: BoolTensor<Self>) -> IntTensor<Self> {
    // ...
}

// 0.21
fn bool_empty(
    shape: Shape,
    device: &Device<Self>,
    dtype: BoolDType,
) -> BoolTensor<Self> {
    // ...
}

fn bool_into_int(
    tensor: BoolTensor<Self>,
    out_dtype: IntDType,
) -> IntTensor<Self> {
    // ...
}
```

Associated types moved from `Backend` to a new `BackendTypes` trait. Use these aliases in implementor code:

```rust
type Device<B>      = <<B as Backend>::Types as BackendTypes>::Device;
type FloatTensor<B> = <<B as Backend>::Types as BackendTypes>::FloatTensor;
type BoolTensor<B>  = <<B as Backend>::Types as BackendTypes>::BoolTensor;
type IntTensor<B>   = <<B as Backend>::Types as BackendTypes>::IntTensor;
```

## 10. `burn-dataset` cache directory

Cache locations now follow platform conventions:

| Platform | Old | New |
| -------- | --- | --- |
| Linux (no `XDG_CACHE_HOME`) | `~/.cache/burn-dataset/...` | `~/.cache/burn-dataset/...` (unchanged) |
| Linux (`XDG_CACHE_HOME` set) | `~/.cache/burn-dataset/...` | `$XDG_CACHE_HOME/burn-dataset/...` |
| macOS | `~/.cache/burn-dataset/...` | `~/Library/Caches/burn-dataset/...` |
| Windows | `~/.cache/burn-dataset/...` | `{FOLDERID_LocalAppData}\burn-dataset\...` |

Already-downloaded datasets in the old location aren't auto-migrated. Either copy them into the new location once or let them re-download.

## What did NOT break

A lot of the framework didn't move. If you don't see it in the list above, assume it still works:
- The `Module` derive macro and `Module` trait usage
- `Tensor` operations beyond `powf` for ints (matmul, reshape, slice, gather, etc. — all unchanged)
- The `Learner` and `SupervisedTraining` API
- `nn` module surface (Linear, Conv*, BatchNorm, LayerNorm, Dropout, etc.)
- `burn-store` PyTorch / SafeTensors path
- Most of `burn-train` (besides the new RL additions)
- Optimizers (`AdamConfig`, `AdamWConfig`, `SgdConfig`, etc.)

## Recommended migration order

1. Convert old `BinFileRecorder` records on 0.20 → `NamedMpkFileRecorder<FullPrecisionSettings>`.
2. Bump `burn` to `0.21` in `Cargo.toml`. Run `cargo check` to surface errors.
3. Walk through items 2–8 above in any order — they're independent.
4. Run your test suite. The breaks are mostly type-level; behavior should match.
5. Optional: replace `burn-ndarray` with `burn-flex` (for embedded/WASM) or the CubeCL `cpu` backend (for production CPU). `burn-ndarray` will be removed in one or two more releases.
