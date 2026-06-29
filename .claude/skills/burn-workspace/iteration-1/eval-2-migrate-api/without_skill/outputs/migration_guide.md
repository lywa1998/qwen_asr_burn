# Burn 0.19/0.20 to 0.21 Migration Guide

This guide covers every breaking change in the user's model code. Each section
has the "before" (0.19/0.20) and "after" (0.21+) code diff.

**Warning: records saved with `BinFileRecorder` / `BinBytesRecorder` under 0.20
are not forward-compatible with 0.21.** Run a conversion step under 0.20 before
upgrading (see Section 1 below).

---

## 1. Convert old binary records BEFORE upgrading

Records written with the legacy binary recorders are not forward-compatible.
Run the following **on Burn 0.20** before bumping your `Cargo.toml`:

```rust
use burn::record::{BinFileRecorder, NamedMpkFileRecorder, FullPrecisionSettings};

let recorder = BinFileRecorder::<FullPrecisionSettings>::default();
let record: ModelRecord<B> = recorder.load("old_model.bin", &device).unwrap();

let new_recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::default();
new_recorder.record(record, "model.mpk".into()).unwrap();
```

Then bump to 0.21 and load with `NamedMpkFileRecorder`.

---

## 2. Remove `<B: Backend>` generic from `Module` structs and `impl` blocks

In 0.21, `Tensor` and `Module` no longer carry a `Backend` type parameter.
Backend selection moves to runtime via `burn-dispatch` and `Device`.

### Struct definition

```diff
-#[derive(Module, Debug)]
-pub struct Model<B: Backend> {
-    conv1: Conv2d<B>,
-    conv2: Conv2d<B>,
-    linear: Linear<B>,
-    activation: Gelu,
-}

+#[derive(Module, Debug)]
+pub struct Model {
+    conv1: Conv2d,
+    conv2: Conv2d,
+    linear: Linear,
+    activation: Gelu,
+}
```

### impl block and forward method

```diff
-impl<B: Backend> Model<B> {
-    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 2> {
-        // ...
-    }
-}

+impl Model {
+    pub fn forward(&self, x: Tensor<3>) -> Tensor<2> {
+        // ...
+    }
+}
```

### Device construction (new style)

```diff
-// 0.19/0.20
-use burn::backend::Wgpu;
-type B = Wgpu;
-let device = B::Device::default();

+// 0.21
+use burn::prelude::*;
+let device = Device::wgpu(DeviceKind::DefaultDevice);
+// Other options: Device::cuda(0), Device::metal(), Device::flex()
```

---

## 3. `Gelu` is now a constructed type

`Gelu` now carries an "approximation" flag (exact vs. tanh approximation)
internally. The bare type name no longer works as a value; you must call
`Gelu::new()` or `Gelu::default()`.

```diff
 // Struct field stays the same:
     activation: Gelu,

 // But when initializing the model:
-Model { activation: Gelu, ... }
+Model { activation: Gelu::new(), ... }
+// or equivalently:
+Model { activation: Gelu::default(), ... }
```

---

## 4. `Ignored<T>` → `#[module(skip)]`

The `Ignored` newtype wrapper is deprecated. Replace it with the `#[module(skip)]`
field attribute. The field keeps its original type and visibility.

```diff
-#[derive(Module, Debug)]
-pub struct ConvBlock<B: Backend> {
-    pub padding: Ignored<PaddingConfig1d>,
-    pub conv: Conv1d<B>,
-}

+#[derive(Module, Debug)]
+pub struct ConvBlock {
+    #[module(skip)]
+    pub padding: PaddingConfig1d,
+    pub conv: Conv1d,
+}
```

---

## 5. `PaddingConfig::Explicit` now takes all sides

`Explicit` was changed to support asymmetric padding. 1d now takes 2 values
(left, right); 2d now takes 4 values (top, bottom, left, right).

```diff
 // 1d
-PaddingConfig1d::Explicit(1)
+PaddingConfig1d::Explicit(1, 1)       // (left, right)

 // 2d
-PaddingConfig2d::Explicit(1, 1)
+PaddingConfig2d::Explicit(1, 1, 1, 1) // (top, bottom, left, right)
```

---

## 6. `DType::Bool` match arms must handle the storage discriminator

`DType::Bool` now carries a `BoolStore` variant because different backends
store boolean tensors differently (bit-packed, byte-per-element, or u32-per-element).

### Option A -- wildcard (recommended for most cases)

```diff
 match tensor.dtype() {
-    DType::Bool => { /* handle bool tensor */ }
+    DType::Bool(_) => { /* handle bool tensor, any storage */ }
     _ => unreachable!(),
 }
```

### Option B -- handle each variant explicitly

```diff
 match tensor.dtype() {
-    DType::Bool => { ... }
+    DType::Bool(BoolStore::Native) => { ... }
+    DType::Bool(BoolStore::U8)     => { ... }
+    DType::Bool(BoolStore::U32)    => { ... }
     _ => unreachable!(),
 }
```

---

## 7. `Shape` API changes (bonus -- may apply if you access dims directly)

If your code accesses shape dimensions or catches shape errors:

```diff
-let b = tensor.shape().dims[0];
+let b = tensor.shape()[0];              // Index, not field access

-let swapped = shape.swap(1, 2);
+let swapped = shape.swapped(1, 2);     // Past-tense naming

-let permuted = shape.permute(&[...]);
+let permuted = shape.permuted(&[...]); // Past-tense naming

-ShapeError::RankMismatch{...}
+MetadataError::RankMismatch{...}
```

---

## 8. `powf` / `powf_scalar` removed for `Int` tensors (bonus)

These functions silently truncated results back to integers. Cast explicitly:

```diff
-tensor_int.powf(tensor_float);
+tensor_int.float().powf(tensor_float);            // stay in float
+tensor_int.float().powf(tensor_float).int();      // cast back to int
```

---

## Migration checklist

1. **Convert records.** If you have saved `BinFileRecorder` models, convert them
   to `NamedMpkFileRecorder` under Burn 0.20 before upgrading.
2. Remove `<B: Backend>` from all `#[derive(Module)]` structs, `impl` blocks,
   and `Tensor<B, D>` / `Conv2d<B>` / `Linear<B>` type annotations.
3. Change bare `Gelu` values to `Gelu::new()` or `Gelu::default()`.
4. Replace `Ignored<SomeType>` with `#[module(skip)] pub field: SomeType`.
5. Double all `PaddingConfig1d::Explicit` / quadruple `PaddingConfig2d::Explicit` arguments.
6. Change `DType::Bool` match arms to `DType::Bool(_)` or handle each variant.
7. Bump `burn` to `"0.21"` in `Cargo.toml`.
8. Run `cargo check` and then your test suite.
