# Burn 0.19 / 0.20 → 0.21 Migration Guide

This guide covers every breaking change that applies to the user's model code.
Each section has the "before" (0.19/0.20) and "after" (0.21+) diff.

---

## 1. Remove the `B: Backend` generic from user-facing code

In 0.21, `Tensor` and `Module` no longer carry a `Backend` type parameter.
Backend selection moves to the `Device` layer via `burn-dispatch`.

```diff
-#[derive(Module, Debug)]
-pub struct Model<B: Backend> {
-    conv1: Conv2d<B>,
-    conv2: Conv2d<B>,
-    linear: Linear<B>,
-    activation: Gelu,
-}

+use burn::nn::{conv::Conv2d, Linear};

+#[derive(Module, Debug)]
+pub struct Model {
+    conv1: Conv2d,
+    conv2: Conv2d,
+    linear: Linear,
+    activation: Gelu,
+}
```

```diff
-impl<B: Backend> Model<B> {
-    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 2> {
-        ...
-    }
-}

+impl Model {
+    pub fn forward(&self, x: Tensor<3>) -> Tensor<2> {
+        ...
+    }
+}
```

**Device construction (new style):**
```rust
// 0.19/0.20
use burn::backend::Wgpu;
type B = Wgpu;
let device = B::Device::default();

// 0.21
use burn::prelude::*;
let device = Device::wgpu(DeviceKind::DefaultDevice);
// or: Device::cuda(0), Device::metal(), Device::flex()
```

---

## 2. `Gelu` is now a constructed type

`Gelu` now carries an "approximation" flag internally. You must call `Gelu::new()`
or `Gelu::default()` instead of using the bare type name.

```diff
-    activation: Gelu,
+    activation: Gelu,

 // And when initializing:
-Model { activation: Gelu, ... }
+Model { activation: Gelu::new(), ... }
```

---

## 3. `Ignored<T>` → `#[module(skip)]`

The `Ignored` newtype wrapper is deprecated. Use the `#[module(skip)]` field
attribute instead. The field keeps its original type.

```diff
-#[derive(Module, Debug)]
-pub struct Conv1d<B: Backend> {
-    pub padding: Ignored<PaddingConfig1d>,
-}

+#[derive(Module, Debug)]
+pub struct Conv1d {
+    #[module(skip)]
+    pub padding: PaddingConfig1d,
+}
```

---

## 4. `PaddingConfig::Explicit` now takes all sides

`Explicit` was changed to support asymmetric padding. 1d now takes 2 values;
2d now takes 4 values.

```diff
 // 1d
-PaddingConfig1d::Explicit(1)
+PaddingConfig1d::Explicit(1, 1)       // (left, right)

 // 2d
-PaddingConfig2d::Explicit(1, 1)
+PaddingConfig2d::Explicit(1, 1, 1, 1) // (top, bottom, left, right)
```

---

## 5. `DType::Bool` match arms need updating

`DType::Bool` now carries a `BoolStore` discriminator because different
backends store boolean tensors differently.

**Option A — handle each storage variant:**
```diff
 match tensor.dtype() {
-    DType::Bool => { ... }
+    DType::Bool(BoolStore::Native) => { ... }
+    DType::Bool(BoolStore::U8)     => { ... }
+    DType::Bool(BoolStore::U32)    => { ... }
     _ => unreachable!(),
 }
```

**Option B — wildcard (simpler, recommended for most cases):**
```diff
 match tensor.dtype() {
-    DType::Bool => { ... }
+    DType::Bool(_) => { ... }
     _ => unreachable!(),
 }
```

---

## 6. `Shape` API changes (bonus, may apply)

If the user accesses shape dims directly or catches shape errors:

```diff
-let b = tensor.shape().dims[0];
+let b = tensor.shape()[0];              // Index, not field access

-let swapped = shape.swap(1, 2);
+let swapped = shape.swapped(1, 2);     // Past-tense form

-let permuted = shape.permute(&[...]);
+let permuted = shape.permuted(&[...]); // Past-tense form

-ShapeError::RankMismatch{...}
+MetadataError::RankMismatch{...}
```

---

## 7. `powf` / `powf_scalar` removed for `Int` tensors (bonus)

These functions silently truncated results back to integers, so they were
removed. Cast explicitly:

```diff
-tensor_int.powf(tensor_float);
+tensor_int.float().powf(tensor_float);            // stay in float
+tensor_int.float().powf(tensor_float).int();      // cast back to int
```

---

## Migration checklist

1. Remove `<B: Backend>` from all `#[derive(Module)]` structs, `impl` blocks,
   and `Tensor<B, D>` type annotations.
2. Change `Gelu` bare type to `Gelu::new()` or `Gelu::default()`.
3. Replace `Ignored<SomeType>` with `#[module(skip)] pub field: SomeType`.
4. Double all `PaddingConfig::Explicit` arguments.
5. Change `DType::Bool` match arms to `DType::Bool(_)` or handle each variant.
6. Bump `burn` in `Cargo.toml` to `"0.21"`.
7. Run `cargo check` and then your test suite.
