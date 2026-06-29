// Corrected model code for Burn 0.21
//
// Breaking changes addressed:
//   1. Removed <B: Backend> generic — Tensor / Module no longer parameterized by backend
//   2. Gelu → Gelu::new()   (Gelu is now a constructed type)
//   3. Ignored<PaddingConfig1d> → #[module(skip)]
//   4. PaddingConfig1d::Explicit(1) → Explicit(1, 1)
//   5. DType::Bool → DType::Bool(_)  (Bool now carries a storage discriminator)
//
// Note: The "no backend generic" change is the foundation of burn-dispatch.
// Device construction moves from type parameters to runtime Device values
// (e.g. Device::wgpu(DeviceKind::DefaultDevice), Device::cuda(0), Device::flex()).

use burn::{
    nn::{conv::Conv2d, Linear, Gelu},
    prelude::*,
};

// --- BEFORE (0.19/0.20 — does NOT compile on 0.21) ---
//
// #[derive(Module, Debug)]
// pub struct Model<B: Backend> {
//     conv1: Conv2d<B>,
//     conv2: Conv2d<B>,
//     linear: Linear<B>,
//     activation: Gelu,                          // bare type used as value
// }
//
// impl<B: Backend> Model<B> {
//     pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 2> {
//         let x = self.activation.forward(x);
//         let x = self.conv1.forward(x);
//         let x = self.conv2.forward(x);
//         let x = x.flatten(1, 3);
//         self.linear.forward(x)
//     }
// }

// --- AFTER (0.21 — compiles) ---

#[derive(Module, Debug)]
pub struct Model {
    conv1: Conv2d,
    conv2: Conv2d,
    linear: Linear,
    activation: Gelu,
}

impl Model {
    pub fn forward(&self, x: Tensor<3>) -> Tensor<2> {
        let x = self.activation.forward(x);
        let x = self.conv1.forward(x);
        let x = self.conv2.forward(x);
        let x = x.flatten(1, 3);
        self.linear.forward(x)
    }
}

// --- Fix 2: Gelu is now constructed ---
// When creating the model, use Gelu::new() or Gelu::default():
//
// Model {
//     conv1: Conv2dConfig::new([1, 16], [3, 3]).init(device),
//     conv2: Conv2dConfig::new([16, 32], [3, 3]).init(device),
//     linear: LinearConfig::new(32 * 5 * 5, 10).init(device),
//     activation: Gelu::new(),     // <-- was bare Gelu, now calls ::new()
// }

// --- Fix 3: Ignored<T> → #[module(skip)] ---

// BEFORE (0.19/0.20):
// #[derive(Module, Debug)]
// pub struct ConvBlock<B: Backend> {
//     padding: Ignored<PaddingConfig1d>,
//     conv: Conv1d<B>,
// }

#[derive(Module, Debug)]
pub struct ConvBlock {
    #[module(skip)]
    pub padding: PaddingConfig1d,
    pub conv: Conv1d,
}

// --- Fix 4: PaddingConfig::Explicit takes all sides ---

// BEFORE:
// let pad = PaddingConfig1d::Explicit(1);
// let pad = PaddingConfig2d::Explicit(1, 1);

// AFTER:
pub fn demo_padding() {
    let _pad_1d = PaddingConfig1d::Explicit(1, 1);       // (left, right)
    let _pad_2d = PaddingConfig2d::Explicit(1, 1, 1, 1); // (top, bottom, left, right)
}

// --- Fix 5: DType::Bool carries a storage discriminator ---

pub fn validate_dtype(tensor: &Tensor<2, Bool>) {
    // BEFORE:
    // match tensor.dtype() {
    //     DType::Bool => { /* handle bool tensor */ }
    //     _ => unreachable!(),
    // }

    // AFTER — wildcard pattern (simplest, recommended):
    match tensor.dtype() {
        DType::Bool(_) => {
            // handle bool tensor, regardless of storage backend
        }
        _ => unreachable!(),
    }

    // Or handle each storage variant explicitly:
    match tensor.dtype() {
        DType::Bool(BoolStore::Native) => { /* bit-packed */ }
        DType::Bool(BoolStore::U8)     => { /* byte-per-element */ }
        DType::Bool(BoolStore::U32)    => { /* u32-per-element */ }
        _ => unreachable!(),
    }
}
