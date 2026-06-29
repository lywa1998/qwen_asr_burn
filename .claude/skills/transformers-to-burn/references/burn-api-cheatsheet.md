# Burn API Cheatsheet

Quick reference for common Burn operations used in model implementations.

## Tensor Creation

```rust
use burn::tensor::{Tensor, Int, Bool, Device};

// From Vec<f32>
Tensor::<1>::from_floats(floats.as_slice(), &device)

// From Vec<i32> (Int type)
Tensor::<1, Int>::from_ints(ints.as_slice(), &device)

// From TensorData (generic, for Bool etc.)
let data: TensorData = values.as_slice().into();
Tensor::<1, Bool>::from_bool(data, &device)

// Constants
Tensor::ones([size], &device)
Tensor::zeros([size], &device)

// 2D Int helper (used for token ID inputs)
fn int_tensor_2d(ids: &[u32], device: &Device) -> Tensor<2, Int> {
    let ints: Vec<i32> = ids.iter().map(|&id| id as i32).collect();
    Tensor::<1, Int>::from_ints(ints.as_slice(), device).unsqueeze_dim::<2>(0)
}
```

## Shape Manipulation

```rust
// Reshape (panics if sizes don't match)
x.reshape([batch, seq_len, hidden])

// Swap dimensions (transpose)
x.swap_dims(1, 2)   // swap dim 1 <-> dim 2

// Unsqueeze (add dimension)
x.unsqueeze_dim::<4>(1)  // <TARGET_RANK>(position)

// Narrow (take a slice by offset + length)
x.narrow(1, start, length)   // along dim 1

// Slice (range-based)
x.slice([0..batch, start..end, 0..hidden])

// Expand dimensions
x.expand([batch, num_heads, seq_len, head_dim])

// Repeat along a dimension
x.repeat_dim(2, n_rep)   // repeat dim 2 n_rep times

// Concatenate
Tensor::cat(vec![a, b, c], 1)  // along dim 1

// Get dimension sizes
let [batch, seq_len, hidden] = x.dims();  // returns [usize; N]
let dims = x.dims();
```

## Linear Layers

```rust
use burn::nn::{Linear, LinearConfig, Embedding, EmbeddingConfig};

// Linear
LinearConfig::new(in_features, out_features).init(device)                    // with bias
LinearConfig::new(in_features, out_features).with_bias(false).init(device)   // no bias

// Embedding
EmbeddingConfig::new(vocab_size, hidden_size).init(device)

// Conv2d
use burn::nn::conv::{Conv2d, Conv2dConfig};
use burn::nn::PaddingConfig2d;

Conv2dConfig::new([in_ch, out_ch], [kernel_h, kernel_w])
    .with_stride([2, 2])
    .with_padding(PaddingConfig2d::Explicit(pad_top, pad_bottom, pad_left, pad_right))
    .with_bias(true)
    .init(device)
```

## Activations

```rust
use burn::tensor::activation::{softmax, silu, gelu, sigmoid, relu};

softmax(x, 3)    // along dim 3
silu(x)          // SiLU / Swish
gelu(x)          // GELU
sigmoid(x)       // Sigmoid
relu(x)          // ReLU
```

## Math Operations

```rust
// Basic arithmetic
x.add(y)          // x + y
x.mul(y)          // x * y
x.div(y)          // x / y
x.sub(y)          // x - y
x.neg()           // -x

// Scalar operations
x.add_scalar(v)   // x + v
x.mul_scalar(v)   // x * v
x.div_scalar(v)   // x / v
x.powf_scalar(v)  // x^v

// Reduction
x.mean_dim(2)     // mean along dim 2
x.sum()           // sum all
x.sum_dim(1)      // sum along dim 1
x.argmax(1)       // argmax along dim 1

// Advanced math
x.sqrt()          // sqrt
x.exp()           // e^x
x.cos()           // cos
x.sin()           // sin
x.abs()           // |x|

// Matrix ops
a.matmul(b)       // matrix multiply
```

## Mask & Index

```rust
// Bool mask fill
// mask is Tensor<4, Bool>, true positions get filled with value
attn_weights.mask_fill(mask, f32::NEG_INFINITY)

// Mask where (set positions matching condition to value)
// mask_fill equivalent for additive masks
let mask = causal_float_mask(seq_len, &device);
let attn_weights = softmax(attn_weights + mask, 3);
```

## Causal Mask Creation

```rust
// Bool-style mask (for mask_fill)
pub fn create_causal_mask(seq_len: usize, past_len: usize, device: &Device) -> Tensor<4, Bool> {
    let total_len = past_len + seq_len;
    let mut values = Vec::with_capacity(seq_len * total_len);
    for row in 0..seq_len {
        let current_pos = past_len + row;
        for col in 0..total_len {
            values.push(col > current_pos);
        }
    }
    let data: TensorData = values.as_slice().into();
    Tensor::<1, Bool>::from_bool(data, device)
        .reshape([seq_len, total_len])
        .unsqueeze_dim::<3>(0)
        .unsqueeze_dim::<4>(0)
}

// Float-style mask (for additive masking)
fn causal_mask(seq_len: usize, device: Device) -> Tensor<4> {
    let mut data = Vec::with_capacity(seq_len * seq_len);
    for i in 0..seq_len {
        for j in 0..seq_len {
            data.push(if j > i { f32::NEG_INFINITY } else { 0.0f32 });
        }
    }
    Tensor::<1>::from_floats(data.as_slice(), &device)
        .reshape([1, 1, seq_len, seq_len])
}
```

## Data Extraction

```rust
// Get raw data back from tensor
let data = tensor.into_data();     // TensorData
let vals: &[f32] = data.as_slice().unwrap();  // F32 only

// Extract scalar from argmax
let token_data = logits.argmax(1).reshape([1]).into_data();
let token_id = i32::from_le_bytes([
    token_data.bytes[0], token_data.bytes[1],
    token_data.bytes[2], token_data.bytes[3],
]) as u32;
```

## Module Patterns

```rust
use burn::module::{Module, Param};
use burn::config::Config;

// Module with trainable params
#[derive(Module, Debug)]
pub struct MyLayer {
    pub linear: nn::Linear,
    pub norm: MyNorm,
    #[module(skip)]  // scalar field, not a parameter
    num_heads: usize,
}

// Config with init
#[derive(Config, Debug)]
pub struct MyLayerConfig {
    hidden_size: usize,
    num_heads: usize,
}

impl MyLayerConfig {
    pub fn init(&self, device: &burn::tensor::Device) -> MyLayer {
        // ... create sub-layers
    }
}

// Param wrapper (for norm weights)
#[derive(Module, Debug)]
pub struct MyNorm {
    pub weight: Param<Tensor<1>>,
    #[module(skip)]
    epsilon: f64,
}
```

## Feature Gating

```rust
// Metal vs CUDA specific code
#[cfg(feature = "metal")]
let adapter = ChainAdapter::new(PyTorchToBurnAdapter, Bf16ToF32Adapter);
#[cfg(not(feature = "metal"))]
let adapter = PyTorchToBurnAdapter;
```

## Device

```rust
// Get device from tensor
let device = tensor.device();

// Create device directly
#[cfg(feature = "cuda")]
let device: Device = Device::cuda(0);
#[cfg(feature = "metal")]
let device: Device = Device::metal(DeviceKind::DefaultDevice);
```
