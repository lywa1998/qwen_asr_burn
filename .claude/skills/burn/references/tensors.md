# Tensors

Burn's tensor API. Most of this looks like PyTorch, with key differences in ownership and the type system.

## Type system

```rust
Tensor<D>            // Float tensor (default), rank D, known at compile time
Tensor<D, Float>     // Same, explicit
Tensor<D, Int>       // Integer tensor
Tensor<D, Bool>      // Boolean tensor
```

`D` is the **rank** (number of dimensions), not the shape. Shape is dynamic. A 2-D tensor of shape `[5, 3]` and a 2-D tensor of shape `[10, 1]` are both `Tensor<2>`.

The element type for `Float`/`Int` is decided by the device's `DeviceConfig` (default `f32`/`i32`). Cast with `tensor.cast(dtype)` or build a device with `Device::wgpu(...).configure(DeviceConfig::default().float_dtype(FloatDType::F16))`.

## Initialization

```rust
use burn::prelude::*;

let device = Default::default();

// From literal data
let a = Tensor::<1>::from_floats([1., 2., 3.], &device);
let b = Tensor::<2>::from_data([[1., 2.], [3., 4.]], &device);

// From TensorData (backend-agnostic byte container)
let data = TensorData::from([1.0, 2.0, 3.0]);
let c = Tensor::<1>::from_data(data, &device);

// Int / Bool tensors
let ints = Tensor::<1, Int>::from_data([1, 2, 3], &device);
let mask = Tensor::<2, Bool>::from_data([[true, false], [false, true]], &device);

// Common factories
let zeros = Tensor::<2>::zeros([3, 4], &device);
let ones = Tensor::<2>::ones([3, 4], &device);
let full = Tensor::<2>::full([3, 4], 0.5, &device);
let rng = Tensor::<2>::random([3, 4], Distribution::Default, &device);
let eye = Tensor::<2>::eye(5, &device);
let range = Tensor::<1, Int>::arange(0..10, &device);
let like_a = a.zeros_like();
```

`TensorData::from` is the universal entrypoint when you have arbitrary numeric input. The struct exposes `as_slice`, `as_mut_slice`, `to_vec`, `iter` for inspecting bytes.

## Ownership and cloning

**Every tensor operation consumes its inputs by default.** This is intentional — fusion uses the reference count to know what can be reused in place.

```rust
// BREAKS: input is moved by .min()
let input = Tensor::<1>::from_floats([1., 2., 3., 4.], &device);
let min = input.min();
let max = input.max(); // ERROR: use of moved value

// WORKS: clone where you need to keep the original alive
let input = Tensor::<1>::from_floats([1., 2., 3., 4.], &device);
let min = input.clone().min();
let max = input.clone().max();
let normalized = (input - min.clone()).div(max - min);
```

Cloning is **cheap** — it bumps an atomic refcount on the underlying buffer, never copies the actual data. The fusion engine prefers code where you clone only as needed, because that lets it determine the last use of each buffer.

This means: when the borrow checker complains about a tensor, the fix is almost always `.clone()`, not refactoring.

There are no explicit in-place ops by design. If a tensor is used only once and the backend supports it, the framework will execute the op in place automatically.

## Reading data back (sync points)

These all **block on backend execution**:

```rust
let scalar = tensor.into_scalar();      // scalar tensor → host value
let data = tensor.to_data();            // by reference, tensor is reusable
let data = tensor.into_data();          // by value, tensor is consumed (more efficient if last use)
```

Inside a training loop, batch reads via `Transaction` to do **one** sync for many tensors:

```rust
use burn::tensor::Transaction;

let [output_data, loss_data, target_data] = Transaction::default()
    .register(output)
    .register(loss)
    .register(targets)
    .execute()
    .try_into()
    .expect("Three tensors registered");
```

`.to_device(&device)` is **also** a sync point on some backends. Try to keep tensors on a single device unless you have a reason.

## Operations cheatsheet

Reproduced/condensed from the Burn book. PyTorch equivalents on the right.

### Basic (all kinds)

| Burn | PyTorch |
| --- | --- |
| `Tensor::cat(tensors, dim)` | `torch.cat(tensors, dim)` |
| `Tensor::stack(tensors, dim)` | `torch.stack(tensors, dim)` |
| `tensor.reshape(shape)` | `tensor.view(shape)` |
| `tensor.flatten(start_dim, end_dim)` | `tensor.flatten(start, end)` |
| `tensor.squeeze::<D2>()` | `tensor.squeeze()` |
| `tensor.squeeze_dim::<D2>(dim)` | `tensor.squeeze(dim)` |
| `tensor.unsqueeze::<D2>()` | n/a |
| `tensor.unsqueeze_dim::<D2>(dim)` | `tensor.unsqueeze(dim)` |
| `tensor.permute(axes)` | `tensor.permute(axes)` |
| `tensor.swap_dims(d1, d2)` | `tensor.transpose(d1, d2)` |
| `tensor.transpose()` | `tensor.T` |
| `tensor.flip(axes)` | `tensor.flip(axes)` |
| `tensor.slice(slices)` | `tensor[(ranges,)]` |
| `tensor.slice_assign(slices, values)` | `tensor[(ranges,)] = values` |
| `tensor.narrow(dim, start, length)` | `tensor.narrow(dim, start, length)` |
| `tensor.chunk(n, dim)` | `tensor.chunk(n, dim)` |
| `tensor.split(size, dim)` | `tensor.split(size, dim)` |
| `tensor.gather(dim, indices)` | `torch.gather(t, dim, indices)` |
| `tensor.scatter(dim, indices, values, update)` | `tensor.scatter_add(...)` |
| `tensor.select(dim, indices)` | `tensor.index_select(dim, indices)` |
| `tensor.expand(shape)` | `tensor.expand(shape)` |
| `tensor.repeat(sizes)` | `tensor.repeat(sizes)` |
| `tensor.mask_fill(mask, value)` | `tensor.masked_fill(mask, value)` |
| `tensor.mask_where(mask, other)` | `torch.where(mask, other, t)` |
| `tensor.dims()` | `tensor.size()` |
| `tensor.shape()` | `tensor.shape` |
| `tensor.device()` | `tensor.device` |
| `tensor.to_device(&device)` | `tensor.to(device)` |

### Numeric (Float + Int)

| Burn | PyTorch |
| --- | --- |
| `a + b`, `a - b`, `a * b`, `a / b` | same |
| `a.add_scalar(s)`, `a + s` | `a + s` |
| `a.matmul(b)` | `a.matmul(b)` (Float only) |
| `tensor.sum()` / `.sum_dim(d)` | `tensor.sum()` / `.sum(d, keepdim=True)` |
| `tensor.mean()` / `.mean_dim(d)` | `tensor.mean()` / `.mean(d, keepdim=True)` |
| `tensor.max()`, `.min()` | same |
| `tensor.max_dim(d)`, `.min_dim(d)` | `keepdim=True` variants |
| `tensor.max_dim_with_indices(d)` | `tensor.max(d)` (returns values+indices) |
| `tensor.argmax(d)`, `.argmin(d)` | same |
| `tensor.topk(k, d)` / `.topk_with_indices(k, d)` | `tensor.topk(k, d)` |
| `tensor.sort(d)` / `.sort_with_indices(d)` | `tensor.sort(d)` |
| `tensor.clamp(min, max)` | `tensor.clamp(min, max)` |
| `tensor.greater(other)` / `.greater_elem(s)` | `>` |
| `tensor.lower(other)` / `.lower_elem(s)` | `<` |
| `tensor.equal(other)` / `.equal_elem(s)` | `==` |
| `tensor.cumsum(d)`, `.cumprod(d)`, `.cummax(d)`, `.cummin(d)` | same |
| `tensor.abs()`, `.neg()`, `.sign()` | same |
| `tensor.powf(other)`, `.powi(int_other)`, `.powf_scalar(s)` | `tensor.pow(...)` |
| `tensor.one_hot(num_classes)` | `F.one_hot` |

### Float only

| Burn | PyTorch |
| --- | --- |
| `tensor.exp()`, `.log()`, `.log1p()`, `.sqrt()`, `.square()`, `.recip()` | same |
| `tensor.sin()`, `.cos()`, `.tan()`, `.tanh()`, `.atan2(other)` | same |
| `tensor.var(d)`, `.var_mean(d)`, `.median(d)` | same |
| `tensor.is_nan()`, `.is_inf()`, `.is_finite()`, `.contains_nan()` | mostly same |
| `tensor.is_close(other, atol, rtol)`, `.all_close(other, atol, rtol)` | `torch.isclose`, `torch.allclose` |
| `tensor.cast(dtype)` | `tensor.to(dtype)` |
| `tensor.int()` | `tensor.long()` |

### Int only

| Burn | PyTorch |
| --- | --- |
| `Tensor::arange(start..end, &device)` | `torch.arange(start, end)` |
| `Tensor::arange_step(start..end, step, &device)` | `torch.arange(start, end, step)` |
| `tensor.float()` | `tensor.float()` |
| `tensor.bitwise_and(other)`, `.bitwise_or(...)`, `.bitwise_xor(...)`, `.bitwise_not()` | same |
| `tensor.bitwise_left_shift(...)`, `.bitwise_right_shift(...)` | same |

### Bool only

| Burn | PyTorch |
| --- | --- |
| `tensor.bool_and()`, `.bool_or()`, `.bool_not()`, `.bool_xor()` | `logical_and` etc. |
| `tensor.argwhere()`, `.nonzero()` | same |
| `Tensor::tril_mask(shape, diagonal)`, `::triu_mask(...)`, `::diag_mask(...)` | n/a |

## Activations and functional ops

Live under `burn::tensor::activation::*`:

```rust
use burn::tensor::activation::{relu, gelu, sigmoid, softmax, log_softmax};

let x = relu(x);
let probs = softmax(logits, /* dim */ 1);
```

Module-style versions (e.g. `nn::Relu`, `nn::Gelu`) wrap these and are zero-sized — useful when you want to put the activation in your model struct as a field.

## linalg, grid, signal modules

- `burn::tensor::linalg::*` — `cosine_similarity`, `vector_norm`, `outer`, `det`, `lu`, `trace`, `matvec`, etc.
- `burn::tensor::grid::*` — `meshgrid`, `affine_grid_2d` for spatial ops.
- `burn::tensor::signal::*` — `rfft`, `irfft`, `stft`, `istft`, window functions (`hann_window`, `hamming_window`, `blackman_window`). FFT size must be a power of two. The 0.21 release post calls FFT/IFFT support **"the first step toward supporting complex tensors in Burn"** — expect a complex-tensor type and richer DSP ops in subsequent releases.

## `Param<Tensor<D>>` — when to use what

In a module, three patterns:

| Wrapper | Behavior |
| ------- | -------- |
| `Param<Tensor<D>>` | Tracked parameter. Saved in `ModuleRecord`. Updated by optimizers. |
| `Param<Tensor<D>>` + `.set_require_grad(false)` | Saved in record but **not** updated. E.g. running statistics. |
| `Tensor<D>` | Plain field. Not in record, regenerated when the module is built. E.g. sinusoidal embeddings. |

```rust
use burn::module::Param;

#[derive(Module, Debug)]
struct LinearWithFrozenBias {
    weight: Param<Tensor<2>>,
    bias: Param<Tensor<1>>,         // tracked, updated by optimizer
    pos_enc: Tensor<2>,             // not tracked, recomputed each init
}
```

## Printing

```rust
println!("{}", tensor);       // default precision
println!("{:.2}", tensor);    // two decimals
```

For global settings:

```rust
use burn::tensor::{set_print_options, PrintOptions};

set_print_options(PrintOptions {
    precision: Some(2),
    threshold: 1000,   // max elements to show before summarizing
    edge_items: 3,     // how many items at start/end when summarizing
    ..Default::default()
});
```

For floating-point comparison (e.g. validating a port from PyTorch):

```rust
use burn::tensor::check_closeness;
check_closeness(&tensor_burn, &tensor_pytorch);
// prints a per-epsilon table with green/yellow/red coding
```
