# Performance

The Burn execution model is async + JIT-compiled. Most performance issues come from breaking that model — forcing premature syncs, or feeding the JIT shapes it can't fuse well. This file is a tour of what to watch for.

## What 0.21 changed

If a user is comparing 0.20 vs 0.21 numbers, here's what to expect (drawn from the 0.21 release post):

- **Framework launch overhead dropped massively.** Average eager speedup ~3.4×, average fusion speedup ~5.4×, best 8.2×, no regressions. The headline is small-tensor workloads where Burn was previously bottlenecked by a recursive mutex on device handles. 0.21 replaces it with a custom "lazy fire-and-forget" channel; multiple services share a thread and fusion pipelines with the CubeCL runtime.
- **Distributed (4×CUDA) primitives got 6–21× faster.** `to_device` 16–21×, `all_reduce` ~6×. Built around new differentiable collective operations.
- **GEMV is now competitive with LibTorch on column-major.** Shape `(1,1,4096) × (4096,4096)`: CubeCL CUDA 174 µs vs LibTorch 171 µs. Row-major is intentionally not yet optimized; column-major was prioritized first.
- **Top-k is dramatically faster on inner axes.** Up to 41× over LibTorch on `axis=0` cases. `axis=last` with `k=10` is the one place LibTorch still wins (~1.4×) — they ship a specialized kernel; Burn uses a general scheme. Roughly 30% recovery is on the roadmap.
- **Autotune got smarter.** Better kernel grouping, more reliable micro-benchmarks (proper scoring instead of taking the median). Theoretical throughput thresholds for short-circuit autotuning are on the roadmap.
- **Kernel validation layer.** Opt-in via `burn.toml`. Already caught real OOB memory accesses. Useful for custom-kernel work.

If a user upgrades and sees a regression, it's worth filing — the post explicitly claims zero regressions on the bench grid.

## How execution actually works

Most Burn backends (CubeCL: cuda, rocm, wgpu, vulkan, metal, cpu) execute lazily. When you write `let z = x.matmul(y)`, no kernel runs immediately. Instead Burn appends ops to a per-thread queue. Several things trigger an actual sync to the device:

- `tensor.into_scalar()` — must read a value back to host
- `tensor.into_data()` / `tensor.to_data()` — must materialize
- `tensor.sync()` (explicit)
- `tensor.to_device(&other)` — sometimes (depends on backend)

A sync **flushes the queue and waits**. Inside a training loop, it's the single biggest performance footgun. Avoid them in inner loops; batch them when unavoidable.

## Rule 1: don't sync inside hot loops

```rust
// BAD — syncs on every batch to log a Python-style scalar
for batch in dataloader.iter() {
    let loss = model.forward(batch);
    println!("loss = {}", loss.into_scalar());   // <-- sync
    let grads = loss.backward();
    // ...
}
```

```rust
// BETTER — sync at fixed intervals, or accumulate and sync once per epoch
for (i, batch) in dataloader.iter().enumerate() {
    let loss = model.forward(batch);
    let grads = loss.clone().backward();
    if i % 100 == 0 {
        println!("loss = {}", loss.into_scalar());   // sync every 100 steps
    }
    // ...
}
```

When you do need multiple values back at once, batch them with a `Transaction` so it's **one** sync:

```rust
use burn::tensor::Transaction;

let [output, loss, targets] = Transaction::default()
    .register(output)
    .register(loss)
    .register(targets)
    .execute()
    .try_into()
    .expect("Three tensors registered");
```

This is what `burn-train` does internally to compute metrics.

## Rule 2: shape multiples of powers of 2 are dramatically faster

The autotune system picks the best kernel for each (op, dtype, shape) tuple at first encounter, then caches the result on disk. Shapes like `[1024, 1024]` autotune to vectorized kernels with no bounds checks. Shapes like `[1000, 1000]` lose vectorization and fall back to slower variants.

When you can't choose your shape, prefer to do the awkward shape transformation in **one** kernel, then run all subsequent ops on a clean shape:

```rust
// If you must compute on [1000, 1000]:
let x = unfortunate_input;             // [1000, 1000]
let x = x.pad([(0, 24), (0, 24)], 0.); // [1024, 1024]
// ... rest of the network on power-of-two shapes ...
```

## Rule 3: respect the fusion engine

Kernel fusion turns chains of element-wise operations into a single GPU kernel, dramatically reducing memory traffic. The simpler heuristics:

- **Don't keep tensors alive longer than needed.** A live clone forces fusion to commit (write to global memory) earlier. Drop variables you won't reuse.
- **Group view ops together.** `reshape`, `permute`, `swap_dims`, `transpose`, `unsqueeze`, `squeeze`, `slice`, `select`, `gather`, `scatter` interfere with vectorization because they alter access patterns.

```rust
// HARDER TO FUSE — view ops interleaved with compute
let z = a.unsqueeze().matmul(b) + c.unsqueeze();

// EASIER TO FUSE — view ops grouped first
let a = a.unsqueeze();
let c = c.unsqueeze();
let z = a.matmul(b) + c;
```

- **Don't gratuitously clone.** Cloning a tensor is cheap (atomic refcount), but it tells the engine "this buffer must remain valid past this op." The engine will then materialize the buffer rather than fusing it. Clone only when you actually need the value alive past the next consumer.

For deeper details, see `burn-book/src/performance/good-practices/kernel-fusion.md`.

## Rule 4: separate augmentation device from training device

Dataloading and augmentation in Burn are async too. If you do augmentation on the same device as training, it competes with training kernels for the GPU.

```rust
let train_device = Device::cuda(0);
let augment_device = Device::cpu();

// In your batcher: build the batch on `augment_device`, then move once
fn batch(&self, items: Vec<Item>, _device: &Device) -> Batch {
    let batch_cpu = build_on_cpu(items);
    Tensor::from_data(batch_cpu.into_data(), &train_device)
}
```

Then concatenate **on the augmentation device** (CPU), not piecewise on the training device. Many small allocations + a final cat is slower than one CPU-side batch + one transfer.

## Rule 5: the autotune cache is a deployment asset

First run on a new machine + new shape pays a one-time autotune cost. Subsequent runs read from a cache.

For deployment / spot instances / CI, configure the cache via `burn.toml` (0.21+):

```toml
[cubecl.autotune]
level = "balanced"
cache = { file = "autotune.json" }
```

Then bundle `autotune.json` with the binary. The pre-0.21 `CUBECL_CONFIG` env var still works as a fallback. See `references/burn-toml.md` for the full set of knobs (level, streaming, persistent memory, logging).

## Cargo features that affect performance

Always-on for performance:

```toml
burn = { version = "0.21", features = [
    "wgpu",         # or "cuda" / "metal" / etc.
    "fusion",       # kernel fusion — significant speedup
    "autotune",     # kernel selection
    "train",
] }
```

Other useful features:

| Feature | Effect |
| ------- | ------ |
| `simd` | SIMD codegen for the Flex/ndarray backends |
| `apple-amx` | Apple AMX coprocessor on M-series via Flex |
| `x86-v4` | AVX-512 codegen on x86 |
| `accelerate` / `openblas` / `blas-netlib` | BLAS for ndarray backend |
| `cubecl` | Re-export CubeCL for writing custom kernels |

`autotune-checks` is for testing autotune correctness; off in release.

## Mixed precision

Two angles:

1. **Storage precision** — save F16, load as F32. See `references/saving-loading.md` (HalfPrecisionAdapter).
2. **Compute precision** — cast tensors to F16 for compute, keep optimizer state in F32. Use `tensor.cast(DType::F16)` and `model.cast(DType::F16)`. Keep loss in F32 to avoid underflow.

For a fully F16 forward + F32 loss + F32 gradients setup:

```rust
let model_f16 = model.clone().cast(DType::F16);
let output_f16 = model_f16.forward(input_f16);
let output = output_f16.cast(DType::F32);
let loss = compute_loss(output, target);  // F32 loss
let grads = loss.backward();              // gradients in F32 against F32 model
```

(There isn't a single one-line "AMP" toggle — you wire it explicitly.)

## Quantization

Burn supports post-training quantization (PTQ) and quantization-aware training (QAT) for backends that implement quantization strategies (currently CubeCL and LibTorch). See `burn-book/src/performance/quantization.md`.

```rust
let scheme = QuantizationScheme::PerTensor(QuantizationMode::Symmetric, QuantizationType::QInt8);
let qparams = compute_qparams(...);
let q_tensor = tensor.quantize(scheme, qparams);
let dq_tensor = q_tensor.dequantize();
```

## Profiling

CubeCL ships with tracing support — enable the `tracing` feature and instrument with the standard `tracing` crate to see kernel boundaries. For deeper profiling:

- **CUDA:** Nsight Systems / Nsight Compute work directly on Burn binaries
- **WGPU:** RenderDoc with the Vulkan backend
- **Metal:** Xcode's Metal Frame Capture works on the Metal backend

For app-level timing, use `std::time::Instant` around full epoch loops and **do** force a sync at the boundary:

```rust
let start = Instant::now();
for batch in dataloader.iter() { ... }
device.sync()?;                              // make sure work has actually completed
let elapsed = start.elapsed();
```

Without the sync, the timer measures how long it took to enqueue ops, not how long the GPU actually spent.

## When you do need to write a custom kernel

For compute-bound operations where Burn's fusion and autotune still leave performance on the table, write a kernel in CubeCL and call it from a backend extension. See `burn-book/src/advanced/backend-extension/` and the `examples/custom-cubecl-kernel/` and `examples/custom-wgpu-kernel/` examples.

This is rarely necessary for application code — usually it's the right move when integrating a domain-specific operator (e.g. a fused triton-style block of attention). For everyday model work, prefer composing existing operators.

## Anti-patterns

- Calling `.into_scalar()` to log loss every step. Sync per step destroys throughput.
- Cloning tensors "just in case" inside forward methods. Each clone is a hint to the fusion engine that this buffer must be materialized.
- Using `f32` for everything when the model is fine in `f16`. Memory bandwidth is usually the bottleneck.
- Resizing batches mid-training. Each new shape pays an autotune cost. Pick a batch size and stick with it.
- Allocating intermediate tensors per item in the batcher rather than batching first then converting.
- Ignoring the fusion ordering rule. View ops interleaved with compute can cut throughput by 2-5x.
