# Modules

Burn's analog of `torch.nn.Module`. A parameter container with a couple of derived behaviors. Forward methods are your own.

## The `Module` derive

```rust
use burn::{nn, prelude::*};

#[derive(Module, Debug)]
pub struct Block {
    linear_in: nn::Linear,
    linear_out: nn::Linear,
    dropout: nn::Dropout,
    activation: nn::Gelu,
}

impl Block {
    pub fn forward<const D: usize>(&self, x: Tensor<D>) -> Tensor<D> {
        let x = self.linear_in.forward(x);
        let x = self.activation.forward(x);
        let x = self.dropout.forward(x);
        self.linear_out.forward(x)
    }
}
```

The derive generates the necessary trait impls so the struct can be saved/loaded, moved between devices, visited, mapped, etc. It assumes each field is itself a `Module`. Burn does **not** prescribe the shape of `forward` — name it whatever you want, return whatever shape you want, multiple forward methods are fine.

## Config + init pattern

Always pair a `Module` struct with a `Config` struct. The config holds hyperparameters; the model holds initialized weights. This is what makes hyperparameters serializable and lets you reload them when running inference.

```rust
#[derive(Config, Debug)]
pub struct BlockConfig {
    d_model: usize,
    d_ff: usize,
    #[config(default = 0.1)]
    dropout: f64,
}

impl BlockConfig {
    pub fn init(&self, device: &Device) -> Block {
        Block {
            linear_in: nn::LinearConfig::new(self.d_model, self.d_ff).init(device),
            linear_out: nn::LinearConfig::new(self.d_ff, self.d_model).init(device),
            dropout: nn::DropoutConfig::new(self.dropout).init(),
            activation: nn::Gelu::new(),
        }
    }
}
```

`#[derive(Config)]` gives you:

- A `BlockConfig::new(d_model, d_ff)` constructor for the required (non-default) fields.
- `with_dropout(0.2)` builder-style methods on every field.
- `config.save("path.json")?` and `BlockConfig::load("path.json")?`.

By convention, modules that take no learnable parameters (e.g. `Relu`, `Gelu`, `Tanh`) don't need a config — just `Relu::new()`.

## Built-in modules

Inside `burn::nn::*`:

| Category | Modules |
| -------- | ------- |
| Linear / norm / dropout | `Linear`, `LayerNorm`, `RmsNorm`, `BatchNorm`, `GroupNorm`, `InstanceNorm`, `LocalResponseNorm`, `Dropout`, `GaussianNoise` |
| Activations | `Relu`, `LeakyRelu`, `Prelu`, `Gelu`, `Glu`, `Sigmoid`, `Tanh`, `Softplus`, `Selu`, `Elu`, `Celu`, `HardSigmoid`, `HardSwish`, `HardShrink`, `SoftShrink`, `Softsign`, `Shrink`, `ThresholdedRelu` |
| Embedding | `Embedding` |
| Convolutions | `conv::{Conv1d, Conv2d, Conv3d, ConvTranspose1d, ConvTranspose2d, ConvTranspose3d, DeformConv2d}` |
| Pooling | `pool::{AvgPool1d, AvgPool2d, MaxPool1d, MaxPool2d, AdaptiveAvgPool1d, AdaptiveAvgPool2d}` |
| Interpolation | `Interpolate1d`, `Interpolate2d` with `InterpolateMode::{Nearest, Linear, Cubic, Lanczos}` |
| RNN | `Gru`, `BiGru`, `Lstm`, `BiLstm`, `GateController` |
| Attention / transformer | `MultiHeadAttention`, `TransformerEncoder`, `TransformerDecoder`, `PositionalEncoding`, `RotaryEncoding`, `SwiGlu` |
| Loss | `loss::{CrossEntropyLoss, BinaryCrossEntropyLoss, MseLoss, HuberLoss, KLDivLoss, SmoothL1Loss, CTCLoss, RNNTLoss, PoissonNllLoss, CosineEmbeddingLoss, GramMatrixLoss, LpLoss}` |

Each has a `*Config` builder and an `init`/`new` method. Conv/Linear configs take device because they allocate parameters; activations and pooling don't.

## Holding tensors directly

Most module fields are themselves modules. If you need to hold a tensor inline, three options:

```rust
use burn::module::Param;

#[derive(Module, Debug)]
struct CustomModule {
    weight: Param<Tensor<2>>,                              // tracked parameter
    running_var: Param<Tensor<1>>,                         // saved, not optimized — see below
    positional: Tensor<2>,                                 // not saved, constructed at init time
}
```

For `running_var`, set `.set_require_grad(false)` after construction so the optimizer skips it but it's still serialized. For `positional`, compute it in the config's `init` (e.g. a sinusoidal table) — it'll be deterministically rebuilt when loading.

A parameter's `ParamId` (`Param::id`) is its persistent key in `ModuleRecord`. Don't construct `Param`s by hand outside the macro unless you know what you're doing.

## Methods every module inherits

```rust
let n = model.num_params();                  // count of float params
let devices = model.devices();               // list of devices this module touches
let model = model.to_device(&device);        // move all params to a device
let model = model.fork(&device);             // move + detach autodiff history
let model = model.no_grad();                 // freeze all params (require_grad = false)
let model = model.valid();                   // strip autodiff, get inner-backend module
let record = model.into_record();            // serialize parameters
let model = model.load_record(record);       // apply a record
```

`fork` is useful for data parallelism — you copy the model to each device, run forward+backward there, then merge gradients.

## Visitors and mappers

`map` rewrites every parameter through a function; `visit` reads every parameter. The visitor/mapper traits cover all three tensor kinds (`float`, `int`, `bool`); only override what you need.

```rust
use burn::module::{Module, ModuleMapper, ParamId};

struct Clamp { min: f32, max: f32 }

impl ModuleMapper for Clamp {
    fn map_float<const D: usize>(&mut self, _id: ParamId, t: Tensor<D>) -> Tensor<D> {
        t.clamp(self.min, self.max)
    }
}

let model = model.map(&mut Clamp { min: -1.0, max: 1.0 });
```

If you want to apply a mapper **during training** without breaking autodiff:

```rust
fn map_float<const D: usize>(&mut self, _id: ParamId, t: Tensor<D>) -> Tensor<D> {
    let needs_grad = t.is_require_grad();
    let mut t = Tensor::from_inner(t.inner().clamp(self.min, self.max));
    if needs_grad { t = t.require_grad(); }
    t
}
```

Mappers are how optimizers update parameters under the hood. You probably won't write one for everyday work, but they're the right tool for: gradient clipping (use `burn::grad_clipping::*`), parameter freezing by name, EMA weight averaging, custom regularization.

## Custom display

By default a module prints its full structure. To customize:

```rust
use burn::module::{DisplaySettings, ModuleDisplay, Content};

#[derive(Module, Debug)]
#[module(custom_display)]
pub struct Block { ... }

impl ModuleDisplay for Block {
    fn custom_settings(&self) -> Option<DisplaySettings> {
        DisplaySettings::new()
            .with_show_all_attributes(false)
            .with_show_num_parameters(true)
            .optional()
    }

    fn custom_content(&self, content: Content) -> Option<Content> {
        content
            .add("linear_in", &self.linear_in)
            .add("linear_out", &self.linear_out)
            .optional()
    }
}
```

## Common module patterns

**Residual block** — just call `forward` twice and add:

```rust
let y = self.norm.forward(x.clone());
let y = self.attn.forward(y);
let x = x + y;
```

**Choose-your-loss in forward** — pull the loss config into a top-level method:

```rust
impl Model {
    pub fn forward_classification(&self, batch: MyBatch) -> ClassificationOutput {
        let output = self.forward(batch.inputs);
        let loss = nn::loss::CrossEntropyLossConfig::new()
            .init(&output.device())
            .forward(output.clone(), batch.targets.clone());
        ClassificationOutput::new(loss, output, batch.targets)
    }
}
```

The forward+loss method is what `TrainStep::step` and `InferenceStep::step` will both call.

**Picking the device inside a module** — `tensor.device()` gives you the device of any tensor you've received, so intermediate tensors get built on the right device automatically. Avoid hardcoding `Default::default()` inside forward methods.

## Anti-patterns

- Defining `forward` to take `&Tensor<D>` instead of `Tensor<D>`. Burn ops consume tensors. Pass-by-value and clone inside if needed.
- Returning `Result<Tensor<D>, _>` from a forward method. Burn ops don't return `Result`. Validate shapes at module boundaries, then trust them.
- Using `Param<Tensor<D>>` for everything, including constants. Plain `Tensor<D>` fields are smaller in the saved record.
- Reaching into `param.val()` and mutating. Use mappers / visitors so it composes with autodiff.
