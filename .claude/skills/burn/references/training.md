# Training

Two ways to train in Burn: the **Learner / SupervisedTraining** path (high-level, gives you a checkpointer + TUI dashboard + metrics) and a **manual loop**. Use the high-level one unless you have a reason.

## High-level: `SupervisedTraining` + `Learner`

The full pipeline:

1. Define your model and a `*Config` for it.
2. Define a `Batcher` that converts a `Vec<Item>` into a struct of tensors.
3. Define `forward_classification` (or `_regression`) on the model that returns a `ClassificationOutput` / `RegressionOutput`.
4. `impl TrainStep for Model` and `impl InferenceStep for Model`.
5. Build dataloaders.
6. Build a `SupervisedTraining` instance with metrics, checkpointer, num_epochs.
7. `training.launch(Learner::new(model, optimizer, lr_or_scheduler))`.

```rust
use burn::{
    data::{dataloader::DataLoaderBuilder, dataset::vision::MnistDataset},
    nn::loss::CrossEntropyLossConfig,
    optim::AdamConfig,
    prelude::*,
    train::{
        ClassificationOutput, InferenceStep, Learner, SupervisedTraining,
        TrainOutput, TrainStep,
        metric::{AccuracyMetric, LossMetric},
    },
};

#[derive(Config, Debug)]
pub struct TrainingConfig {
    pub model: ModelConfig,
    pub optimizer: AdamConfig,
    #[config(default = 10)]   pub num_epochs: usize,
    #[config(default = 64)]   pub batch_size: usize,
    #[config(default = 4)]    pub num_workers: usize,
    #[config(default = 42)]   pub seed: u64,
    #[config(default = 1e-4)] pub learning_rate: f64,
}

impl Model {
    pub fn forward_classification(&self, batch: MnistBatch) -> ClassificationOutput {
        let output = self.forward(batch.images);
        let loss = CrossEntropyLossConfig::new()
            .init(&output.device())
            .forward(output.clone(), batch.targets.clone());
        ClassificationOutput::new(loss, output, batch.targets)
    }
}

impl TrainStep for Model {
    type Input = MnistBatch;
    type Output = ClassificationOutput;

    fn step(&self, batch: MnistBatch) -> TrainOutput<ClassificationOutput> {
        let item = self.forward_classification(batch);
        TrainOutput::new(self, item.loss.backward(), item)
    }
}

impl InferenceStep for Model {
    type Input = MnistBatch;
    type Output = ClassificationOutput;

    fn step(&self, batch: MnistBatch) -> ClassificationOutput {
        self.forward_classification(batch)
    }
}

pub fn train(artifact_dir: &str, config: TrainingConfig, device: Device) {
    std::fs::remove_dir_all(artifact_dir).ok();
    std::fs::create_dir_all(artifact_dir).ok();
    config.save(format!("{artifact_dir}/config.json")).unwrap();

    device.seed(config.seed);
    let autodiff_device = device.clone().autodiff();

    let dataloader_train = DataLoaderBuilder::new(MnistBatcher::default())
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(MnistDataset::train());

    let dataloader_test = DataLoaderBuilder::new(MnistBatcher::default())
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(MnistDataset::test());

    let training = SupervisedTraining::new(artifact_dir, dataloader_train, dataloader_test)
        .metrics((AccuracyMetric::new(), LossMetric::new()))
        .with_checkpointer()
        .num_epochs(config.num_epochs)
        .summary();

    let model = config.model.init(&autodiff_device);
    let result = training.launch(Learner::new(
        model,
        config.optimizer.init(),
        config.learning_rate,
    ));

    result.model
        .into_record()
        .save(format!("{artifact_dir}/model"))
        .unwrap();
}
```

That's the whole training program. Run it and you get a TUI dashboard with progress bars and metric plots (when the `tui` feature is enabled).

## Important details about the high-level path

- **`device.clone().autodiff()`** is what enables gradient tracking. Pass the autodiff-tracked device to `model.init`. Inference paths use the plain device (or `device.clone().inner()` from an autodiff one).
- **`device.seed(config.seed)`** seeds RNG at the device level. There's no longer a `B::seed` free function.
- **`with_checkpointer()`** writes `.bpk` files for the model, optimizer, and LR scheduler each epoch. You can resume from a checkpoint with `.with_checkpoint(epoch)`.
- **The third arg to `Learner::new`** is a learning rate scheduler. A bare `f64` is auto-wrapped as a constant scheduler. Use real schedulers from `burn::lr_scheduler::*` for cosine annealing, warmup, etc.
- **`output.device()`** is the right way to get a device inside a model method — don't pass it as an extra arg.

## Output structs

`burn::train::ClassificationOutput` and `RegressionOutput` are the standard output containers. They implement traits that let metrics like `AccuracyMetric` / `LossMetric` read them automatically. If you need a custom output (multi-task, multi-loss), define your own struct and the metrics that consume it — see `crates/burn-train/src/metric/`.

## Metrics

```rust
.metrics((AccuracyMetric::new(), LossMetric::new()))
.metric_train_numeric(LearningRateMetric::new())     // training-only, plottable
.metric_valid(MyCustomMetric::new())                 // validation-only
```

Available out of the box:
- `AccuracyMetric`, `TopKAccuracyMetric`
- `LossMetric`
- `LearningRateMetric`
- `AucPrMetric`
- `CudaMetric`, `CpuMemoryUsageMetric`, `GpuMemoryUsageMetric` (with `metrics` feature)

## Learning rate schedulers

In `burn::lr_scheduler::*`:

```rust
use burn::lr_scheduler::{
    composed::ComposedLrSchedulerConfig,
    cosine::CosineAnnealingLrSchedulerConfig,
    linear::LinearLrSchedulerConfig,
    step::StepLrSchedulerConfig,
};

let scheduler = ComposedLrSchedulerConfig::new()
    .linear(LinearLrSchedulerConfig::new(1e-8, 1.0, 2_000))   // warmup
    .cosine(CosineAnnealingLrSchedulerConfig::new(1.0, 50_000));
```

The composed scheduler runs sub-schedulers in sequence by step count. Each step is one batch (not one epoch).

## Manual training loop

When you need full control (GANs, RL, weird optimization regimes, multiple optimizers):

```rust
use burn::optim::{AdamConfig, GradientsParams, Optimizer};

pub fn run(device: Device) {
    let autodiff_device = device.autodiff();
    let mut model = ModelConfig::new(10).init(&autodiff_device);
    let mut optim = AdamConfig::new().init();
    let lr = 1e-4;

    for epoch in 1..=num_epochs {
        for batch in dataloader_train.iter() {
            let output = model.forward(batch.images);
            let loss = nn::loss::CrossEntropyLoss::new(None, &output.device())
                .forward(output, batch.targets);

            let grads = loss.backward();
            let grads = GradientsParams::from_grads(grads, &model);
            model = optim.step(lr, model, grads);
        }

        // validation pass on the no-autodiff inner version
        let model_valid = model.valid();
        for batch in dataloader_test.iter() {
            let output = model_valid.forward(batch.images);
            // compute metrics ...
        }
    }
}
```

Things to know:

- **`loss.backward()`** returns gradients **for that backward pass**. There are no "global" gradients sitting on tensors.
- **`GradientsParams::from_grads(grads, &model)`** maps the computed gradients to the model's parameters by ID. This is what makes multi-optimizer setups possible.
- **`optim.step(lr, model, grads)`** consumes `model` and returns a new one. Reassign with `model = optim.step(...)`. There's no `zero_grad` — `step` consumes the gradients.
- **`model.valid()`** returns a non-autodiff view of the model for validation. Required if you trained on an autodiff backend; without `.valid()` you'd get a compile error trying to use autodiff-only methods on the validation batcher's inner-backend tensors.

## Gradient accumulation

```rust
use burn::optim::GradientsAccumulator;

let mut accum = GradientsAccumulator::new();
for batch in batches.into_iter().take(accum_steps) {
    let loss = ...;
    let grads = loss.backward();
    let grads = GradientsParams::from_grads(grads, &model);
    accum.accumulate(&model, grads);
}
let grads = accum.grads();
model = optim.step(lr, model, grads);
```

## Multiple optimizers / per-layer learning rates

```rust
let grads = loss.backward();

let grads_conv = GradientsParams::from_module(&mut grads, &model.conv1);
let grads_lin = GradientsParams::from_module(&mut grads, &model.linear);

model = optim.step(lr * 2.0, model, grads_conv);
model = optim.step(lr * 0.5, model, grads_lin);

// remaining params (whatever's left in `grads`)
let rest = GradientsParams::from_grads(grads, &model);
model = optim.step(lr, model, rest);
```

You can use **different optimizer types** for each split — `AdamW` for the encoder, `SGD` for the head, etc. Each `step` call is independent.

## Optimizers available

`burn::optim::*`:

| Optimizer | Config |
| --------- | ------ |
| Adam | `AdamConfig::new()` |
| AdamW | `AdamWConfig::new()` |
| SGD | `SgdConfig::new()` |
| RmsProp | `RmsPropConfig::new()` |
| Adagrad | `AdagradConfig::new()` |
| Lion | `LionConfig::new()` (when available) |

All configs have `with_*` builders for hyperparameters (`weight_decay`, `momentum`, betas, etc.).

## Gradient clipping

```rust
use burn::grad_clipping::{GradientClipping, GradientClippingConfig};

let optim = AdamConfig::new()
    .with_grad_clipping(Some(GradientClippingConfig::Norm(1.0)))
    .init();
```

Choices: `Value(threshold)` (per-element), `Norm(max_norm)` (per-tensor norm), `GlobalNorm(max_norm)` (single global norm across all params).

## Custom training strategy

`SupervisedTraining` accepts a custom strategy if the supervised pattern doesn't fit but you still want the dashboard, checkpointing, and metrics:

```rust
use burn::train::CustomTrainingStrategy;

let training = SupervisedTraining::new(...)
    .strategy(MyCustomStrategy::new());
```

Look at `examples/custom-learning-strategy/` for a concrete example.

## Resuming

```rust
let training = SupervisedTraining::new(artifact_dir, dataloader_train, dataloader_test)
    .with_checkpointer()
    .with_checkpoint(5)        // resume from epoch 5
    .num_epochs(20);
```

The checkpointer reads `model-{epoch}.bpk`, `optim-{epoch}.bpk`, `scheduler-{epoch}.bpk` from the artifact directory.

## Reinforcement learning: `RLTraining` + off-policy strategy

Burn 0.21 added a sibling to `SupervisedTraining` for RL workloads: `RLTraining` (in `burn::train`) plus an off-policy strategy (`RLStrategies::OffPolicyStrategy`) that handles replay buffers, exploration schedules, and environment stepping. The vocabulary is "agent / environment / episode" rather than "model / dataloader / epoch", but the structural ideas (configurable, metric-driven, checkpointer-friendly) carry over.

This is the minimal DQN setup, mirrored from the 0.21 release post:

```rust
use burn::{
    optim::{AdamWConfig, GradientClippingConfig},
    prelude::*,
    train::{
        RLStrategies, RLTraining,
        metric::{
            CumulativeRewardMetric, EpisodeLengthMetric,
            ExplorationRateMetric, LossMetric,
        },
        record::CompactRecorder,
    },
};
use burn_rl::{
    dqn::{DqnAgentConfig, DqnLearningAgent},
    OffPolicyConfig,
    network::{MlpNet, MlpNetConfig},
};

const ARTIFACT_DIR: &str = "/tmp/cartpole-dqn";

pub fn run(device: Device) {
    // Network
    let net_config = MlpNetConfig {
        num_layers: 3,
        dropout: 0.0,
        d_input: 4,        // CartPole obs
        d_output: 2,       // CartPole actions
        d_hidden: 64,
    };
    let policy_model = MlpNet::new(&net_config, &device);

    // Optimizer
    let optimizer = AdamWConfig::new()
        .with_grad_clipping(Some(GradientClippingConfig::Value(100.0)))
        .init();

    // DQN agent hyperparameters
    let dqn_config = DqnAgentConfig {
        gamma: 0.99,
        learning_rate: 1e-3,
        tau: 0.005,
        epsilon_start: 1.0,
        epsilon_end: 0.05,
        epsilon_decay: 50_000,
    };
    let agent = DqnLearningAgent::new(policy_model, optimizer, dqn_config);

    // Off-policy training config
    let learning_config = OffPolicyConfig {
        num_envs: 4,
        autobatch_size: 64,
        replay_buffer_size: 50_000,
        train_interval: 1,
        eval_interval: 5_000,
        eval_episodes: 10,
        train_batch_size: 128,
        train_steps: 100_000,
        warmup_steps: 1_000,
    };

    let training = RLTraining::new(ARTIFACT_DIR, CartPoleWrapper::new)
        .metrics_train(LossMetric::new())
        .metrics_agent(ExplorationRateMetric::new())
        .metrics_episode((
            CumulativeRewardMetric::new(),
            EpisodeLengthMetric::new(),
        ))
        .with_file_checkpointer(CompactRecorder::new())
        .num_steps(learning_config.train_steps)
        .with_learning_strategy(RLStrategies::OffPolicyStrategy(learning_config))
        .summary();

    training.launch(agent);
}
```

Things to know:

- **`RLTraining::new(artifact_dir, env_factory)`** — `env_factory` is a closure that builds a fresh environment instance. The strategy uses it to spin up `num_envs` environments for parallel data collection.
- **Three metric channels**, not two:
  - `metrics_train(...)` — every gradient step (loss, etc.)
  - `metrics_agent(...)` — agent state (exploration rate, target-net distance)
  - `metrics_episode(...)` — per-episode (reward, length)
- **`autobatch_size`** — how many environment steps to batch together before sending to the policy network for action selection. Decouples env throughput from network call cost.
- **`warmup_steps`** — random-action steps to populate the replay buffer before any gradient steps fire. Without warmup, early training is gradient-on-empty-buffer noise.
- **Checkpointer is the same shape** as the supervised path — drop in any recorder (`CompactRecorder`, `NamedMpkFileRecorder<FullPrecisionSettings>`, etc.) and the strategy handles save/restore.

When to use this vs a manual loop:
- Use `RLTraining` when your problem fits the off-policy mold: a single-agent MDP, a Gym-style environment, replay-buffer training. DQN, DDQN, SAC, TD3 all fit.
- Use a manual loop for on-policy methods (PPO, A2C), multi-agent, model-based RL, or when the framework's metric channels don't match your objective.

The 0.21 release post explicitly says training-loop generalization is ongoing — expect more strategies (on-policy, distillation, etc.) in subsequent releases.

## Things that look like PyTorch but aren't

| You'd write in PyTorch | In Burn you write |
| ---------------------- | ----------------- |
| `optimizer.zero_grad()` | nothing — `step` consumes grads |
| `loss.backward()` then `optimizer.step()` | `let grads = loss.backward(); model = optim.step(lr, model, GradientsParams::from_grads(grads, &model));` |
| `with torch.no_grad():` | call `model.valid()` once, use that for inference |
| `model.to(device)` | `model.to_device(&device)` |
| `model.eval()` / `model.train()` | dropout/norm modules use a `running` flag managed by the framework — you usually don't switch modes manually |
| `torch.save(model.state_dict(), path)` | `model.into_record().save(path)?` |
| `model.load_state_dict(torch.load(path))` | `model.load_record(ModuleRecord::load(path)?)` |
| `lr_scheduler.step()` | done automatically inside `optim.step` when using the `Learner` path |
