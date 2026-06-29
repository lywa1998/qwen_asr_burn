// `recursion_limit` is required by the WGPU backend — the nested associated
// types in CubeCL exceed the default 128.
#![recursion_limit = "256"]

mod model;

use burn::prelude::*;
use model::ModelConfig;

fn main() {
    // ---- device setup ----
    // Pick WGPU for cross-platform GPU (Vulkan / Metal / DX12 / WebGPU).
    // DeviceKind::DefaultDevice picks the first available discrete GPU.
    let device = Device::wgpu(DeviceKind::DefaultDevice);

    // For training, wrap the device with autodiff tracking so gradients flow.
    let autodiff_device = device.clone().autodiff();

    // ---- model init ----
    let model = ModelConfig::new(10) // 10 CIFAR-10 classes
        .init(&autodiff_device);

    println!("Model initialized on {:?}", autodiff_device);
    println!("Number of parameters: {}", model.num_params());

    // ---- quick smoke-test forward pass with dummy data ----
    // Create a random batch of 4 images (BCHW: 4 × 3 × 32 × 32).
    let dummy_images = Tensor::<4>::random(
        [4, 3, 32, 32],
        Distribution::Default,
        &device,
    );

    let logits = model.forward(dummy_images);
    println!("Forward pass output shape: {:?}", logits.dims()); // [4, 10]

    // ---- if you were training, you would continue with: ----
    //
    // use burn::{
    //     data::dataloader::DataLoaderBuilder,
    //     optim::AdamConfig,
    //     train::{
    //         ClassificationOutput, InferenceStep, Learner,
    //         SupervisedTraining, TrainOutput, TrainStep,
    //         metric::{AccuracyMetric, LossMetric},
    //     },
    // };
    //
    // // Build dataloaders from the built-in CifarDataset:
    // let dataloader_train = DataLoaderBuilder::new(CifarBatcher::default())
    //     .batch_size(64)
    //     .shuffle(42)
    //     .num_workers(4)
    //     .build(burn::data::dataset::vision::CifarDataset::train());
    //
    // let dataloader_test = DataLoaderBuilder::new(CifarBatcher::default())
    //     .batch_size(64)
    //     .num_workers(4)
    //     .build(burn::data::dataset::vision::CifarDataset::test());
    //
    // let training = SupervisedTraining::new(
    //     "artifacts/cifar10-cnn",
    //     dataloader_train,
    //     dataloader_test,
    // )
    // .metrics((AccuracyMetric::new(), LossMetric::new()))
    // .with_checkpointer()
    // .num_epochs(20)
    // .summary();
    //
    // let result = training.launch(Learner::new(
    //     model,
    //     AdamConfig::new().init(),
    //     1e-3, // constant LR
    // ));
    //
    // result.model.into_record().save("artifacts/cifar10-cnn/model").unwrap();
}
