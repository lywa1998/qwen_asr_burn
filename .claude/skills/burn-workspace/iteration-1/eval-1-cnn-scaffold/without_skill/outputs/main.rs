use burn::prelude::*;

mod model;
use model::Cifar10Cnn;

fn main() {
    // Create a WGPU device on the default GPU.
    let device = Device::wgpu(DeviceKind::DefaultDevice);

    println!("Device: {:?}", device);

    // Build the model on the selected device.
    let model = Cifar10Cnn::new(&device);
    println!("CIFAR-10 CNN model created successfully.");
    println!("Model: {:?}", model);

    // --- Quick smoke-test with a dummy batch of 2 images ---
    // CIFAR-10 images are 3×32×32 RGB, normalized to [0, 1].
    // We create a zero-filled tensor as a placeholder.
    let dummy_input = Tensor::<4>::zeros([2, 3, 32, 32], &device);

    let logits = model.forward(dummy_input);
    println!("Forward pass complete.");
    println!("Output shape: {:?}", logits.dims());
    println!("Output (first row logits): {:?}", logits.slice([0..1, 0..10]));

    // Show predicted class for the first dummy image.
    let predicted: i64 = logits
        .slice([0..1, 0..10])
        .argmax(1)
        .into_scalar();
    println!("Predicted class (dummy): {predicted}");

    println!("\n--- Setup ready for CIFAR-10 training ---");
    println!("Next steps:");
    println!("  1. Add a batcher that loads CIFAR-10 PNGs or the built-in dataset.");
    println!("  2. Implement TrainStep + InferenceStep on Cifar10Cnn.");
    println!("  3. Wire up DataLoader + Learner to run epochs.");
}
