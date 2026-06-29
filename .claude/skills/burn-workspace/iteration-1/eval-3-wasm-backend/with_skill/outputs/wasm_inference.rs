//! Minimal Burn inference for WebAssembly (browser, pure CPU, no GPU).
//!
//! Build: wasm-pack build --target web --release
//!
//! This example uses the Burn 0.21 API:
//!   - Flex backend (pure-Rust CPU, ideal for WASM)
//!   - No backend generic (`Tensor<D>`, not `Tensor<B, D>`)
//!   - `ModuleRecord::from_bytes` for compile-time weight embedding
//!   - `include_bytes!` to bake the `.bpk` file into the WASM binary

use burn::{
    nn::{Linear, LinearConfig, Relu},
    prelude::*,
    store::ModuleRecord,
};
use wasm_bindgen::prelude::*;

// ---------------------------------------------------------------------------
// 1. Define the model (Burn 0.21 style — no B: Backend generic)
// ---------------------------------------------------------------------------

#[derive(Module, Debug)]
pub struct Model {
    linear1: Linear,
    linear2: Linear,
    activation: Relu,
}

#[derive(Config, Debug)]
pub struct ModelConfig {
    input_dim: usize,
    hidden_dim: usize,
    output_dim: usize,
}

impl ModelConfig {
    /// Initialize the model on the given device with random weights.
    pub fn init(&self, device: &Device) -> Model {
        Model {
            linear1: LinearConfig::new(self.input_dim, self.hidden_dim).init(device),
            linear2: LinearConfig::new(self.hidden_dim, self.output_dim).init(device),
            activation: Relu::new(),
        }
    }
}

impl Model {
    /// Forward pass. Input shape: [batch_size, input_dim].
    pub fn forward(&self, x: Tensor<2>) -> Tensor<2> {
        let x = self.linear1.forward(x);
        let x = self.activation.forward(x);
        self.linear2.forward(x)
    }
}

// ---------------------------------------------------------------------------
// 2. Embed the compiled-in .bpk weights (baked at compile time, no runtime fs)
// ---------------------------------------------------------------------------

static MODEL_WEIGHTS: &[u8] = include_bytes!("../assets/model.bpk");

// ---------------------------------------------------------------------------
// 3. Create the model, load weights, and run inference
// ---------------------------------------------------------------------------

/// One-time initialization. Call this once from JavaScript before inference.
#[wasm_bindgen]
pub fn init_model() -> Result<(), JsValue> {
    let device = Device::flex();

    // Parse the compiled-in bytes into a backend-independent ModuleRecord
    let record = ModuleRecord::from_bytes(MODEL_WEIGHTS)
        .map_err(|e| JsValue::from_str(&format!("Failed to parse model weights: {e}")))?;

    // Load the weights into the model
    let model = ModelConfig::new(784, 128, 10)
        .init(&device)
        .load_record(record);

    // Optional: verify a forward pass works
    let dummy = Tensor::<2>::zeros([1, 784], &device);
    let _output = model.forward(dummy);

    // In a real app, store the model in a static (e.g., once_cell::sync::OnceCell)
    // or thread-local and use it from `run_inference`.

    Ok(())
}

/// Run inference on a single input sample (flat f32 slice).
/// Returns the output logits as a Vec<f32>.
#[wasm_bindgen]
pub fn run_inference(input: &[f32], input_dim: usize, output_dim: usize) -> Vec<f32> {
    let device = Device::flex();

    // Load model every call for simplicity — in production, use OnceCell
    let record = ModuleRecord::from_bytes(MODEL_WEIGHTS)
        .expect("Model weights should parse");
    let model = ModelConfig::new(input_dim, 128, output_dim)
        .init(&device)
        .load_record(record);

    // Build input tensor from the flat slice
    let input_tensor = Tensor::<2>::from_data(
        TensorData::new(input.to_vec(), Shape::new([1, input_dim])),
        &device,
    );

    // Forward pass
    let output = model.forward(input_tensor);

    // Sync the result back to host (this is the one sync per inference call)
    output.into_data().to_vec().unwrap()
}

// ---------------------------------------------------------------------------
// 4. Production-ready pattern: use OnceCell to init once
// ---------------------------------------------------------------------------

#[cfg(feature = "once_cell")]
mod production_pattern {
    use super::*;
    use once_cell::sync::OnceCell;

    static MODEL: OnceCell<Model> = OnceCell::new();

    fn get_model() -> &'static Model {
        MODEL.get_or_init(|| {
            let device = Device::flex();
            let record = ModuleRecord::from_bytes(MODEL_WEIGHTS)
                .expect("Failed to parse compiled-in model weights");
            ModelConfig::new(784, 128, 10)
                .init(&device)
                .load_record(record)
        })
    }

    #[wasm_bindgen]
    pub fn run_inference_cached(input: &[f32]) -> Vec<f32> {
        let device = Device::flex();
        let model = get_model();

        let input_tensor = Tensor::<2>::from_data(
            TensorData::new(input.to_vec(), Shape::new([1, 784])),
            &device,
        );

        model.forward(input_tensor).into_data().to_vec().unwrap()
    }
}
