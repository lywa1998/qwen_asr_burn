//! Burn WASM inference example — pure CPU via Flex backend.
//!
//! Prerequisites:
//! 1. Train your model natively and save it: `model.into_record().save("model")?;`
//! 2. Copy `model.bpk` into your project's `assets/` directory.
//! 3. Build: `wasm-pack build --target web`
//! 4. Serve from a web page (see index.html snippet at bottom).
//!
//! Cargo.toml snippet:
//! ```toml
//! [dependencies]
//! burn = { version = "0.21", default-features = false, features = ["flex", "store", "std"] }
//! wasm-bindgen = "0.2"
//! ```

use burn::{
    nn::{Linear, LinearConfig, Relu},
    prelude::*,
    store::ModuleRecord,
};
use wasm_bindgen::prelude::*;

// ---------------------------------------------------------------------------
// 1. Define your model (same struct as on the training side)
// ---------------------------------------------------------------------------

#[derive(Module, Debug)]
pub struct MyModel {
    linear1: Linear,
    linear2: Linear,
    activation: Relu,
}

#[derive(Config, Debug)]
pub struct MyModelConfig {
    input_dim: usize,
    hidden_dim: usize,
    output_dim: usize,
}

impl MyModelConfig {
    pub fn init(&self, device: &Device) -> MyModel {
        MyModel {
            linear1: LinearConfig::new(self.input_dim, self.hidden_dim).init(device),
            linear2: LinearConfig::new(self.hidden_dim, self.output_dim).init(device),
            activation: Relu::new(),
        }
    }
}

impl MyModel {
    pub fn forward(&self, x: Tensor<2>) -> Tensor<2> {
        let x = self.linear1.forward(x);
        let x = self.activation.forward(x);
        self.linear2.forward(x)
    }
}

// ---------------------------------------------------------------------------
// 2. Embed model weights at compile time
// ---------------------------------------------------------------------------

/// Compile-time embedded model weights from a `.bpk` file.
///
/// Place your `model.bpk` in the project's `assets/` directory.
/// `include_bytes!` bakes the bytes into the WASM binary — zero filesystem
/// access at runtime.
static MODEL_WEIGHTS: &[u8] = include_bytes!("../assets/model.bpk");

// ---------------------------------------------------------------------------
// 3. One-time initialization (call once at page load)
// ---------------------------------------------------------------------------

/// Initialize the model and load weights from the embedded bytes.
///
/// Call this once at startup and hold onto the model for repeated inference.
/// Returns `true` on success, panics on failure with a console error.
#[wasm_bindgen]
pub fn init_model() -> bool {
    // Flex backend: pure-Rust CPU, runs in any WASM runtime, no GPU needed.
    let device = Device::flex();

    // Deserialize weights from the compile-time embedded bytes.
    let record = ModuleRecord::from_bytes(MODEL_WEIGHTS)
        .expect("Failed to deserialize model weights from embedded .bpk");

    // Build the model and load weights — same pattern as native code.
    let model = MyModelConfig::new(784, 128, 10)
        .init(&device)
        .load_record(record);

    // Store the model in a global or thread-local so `run_inference` can
    // access it. The pattern below uses `std::cell::RefCell<Option<MyModel>>`
    // in a `thread_local!` (or you can return the model handle and manage it
    // in JavaScript with `wasm_bindgen` exported structs).
    //
    // For simplicity this example uses a thread-local:
    MODEL.with(|cell| {
        *cell.borrow_mut() = Some(model);
    });
    true
}

// ---------------------------------------------------------------------------
// 4. Inference — call per input batch
// ---------------------------------------------------------------------------

/// Run inference on a batch of inputs.
///
/// `input_data` is a flat f32 slice. The first `batch_size * input_dim`
/// elements are used. Returns the flat output floats.
#[wasm_bindgen]
pub fn run_inference(input_data: &[f32]) -> Vec<f32> {
    MODEL.with(|cell| {
        let model = cell.borrow();
        let model = model.as_ref().expect("init_model() must be called first");

        let device = Device::flex();

        // Reshape the flat input into a [batch_size, input_dim] tensor.
        // Hardcoded batch_size=1, input_dim=784 for this example.
        let batch_size = 1;
        let input_dim = 784;
        let input = Tensor::<2>::from_floats(input_data, &device)
            .reshape([batch_size, input_dim]);

        let output: Tensor<2> = model.forward(input);

        // into_data() synchronizes — blocks until the tensor is available.
        // Flex is eager and runs on the calling thread, so this is immediate.
        output.into_data().to_vec().unwrap()
    })
}

/// Return the output dimension so callers know how many floats to expect.
#[wasm_bindgen]
pub fn output_dim() -> usize {
    10
}

// ---------------------------------------------------------------------------
// 5. Thread-local model storage (single-threaded WASM)
// ---------------------------------------------------------------------------

std::thread_local! {
    static MODEL: std::cell::RefCell<Option<MyModel>> = const { std::cell::RefCell::new(None) };
}

// ---------------------------------------------------------------------------
// 6. HTML frontend snippet (save as index.html)
// ---------------------------------------------------------------------------
//
// ```html
// <!DOCTYPE html>
// <html>
// <head><meta charset="utf-8"><title>Burn WASM Inference</title></head>
// <body>
//   <canvas id="input-canvas" width="28" height="28"></canvas>
//   <button id="predict-btn">Predict</button>
//   <pre id="output"></pre>
//
//   <script type="module">
//     import init, { init_model, run_inference, output_dim } from './pkg/burn_wasm_inference.js';
//
//     async function main() {
//       await init();                   // load the .wasm file
//       init_model();                   // deserialize weights
//       console.log("Model ready.");
//
//       document.getElementById('predict-btn').onclick = () => {
//         // Gather input from e.g. a canvas or text field
//         const pixels = new Float32Array(784);
//         // ... fill pixels ...
//         const logits = run_inference(pixels);
//         document.getElementById('output').textContent =
//           Array.from(logits).map((v, i) => `class ${i}: ${v.toFixed(4)}`).join('\n');
//       };
//     }
//     main();
//   </script>
// </body>
// </html>
// ```
