# Weight Loading Patterns

## Complete Pipeline Pattern

Every pipeline follows this sequence:

```rust
// 1. Load config from JSON
let config_path = format!("{model_dir}/config.json");
let model_config = ModelConfig::from_file(&config_path)?;

// 2. Create model with random weights
let mut model = Model::new(&model_config, &device);

// 3. Load safetensors and apply to model
let weights_path = format!("{model_dir}/model.safetensors");
let mut store = SafetensorsStore::from_file(&weights_path)
    .with_from_adapter(adapter);
let result = model.load_from(&mut store)?;

// 4. Log results
log::info!("Loaded {} weights, {} errors",
    result.applied.len(), result.errors.len());
```

## Config Deserialization

Config structs map 1:1 to JSON keys using serde:

```rust
#[derive(Debug, Clone, Deserialize)]
pub struct ModelConfig {
    pub hidden_size: usize,           // required field
    pub num_hidden_layers: usize,     // required field
    
    #[serde(default)]                 // uses Default::default() if missing
    pub use_qk_norm: bool,
    
    #[serde(default = "default_eps")] // uses named function if missing
    pub rms_norm_eps: f64,
    
    #[serde(default)]
    pub rope_scaling: Option<RopeScaling>,  // optional sub-struct
    
    #[serde(rename = "model_type")]   // JSON key differs from field name
    pub model_type_field: String,
}

fn default_eps() -> f64 { 1e-5 }

impl ModelConfig {
    pub fn from_file(path: &str) -> anyhow::Result<Self> {
        let content = std::fs::read_to_string(path)?;
        Ok(serde_json::from_str(&content)?)
    }
}
```

## Adapter Chain

### Why adapters exist

HuggingFace safetensors are saved in PyTorch format:
- Linear weights are `[out_features, in_features]` (transposed from burn's convention)
- Parameter paths use dot-separated Python module paths (e.g., `model.layers.0.self_attn.q_proj.weight`)

`PyTorchToBurnAdapter` handles the transpose and path mapping automatically.

### BF16 → F32 Conversion

Burn's Metal backend does not support BF16. When source weights are BF16:

```rust
#[cfg(feature = "metal")]
use burn_store::ChainAdapter;

#[cfg(feature = "metal")]
let adapter = ChainAdapter::new(PyTorchToBurnAdapter, Bf16ToF32Adapter);
#[cfg(not(feature = "metal"))]
let adapter = PyTorchToBurnAdapter;
```

The Bf16ToF32Adapter converts each weight tensor from BF16 to F32 during loading:
```rust
impl ModuleAdapter for Bf16ToF32Adapter {
    fn adapt(&self, snapshot: &TensorSnapshot) -> TensorSnapshot {
        if snapshot.dtype != DType::BF16 {
            return snapshot.clone();
        }
        // Convert to F32 via closure
        let original = snapshot.clone_data_fn();
        let cast = Rc::new(move || {
            let data = original()?;
            Ok(data.convert_dtype(DType::F32))
        });
        TensorSnapshot::from_closure(cast, DType::F32, snapshot.shape.clone(), ...)
    }
}
```

### Partial Loading

For models where not all weights need to be loaded (e.g., alignment head has extra weights):
```rust
let mut store = SafetensorsStore::from_file(&weights_path)
    .with_from_adapter(adapter)
    .allow_partial(true);
let result = model.load_from(&mut store)?;

// Check for errors (expected for partial loads)
if !result.errors.is_empty() {
    log::warn!("{} weight load errors:", result.errors.len());
    for err in result.errors.iter().take(10) {
        log::warn!("  {:?}", err);
    }
}
```

### Pre-converted F32 Weights (Optimization)

To skip BF16 conversion at load time, pre-convert weights to F32 and save as `model_f32.safetensors`:
```rust
let f32_path = format!("{model_dir}/model_f32.safetensors");
let safetensors_path = if std::path::Path::new(&f32_path).exists() {
    f32_path
} else {
    format!("{model_dir}/model.safetensors")
};
```

## Load Result Handling

`model.load_from()` returns a `LoadResult`:
```rust
pub struct LoadResult {
    pub applied: Vec<String>,    // successfully loaded parameter paths
    pub errors: Vec<LoadError>,  // failed loads (mismatched shapes, missing keys)
}
```

Common error causes:
- **Mismatched shapes**: Config doesn't match the saved weights (wrong hidden_size, num_layers, etc.)
- **Missing keys**: Weights in safetensors don't correspond to any module field (e.g., PyTorch-only buffers)
- **Incompatible types**: Weight tensor type doesn't match the module parameter type (without adapter)

## Tokenizer Loading

Use the `tokenizers` crate for HuggingFace-compatible tokenizers:

```rust
use tokenizers::Tokenizer;

let tokenizer_path = format!("{model_dir}/tokenizer.json");
let mut tokenizer = Tokenizer::from_file(&tokenizer_path)?;

// Set padding (optional)
tokenizer.with_padding(Some(tokenizers::PaddingParams {
    strategy: tokenizers::PaddingStrategy::Fixed(0),
    pad_id: pad_token_id as u32,
    ..Default::default()
}));

// Encode
let encoding = tokenizer.encode(text, false)?;
let ids: Vec<u32> = encoding.get_ids().iter().map(|&id| id).collect();

// Decode
let text = tokenizer.decode(&ids, true)?;
```

The `tokenizers` crate supports `tokenizer.json` (HuggingFace fast tokenizer format). If only `vocab.json` + `merges.txt` are available, convert to `tokenizer.json` format using the Python conversion script described in the project CLAUDE.md.
