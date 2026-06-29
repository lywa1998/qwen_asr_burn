# Component Mapping: Python → Rust

## Architecture Patterns

### Inheritance → Composition

Python transformers use class inheritance:
```
PreTrainedModel → ModelPreTrainedModel → ModelForCausalLM (mixes in GenerationMixin)
```

Burn uses composition via struct fields:
```rust
#[derive(Module, Debug)]
pub struct ModelForCausalLM {
    pub model: ModelBackbone,
    pub lm_head: nn::Linear,
}
```

### Config Patterns

**Python** (dataclass-style):
```python
class ModelConfig(PretrainedConfig):
    model_type = "my_model"
    def __init__(self, hidden_size=4096, num_layers=32, **kwargs):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
```

**Rust** (serde-deserialized):
```rust
#[derive(Debug, Clone, Deserialize)]
pub struct ModelConfig {
    pub hidden_size: usize,
    #[serde(default)]
    pub num_hidden_layers: usize,
}
```

## Layer Mappings

### Embedding

**Python:** `nn.Embedding(vocab_size, hidden_size)`
**Rust:** `EmbeddingConfig::new(vocab_size, hidden_size).init(device)`

Forward signature: Input `[B, S]` Int → Output `[B, S, H]` Float

### Linear

**Python:** `nn.Linear(in_features, out_features, bias=False)`
**Rust:** `LinearConfig::new(in_features, out_features).with_bias(false).init(device)`

PyTorch weight shape: `[out_features, in_features]` (transposed from Burn)
PyTorchToBurnAdapter handles the transpose automatically.

### Attention

**Python (typical GQA):**
```python
class Attention(nn.Module):
    def __init__(self, config):
        self.q_proj = nn.Linear(hidden, num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden, num_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden, num_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * head_dim, hidden, bias=False)
```

**Rust:** Same pattern, using `LinearConfig::new(in, out).with_bias(false).init(device)`.

Forward flow in Rust:
1. Linear projections (separate Q/K/V)
2. `reshape([B, S, heads, head_dim])`
3. `swap_dims(1, 2)` → `[B, heads, S, head_dim]`
4. QK norm (if present)
5. Rotary embeddings
6. KV cache concat (if caching)
7. `repeat_kv()` for GQA
8. `q.matmul(k.swap_dims(2, 3)).div_scalar(scale)`
9. Mask application
10. `softmax(..., 3)`
11. `attn.matmul(v)`
12. `swap_dims(1, 2)` → `reshape([B, S, heads*head_dim])`
13. Output projection

### Norm Types

**RMSNorm:**
- Python: `nn.RMSNorm(hidden_size, eps=1e-6)`
- Rust: Custom struct with `Param<Tensor<1>>` weight, `#[module(skip)] epsilon`
- Forward: `rms(x) = sqrt(mean(x^2) + eps)` → `x / rms * weight`

**LayerNorm:**
- Python: `nn.LayerNorm(hidden_size, eps=1e-5)`
- Rust: Custom struct with `Param<Tensor<1>>` weight + bias
- Forward: `(x - mean) / sqrt(var + eps) * weight + bias`

### MLP Types

**SwiGLU** (silu gate):
- Python: `gate = F.silu(self.gate_proj(x)); up = self.up_proj(x); return self.down_proj(gate * up)`
- Rust: `silu(gate_proj.forward(x.clone())) * up_proj.forward(x)` → `down_proj.forward(result)`

**Vanilla FFN:**
- Python: `F.gelu(self.fc1(x))` → `self.fc2(...)`
- Rust: `gelu(self.fc1.forward(x))` → `self.fc2.forward(result)`

### Rotary Embeddings

**Standard RoPE:**
- Python: Pre-computed `cos_cached`, `sin_cached` buffers
- Rust: `compute(seq_len, device) -> (cos, sin)` computes on-the-fly
- Application: split head_dim in half, rotate by cos/sin

**MRoPE (multi-dimensional position):**
- Python: position IDs with multiple components (temporal, height, width) interleaved across frequency bands
- Rust: compute cos/sin from explicit position arrays rather than a single scalar position
- Application: same half-rotation pattern as standard RoPE, applied per frequency band

### Decoder Layer

**Pre-norm (current standard):**
```
residual = hidden
hidden = input_layernorm(hidden)
hidden = attn(hidden, ...)
hidden = hidden + residual
residual = hidden
hidden = post_attention_layernorm(hidden)
hidden = mlp(hidden)
hidden = hidden + residual
```

## Activation Functions

| Python | Rust burn |
|---|---|
| `F.silu(x)` | `silu(x)` |
| `F.gelu(x)` | `gelu(x)` |
| `torch.sigmoid(x)` | `sigmoid(x)` |
| `F.softmax(x, dim=-1)` | `softmax(x, D-1)` |
| `F.relu(x)` | `relu(x)` |

## Tensor Operation Mappings

| Python | Rust burn | Notes |
|---|---|---|
| `x.reshape(a, b)` | `x.reshape([a, b])` | Burn takes array |
| `x.view(a, b)` | `x.reshape([a, b])` | Same as reshape |
| `x.transpose(1, 2)` | `x.swap_dims(1, 2)` | |
| `x.permute(0, 2, 1, 3)` | Multiple swap_dims | No direct permute |
| `x.unsqueeze(0)` | `x.unsqueeze_dim::<D>(0)` | Generic D = target rank |
| `x.unsqueeze(-1)` | `x.unsqueeze_dim::<D>(D-1)` | |
| `x.squeeze(dim)` | No direct equivalent | Reshape to remove dim |
| `x.expand(a, b, c)` | `x.expand([a, b, c])` | |
| `x.repeat(a, b)` | `x.repeat_dim(dim, n)` | Per-dimension |
| `torch.cat([a, b], dim=0)` | `Tensor::cat(vec![a, b], 0)` | |
| `x[:, start:end, :]` | `x.slice([0..B, start..end, 0..H])` | Full-range slice |
| `x[:, -1:, :]` | `x.narrow(1, len-1, 1)` | Narrow is offset+len |
| `torch.bmm(q, k)` | `q.matmul(k)` | |
| `x / y` | `x.div(y)` | |
| `x * y` | `x.mul(y)` | |
| `x + y` | `x.add(y)` | |
| `x ** 2` | `x.powf_scalar(2.0)` | |
| `x.sqrt()` | `x.sqrt()` | |
| `x.mean(dim=-1)` | `x.mean_dim(D-1)` | |
| `x.sum()` | `x.sum()` | |
| `x.argmax(dim=-1)` | `x.argmax(D-1)` | |
| `x.float()` | `x.cast(DType::F32)` | |
| `x.to(device)` | Not needed | Device inferred from inputs |

## Weight Naming Convention

HuggingFace safetensors keys use dotted module paths:
```
model.embed_tokens.weight
model.layers.0.input_layernorm.weight
model.layers.0.self_attn.q_proj.weight
model.layers.0.self_attn.k_proj.weight
model.layers.0.self_attn.v_proj.weight
model.layers.0.self_attn.o_proj.weight
model.layers.0.post_attention_layernorm.weight
model.layers.0.mlp.gate_proj.weight
model.layers.0.mlp.up_proj.weight
model.layers.0.mlp.down_proj.weight
model.norm.weight
lm_head.weight
```

PyTorchToBurnAdapter maps these to the Rust module hierarchy automatically: the dotted path corresponds to struct field access (e.g., `model.layers.0.self_attn.q_proj` → `model.layers[0].self_attn.q_proj`).

## Special Considerations

### QK Normalization

Some models apply per-head normalization to Q and K projections before attention:
- Python: Separate norm layer (typically RMSNorm) applied to Q and K individually
- Rust: A 4D norm that normalizes along the head_dim dimension
- The QK norm weights are separate parameters, stored under the attention module

### MoE (Mixture of Experts)

Models with MoE layers replace the dense MLP with routed experts:
- Gate/router: projects hidden state to num_experts, selects top-k via sigmoid or softmax
- Experts: the weight tensors gain an extra dimension for the expert count
- Outputs from selected experts are weighted and combined, often with a shared expert added

### Multimodal Fusion

Models that combine multiple modalities (audio+text, image+text) typically fuse features at the embedding level:
- The encoder produces features at designated placeholder token positions
- During the forward pass, placeholder embeddings are replaced with encoder outputs
- After prefill, the fused features persist in the KV-cache and are not recomputed during decode
- In Rust, this is typically implemented via a slice-and-concatenate pattern
