---
name: transformers-to-burn
description: Guide for migrating HuggingFace transformers Python models to the Burn Rust deep learning framework. Use when the user asks about porting, migrating, converting, or rewriting a transformers model in Rust, implementing a model architecture from HuggingFace in Burn, or translating Python model code to Rust. Triggers on keywords: transformers to burn, port to rust, migrate model, PyTorch to burn, convert transformer, 迁移到 burn, 模型转换, Python 转 Rust.
---

# Transformers → Burn Migration Guide

This skill provides a high-level plan for migrating a HuggingFace `transformers` Python model to the [Burn](https://github.com/tracel-ai/burn) Rust deep learning framework. For concrete implementation patterns (code templates, API mappings, tensor operation cheatsheets), read the files in `references/`.

## The Migration Process

A migration follows this order:

1. **Analyze** the Python source — identify the component hierarchy, attention mechanism, norm types, activation functions
2. **Map** each Python component to the corresponding Burn pattern (see `references/component-mapping.md`)
3. **Implement** in Rust, following the established file-split convention
4. **Load** weights from safetensors files with the appropriate adapter chain (see `references/weight-loading.md`)
5. **Verify** by comparing forward-pass outputs against the Python reference

## Step 1: Analyze the Python Source

Locate the model definition in the transformers reference code — typically under `transformers/src/transformers/models/<model_name>/` with files named `modeling_*.py` and `configuration_*.py`.

Extract the component hierarchy by tracing `forward()`:

1. What embedding layer?
2. What norm type? (`RMSNorm`, `LayerNorm`, QK-norm?)
3. What attention? Standard MHA, GQA, QK-norm? Any flash-attention or varlen paths?
4. What MLP? SwiGLU, vanilla FFN, MoE?
5. What position encoding? Standard RoPE, MRoPE, ALiBi, learned?
6. Residual connection order? Pre-norm or post-norm?
7. Any multimodal inputs? (audio features, images)
8. What's the top-level model class? `ForCausalLM`, `ForConditionalGeneration`?
9. **What attention mask pattern does each sub-module use?** — Read the Python `forward()` carefully for `cu_seqlens`, custom 4D masks, block-diagonal patterns, or flash-attention varlen paths. These are the most common source of silent correctness bugs.

## Step 2: Map Components to Burn Patterns

Use `references/component-mapping.md` for the detailed Python→Rust mapping tables and code templates.

## Step 3: Plan the Implementation

### File Organization

Split the model into files by component type (attention, mlp, norm, etc.) rather than putting everything in one file. Use a directory module under `src/models/`. Group related types together — each file should be readable on its own without excessive scrolling. Use `super::` for cross-file imports within the model directory.

### Implementation Order

Implement bottom-up: norms → MLP → RoPE → attention → decoder layer → encoder → top-level model → weight loading → pipeline.

### Attention Guidance

Decoder attention follows a common flow across GQA models. See `references/component-mapping.md` for the detailed sequence. Match the mask style (Bool vs. float) used in the Python source.

### When an Encoder is Present

If the model has an audio or vision encoder, pay special attention to:
- Does the encoder use **restricted attention** (block-diagonal, windowed) rather than global self-attention?
- Are positional embeddings **per-chunk** (restarting from 0) or **global** (continuous)?
- Is there a **conv frontend** that downsamples before the transformer layers?

The Python `forward()` method is the ground truth — look for `cu_seqlens`, custom 4D masks, or chunk/pad/unpad logic. Getting this wrong is the most common cause of subtle output degradation.

## Step 4: Load Weights

Follow the patterns in `references/weight-loading.md`. Key considerations:
- Config is deserialized from JSON — include defaults for optional fields
- Weights are loaded from safetensors via an adapter chain
- On Metal backends, BF16 weights need dtype conversion (the backend doesn't support BF16 natively)
- Models with non-critical missing weights can use partial loading

## Step 5: Verify

Verify in this order:
1. `cargo check` — compile with zero warnings
2. Shape check — print tensor dims at each layer boundary; compare against Python
3. Forward pass — run one forward pass with real weights, compare output against Python reference
4. End-to-end — run the full pipeline on a known input

For numerical comparison, save Python reference outputs as flat f32 `.bin` files and load them in Rust for diff comparison.

## Symptom-Driven Debugging

When the migration compiles but produces wrong output, use these patterns:

| Symptom | Most Likely Cause |
|---|---|
| All output garbage / random | Weights not loaded (shape/dtype mismatch) |
| Correct start → degrades into hallucinations | Encoder attention mismatch (global vs. chunked) or positional embedding mismatch (global vs. per-chunk) |
| Output plausible but semantically wrong | Prompt/template error, wrong special tokens |
| Repeats same token endlessly | EOS not recognized, or KV-cache stale |
| Numerical NaNs | Long sequence softmax overflow; segment the input |
| Correct Python, wrong Rust (same weights) | Compare intermediate tensors layer-by-layer to find first divergence |

## Common Pitfalls

- **Attention mode mismatch**: The #1 silent bug. Many encoders use restricted attention. Always check the Python `forward()` for `cu_seqlens`, 4D masks, or varlen paths.
- **Config field defaults**: Missing keys in `config.json` fail serde without `#[serde(default)]`. Audit the JSON against all fields the Python code accesses.
- **Dimension indexing**: Burn 0-indexed. Python `dim=-1` = Burn `D-1`. No negative slicing.
- **Clone before reuse**: Burn tensors are consumed. `.clone()` when needed after an operation.
- **Module privacy**: Config fields accessed across files must be `pub`.
- **`#[module(skip)]`**: Required on all scalar fields stored on a `#[derive(Module)]` struct.
- **Type mismatch**: BF16 weights → F32 adapter must match the backend's float type.
