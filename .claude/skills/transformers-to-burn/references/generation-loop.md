# Generation Loop Patterns

Two generation patterns correspond to two levels of complexity.

## Pattern A: Simple Autoregressive

Used when the model has no KV cache or when simplicity is preferred. Recomputes the full sequence at each step.

```
┌─────────────────────────────────────────┐
│ Tokenize prompt → token_ids              │
│ prompt_len = len(token_ids)              │
│                                          │
│ for step in 0..max_new_tokens:           │
│   input = reshape(token_ids, [1, len])   │
│   logits = model.forward(input, ...)     │  ← full forward pass
│   last_logits = logits[:, -1, :]         │  ← extract last position
│   next_token = sample/argmax(last_logits)│
│   if next_token == eos: break            │
│   token_ids.push(next_token)             │
│                                          │
│ Decode token_ids[prompt_len..] → text    │
└─────────────────────────────────────────┘
```

Python equivalent: `GenerationMixin._sample()` without KV cache.

## Pattern B: KV-Cached Generation

More efficient. Prefill processes all prompt tokens once; decode steps process one token using cached K/V from previous steps.

```
┌──────────────────────────────────────────┐
│ PREFILL:                                 │
│   embeds = embed(prompt)                 │
│   pos_emb = position_encoding(...)       │
│   mask = create_causal_mask(seq_len, 0)  │
│   kv_cache = KvCache::new(num_layers)    │
│   hidden = model.forward(                │
│     embeds, pos_emb, mask, Some(kv_cache))│
│   last_logits = lm_head(hidden[:, -1:])  │
│   current_pos = seq_len                  │
│                                          │
│ DECODE LOOP:                             │
│   for step in 0..max_new:               │
│     next_token = argmax(last_logits)     │
│     if next_token in eos_ids: break      │
│     generated.push(next_token)           │
│     next_embed = embed([next_token])     │
│     pos_emb = position_encoding([current_pos])│
│     mask = create_causal_mask(1, past)   │
│     hidden = model.forward(              │
│       next_embed, pos_emb, mask, Some(kv_cache))│
│     last_logits = lm_head(hidden)        │
│     current_pos += 1                     │
│                                          │
│ Decode generated → text                  │
└──────────────────────────────────────────┘
```

Python equivalent: `GenerationMixin._sample()` with DynamicCache.

### KV Cache Structure

```rust
pub struct KvCacheEntry {
    pub k: Tensor<4>,  // [batch, num_heads, seq_len, head_dim]
    pub v: Tensor<4>,
}

pub struct KvCache {
    layers: Vec<Option<KvCacheEntry>>,  // one per decoder layer
}
```

Cache grows by concatenation along the seq_len dimension:
```rust
let (k_full, v_full) = if let Some(cache) = kv_cache {
    (
        Tensor::cat(vec![cache.k.clone(), k_rot.clone()], 2),
        Tensor::cat(vec![cache.v.clone(), v.clone()], 2),
    )
} else {
    (k_rot.clone(), v.clone())
};
```

### Causal Mask in Decode

During decode, the mask shape is `[1, 1, 1, past_len+1]` — allowing attention to all past tokens plus the current one. The mask values should all be `false` (no positions masked) since the single new token should attend to everything in the cache.

## Sampling Strategies

### Argmax (Greedy)

```rust
fn argmax_token(logits: &[f32]) -> usize {
    logits.iter().enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap_or(0)
}
```

### Top-K + Top-P (Nucleus) Sampling

1. Apply temperature: `logits / temperature`
2. Top-K filter: keep only top-k highest logits
3. Top-P filter: sort descending, softmax, cumulative sum, truncate at `cumsum > top_p`
4. Softmax again on remaining tokens
5. Sample from categorical distribution: random float → find bin

Python equivalent chain:
```python
logits_warper(logits)  # TemperatureLogitsWarper
logits_warper(logits)  # TopKLogitsWarper
logits_warper(logits)  # TopPLogitsWarper
probs = F.softmax(logits, dim=-1)
next_token = torch.multinomial(probs, 1)
```

## Stopping Conditions

Common stopping criteria to implement:
- **EOS token**: Break when generated token matches any eos_token_id
- **Max length**: Break when `step >= max_new_tokens`
- **Max time**: Break when elapsed exceeds a timeout
- **Repetition detection**: Break when tokens form repeating patterns

Full stop-string matching requires incremental decoding and is expensive; implement it only if needed.

## Multimodal Prefill

When the model fuses multiple modalities (audio, images) with text, the prefill step requires additional handling:

1. **Tokenize**: prompt text with placeholder tokens at fusion positions
2. **Embed tokens**: standard embedding lookup for the text portion
3. **Compute modality features**: run the encoder on the raw input
4. **Replace**: substitute placeholder token embeddings with encoder outputs
5. **Prefill**: run the full fused sequence through the decoder with KV cache
6. **Decode**: standard autoregressive — fused features persist in the KV cache

This mirrors Python's `prepare_inputs_for_generation()` which passes modality features as `None` after the first step.
