# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Run

```bash
# macOS (default)
cargo build

# Linux/NVIDIA
cargo build --no-default-features --features cuda

# Run commands
cargo run -- -m models/Qwen3-ASR-0.6B transcribe <input.wav>
cargo run -- -m models/Qwen3-ForcedAligner-0.6B align -i <input.wav> -t <text>
cargo run -- extract <video.mp4> -o audio.wav

# Run with logging
RUST_LOG=info cargo run -- -m models/Qwen3-ASR-0.6B transcribe <input.wav>
```

## Feature Flags

Two mutually exclusive backends controlled by Cargo features:

| Feature | burn backend | Float type | Default |
|---|---|---|---|
| `metal` | `Wgpu<f32, i32>` (via Metal) | `f32` | macOS |
| `cuda` | `Cuda<bf16, i32>` | `bf16` | Linux |

The `metal` feature uses `Wgpu<f32, i32>` because burn-wgpu 0.21 does NOT support BF16 (`cubecl` has open Metal MSL bf16 codegen bugs). At load time, a `Bf16ToF32Adapter` converts BF16 safetensors weights to F32 via `ChainAdapter`. The CUDA backend keeps BF16 natively with no conversion.

## Architecture

**Model architecture**: Qwen3-ASR uses a shared `Qwen3ASR<B>` struct (in `model.rs`) for both ASR and forced alignment. The audio encoder (Conv2D + Transformer) and text decoder share weights between the two tasks, differing only in the LM head output dimension (`vocab_size` vs `classify_num`). Both `AsrPipeline` and `AlignPipeline` instantiate the same `Qwen3ASR` with different configs.

**Pipeline flow** (`transcribe` command):
```
WAV → load_wav_samples (hound + rubato, 16kHz mono f32)
    → VAD (earshot, 16ms/frame, segments ≤30s)
    → per-segment:
        MelSpectrogram (rustfft STFT + mel filterbank)
        → Qwen3ASRAudioEncoder
        → Qwen3ASRThinkerTextModel (causal attention + MRoPE + SwiGLU MLP)
        → LM head → argmax greedy decoding
    → SRT output
```

**Weight loading**: `SafetensorsStore::from_file` → `PyTorchToBurnAdapter` (name mapping + transpose) → optional `Bf16ToF32Adapter` (metal only) → `model.load_from()`. Burn's `load_from` does NOT auto-convert dtypes between stored and target; type mismatches cause `DTypeMismatch` panics at operation time.

**VAD segmentation** (`vad.rs`): earshot `predict_f32` on 256-sample frames → threshold 0.5 → merge gaps <0.5s → split at max 30s (find nearest silence in 1s overlap window) → drop <0.5s. Segments saved as `_segments.json` alongside transcript.

**Tokenization**: The `Qwen2Tokenizer` loads from `tokenizer.json` (HuggingFace fast tokenizer format). If only `vocab.json` + `merges.txt` exist, run:
```python
from tokenizers import Tokenizer, models, pre_tokenizers, decoders
import json
bpe = models.BPE.from_file('vocab.json', 'merges.txt')
tok = Tokenizer(bpe)
tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
tok.decoder = decoders.ByteLevel()
with open('tokenizer_config.json') as f:
    for _, v in sorted(json.load(f)['added_tokens_decoder'].items(), key=lambda x: int(x[0])):
        tok.add_special_tokens([v['content']])
tok.add_special_tokens(['<timestamp>'])
tok.save('tokenizer.json')
```

**Model directory structure** expected under `-m <dir>`:
```
config.json, preprocessor_config.json, generation_config.json,
tokenizer.json, model.safetensors
```

## Key constraints

- Burn 0.21 wgpu Metal backend: supports F32, F16 but NOT BF16. Do not use `Wgpu<bf16>` on macOS.
- F16 also fails with BF16-trained weights: BF16 has 8-bit exponent (≈F32 range), F16 has 5-bit. BF16→F16 conversion causes overflow/underflow in many weight values. Stick with F32 for metal.
- Long audio (>1000 encoder tokens) causes NaN in attention softmax — always use VAD segmentation before ASR.
- **Audio encoder chunked attention**: The `Qwen3ASRAudioEncoder` splits mel into `n_window*2`=100-frame chunks, runs conv on padded batches, adds per-chunk sinusoidal position embedding (positions 0..padded_time, restarted per chunk), then applies **block-diagonal attention** via cu_seqlens (window = `padded_time * n_window_infer/chunk_size` = 50 audio tokens). A float mask with 0 within blocks and `f32::NEG_INFINITY` between blocks is applied as `attn_weights + mask`. **Global attention produces OOD features**: the first few tokens will be correct (local context), then degrade into homophone hallucinations — this is a diagnostic to check the encoder's attention masking and positional embedding logic first (not KV-cache, RoPE, or decoder attention).
- Audio must be 16kHz mono f32 PCM. `load_wav_samples` handles resampling from any source rate/channels.
- The `AlignPipeline` requires `classify_num` and `timestamp_segment_time` from `config.json`; `allow_partial(true)` is set to handle missing non-critical weights.
- When chaining burn-store adapters, use `ChainAdapter::new(a, b)` instead of `a.chain(b)`. The `.chain()` trait method can fail to resolve, causing "not an iterator" errors.
- earshot `Detector::predict_f32` takes `&[f32; 256]` (fixed-size array), not `&[f32]`. Use `.try_into().unwrap()` on 256-sample chunks.
- Transcribe output is always saved: `<input_stem>_transcript.txt` and `<input_stem>_segments.json`. The `-o` flag overrides the txt path.
- Python tokenizer conversion: use `/usr/bin/python3` if venv lacks `tokenizers`. `BPE.from_file(vocab_path, merges_path)` takes two positional args, not keyword args.
