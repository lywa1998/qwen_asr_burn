# qwen-asr-burn

Burn-based Qwen3-ASR inference.

## What this repo contains
- `src/main.rs` — CUDA CLI entrypoint
- `src/pipeline.rs` — ASR orchestration and generation loop
- `src/model.rs` — Burn model implementation
- `src/audio.rs` — mel feature extraction
- `src/tokenizer.rs` — tokenizer loading and special-token handling
- `src/config.rs` — model/config deserialization

## Build
```bash
cargo build
```

## Run
```bash
cargo run -- transcribe <input.wav>
```

Optional model directory:
```bash
cargo run -- --model-dir Qwen3-ASR-0.6B transcribe <input.wav>
```

## Notes
- The binary currently targets CUDA + BF16.
- Model assets are expected in the model directory (for example `Qwen3-ASR-0.6B/`).
- `tokenizer.json`, `config.json`, `generation_config.json`, and `preprocessor_config.json` should be present alongside the model weights.
