# Issues Found During Qwen3-ASR Burn Implementation

## Open Issues (待验证)

### Metal/wgpu vs PyTorch Generation Divergence → 模型实现 Bug

**Symptom:** Same 0.6B model, Python (PyTorch CPU F32) outputs correct text, Rust (Metal/wgpu F32) outputs garbled text. Using Python's exact mel features does NOT fix the issue — generation diverges at token ~10.

**CUDA verification (2026-06-18):** Tested CUDA/BF16 Burn vs Python BF16 on the same segment:
- **Python BF16:** "这是你打开B站每天可以看到的推荐视频，不管他们在讲的内容有多么不同，它基本都有同一个东西，就是字幕..."
- **Burn CUDA BF16:** "是你打开B站，每天可以看到推荐视频。不管他们在讲内容多么，还是什么，我给你说一句，就是字幕..."

CUDA/BF16 同样存在分歧 → **模型实现中存在 bug，而非后端精度问题。**

**Next steps:**
1. 对比 Burn vs Python 的 prefill top-10 logits（已在调试中确认 top-1 token 相同，但 logit 值存在系统性偏差 ~3-4）
2. 逐层对比 audio encoder 输出（Token 0 第一值：Burn=0.021 vs Python=0.0015）
3. 重点检查：MRoPE 实现、SiLU/SwiGLU MLP、RMS/LayerNorm 数值差异

**Status:** 已确认 CUDA 也存在分歧 → 需要排查模型层实现

### qwen-asr PyPI `fix_mistral_regex` Crash

**Symptom:** `TypeError: 'ByteLevel' object does not support item assignment` when calling `Qwen3ASRModel.from_pretrained()`.

**Root cause:** qwen-asr hardcodes `fix_mistral_regex=True` in `AutoProcessor.from_pretrained()`. transformers 4.57.3+ applies regex check to all non-Split tokenizers (#44031). Qwen uses ByteLevel pre-tokenizer which doesn't support item assignment.

**Fix:** Manually removed 3 occurrences in `.venv/lib/.../qwen_asr/`. Upstream fix pending.

**Refs:** [transformers#44031](https://github.com/huggingface/transformers/issues/44031), [transformers#42299](https://github.com/huggingface/transformers/pull/42299)

**Status:** Local workaround applied (changed to `fix_mistral_regex=False` in 3 files).

### F16 Precision Not Viable on Metal

**Symptom:** `Wgpu<f16, i32>` generates empty/invalid tokens (id=198 loop). BF16→F16 weight conversion causes overflow (BF16 exponent range ≈ F32, F16 much narrower).

**Conclusion:** Metal must use F32. H16 not supported for this model's BF16-trained weights.

**Status:** Verified. Stick with F32 permanently.

### Greedy Decoding Repetition

**Symptom:** With max_new=256, segments often generate 256 tokens without hitting EOS, with repeating patterns in later tokens.

**Mitigation:** `fix_repetitions()` post-processing + bigram loop detection. Python reference also has this issue and uses `detect_and_fix_repetitions()`.

**Status:** Mitigated but not fully resolved. CUDA may improve.

---

## Resolved Issues

### 1. Missing `<|im_start|>` Token Before Assistant Turn

**File:** `src/pipeline.rs` — `build_suffix_ids()`

**Symptom:** Model generated wrong text that quickly degenerates into repetition.

**Root Cause:** The prompt suffix was `\nassistant\n` instead of the correct Qwen3 chat template `<|im_start|>assistant\n`. The model expects `<|im_start|>` (token 151644) to mark the beginning of the assistant turn. Without it, the model receives `\nassistant\n` as plain text, violating the chat template format.

**Fix:** Changed `build_suffix_ids` from:
```rust
// Wrong: [audio_end, im_end, encode("\nassistant\n")]
// Correct: [audio_end, im_end, encode("\n"), im_start, encode("assistant\n")]
```

**Reference:** `references/qwen3-asr-rs/src/inference.rs:443-448` (candle Rust reference)

---

### 2. Missing Audio Padding to 30 Seconds

**File:** `src/pipeline.rs` — `infer_segment()`

**Symptom:** Audio encoder produced 367 tokens instead of 390, mel spectrogram was 2822 frames instead of 3000.

**Root Cause:** WhisperFeatureExtractor always pads/truncates audio to exactly `n_samples=480000` (30 seconds at 16kHz) before computing the mel spectrogram. The model was trained expecting exactly 3000 mel frames per segment. My implementation passed VAD-trimmed audio directly without padding, producing variable frame counts.

**Fix:** Added padding to 480,000 samples (30s) before computing mel spectrogram:
```rust
const TARGET_SAMPLES: usize = 480_000;
let padded: Vec<f32> = if samples.len() < TARGET_SAMPLES {
    let mut v = samples.to_vec();
    v.resize(TARGET_SAMPLES, 0.0);
    v
} else { ... };
```

**Reference:** `preprocessor_config.json` (`n_samples: 480000`, `nb_max_frames: 3000`)

---

### 3. Missing `audio_pad` Placeholder Tokens in Prompt

**File:** `src/pipeline.rs` — `infer_segment()`

**Symptom:** Position encoding mismatch between text tokens and audio features.

**Root Cause:** The Qwen3 chat template places `num_audio_tokens` copies of `<|audio_pad|>` (token 151676) between `<|audio_start|>` and `<|audio_end|>`. These placeholder tokens define the positions that audio encoder features replace. The HF processor's `replace_multimodal_special_tokens` expands a single `<|audio_pad|>` to N copies based on the audio encoder's output length. My original code concatenated audio features directly without placeholder tokens.

**Fix:** Build full prompt with `audio_token_id * num_audio_tokens`, then embed the text portions and replace audio_pad positions with audio features:
```rust
prompt_ids.extend(std::iter::repeat_n(self.audio_token_id, num_audio_tokens));
// Then embed before_ids and after_ids, concat with audio_features between them
```

**References:**
- `references/qwen3-asr-rs/src/inference.rs:440-441` (candle Rust reference)
- `references/Qwen3-ASR/qwen_asr/core/transformers_backend/processing_qwen3_asr.py:154` (HF processor)

---

### 4. `<timestamp>` Token Required for ASR Pipeline

**File:** `src/tokenizer.rs`

**Symptom:** `Error: missing required special token: <timestamp>` when running `transcribe` command.

**Root Cause:** `Qwen2Tokenizer::from_dir()` unconditionally required the `<timestamp>` special token, but this token only exists in the forced-aligner's tokenizer (id 151705). The ASR model's tokenizer does not include `<timestamp>` because it's only used for word-level alignment, not transcription.

**Fix:** Made `timestamp_id` an `Option<u32>` field, loaded lazily. The `AlignPipeline` checks for the token presence with a clear error message.

---

### 5. Text Output Not Stripping `language X<asr_text>` Prefix

**File:** `src/pipeline.rs`

**Symptom:** Generated output contained raw token IDs for "language Chinese<asr_text>" before the actual transcription text.

**Root Cause:** Qwen3-ASR generates a language tag followed by `<asr_text>` (token 151704) as a delimiter before the actual transcription. The raw generated tokens need to be split at `<asr_text>` to extract only the text portion.

**Fix:** Added `extract_text()` method that finds `<asr_text>` (151704) and returns everything after it, with a string-based fallback for tokenizers that don't contain this token.

**References:**
- `references/qwen3-asr-rs/src/inference.rs:381-399` — candle `decode_result()`
- `references/Qwen3-ASR/qwen_asr/inference/utils.py:403-470` — Python `parse_asr_output()`

---

### 6. Repetition Detection UTF-8 Panic

**File:** `src/pipeline.rs` — `fix_repetitions()`

**Symptom:** Panic: `end byte index 1 is not a char boundary; it is inside '自' (bytes 0..3 of string)`

**Root Cause:** Chinese characters are multi-byte UTF-8 sequences (3 bytes). The pattern repetition detection used byte-level indexing (`&s[i..i+k]`) which can split multi-byte characters at arbitrary positions.

**Fix:** Rewrote the repetition detection to work at character level using `Vec<char>` instead of raw string slices.

---

### 7. MLP Activation Bug

**File:** `src/model.rs` — `Qwen3MLP::forward()`

**Symptom:** Model produced wrong output; MFCC/log-mel values differed from reference.

**Root Cause:** The SwiGLU activation was implemented using `sigmoid` instead of `silu` (x * sigmoid(x)). The correct implementation is `gate.mul(sigmoid(gate))` (element-wise), which was verified to be equivalent to `silu(gate)`.

**Status:** Confirmed implementation is correct — both are mathematically identical.

---

## Resolved Issues (continued)

### 8. Mel Spectrogram Mismatch with WhisperFeatureExtractor

**File:** `src/audio.rs` — `MelSpectrogram`

**Symptom:** Transcription was completely garbled (e.g., "是你打开冰箱每天可以看" instead of "这是你打开B站每天可以看到的推荐视频"). Mel spectrogram values differed from WhisperFeatureExtractor by mean diff: 0.37.

**Root Cause:** Three implementation differences from the reference:

1. **STFT center padding used zeros instead of reflection:**
   - My code padded center with zeros (`if sample_idx < 0 || sample_idx >= len → 0.0`)
   - `torch.stft` default behavior: `pad_mode='reflect'` with `center=True`
   - Fixed by adding `reflection_pad()` function that mirrors the signal at boundaries

2. **Hann window formula used Whisper-style instead of standard:**
   - My code: `0.5 - 0.5 * cos(2πi / N)` (Whisper numpy style, N=400)
   - `torch.hann_window`: `0.5 * (1 - cos(2πi / (N-1)))` (standard, N-1=399)
   - Fixed to use standard formula matching `torch.hann_window`

3. **Mel filterbank used simple Slaney scale instead of the correct Slaney with linear region:**
   - My code: `2595 * log10(1 + f/700)` (uniform Slaney)
   - Candle reference: Slaney with linear region below 1000Hz (`f_sp = 200/3`, transition at 1000Hz)
   - Fixed to match the candle reference's filterbank (same as librosa `htk=True, norm="slaney"`)

**Fix:** Rewrote `src/audio.rs` to match the candle reference implementation (`references/qwen3-asr-rs/src/mel.rs`), which uses:
- `reflection_pad()` for center padding
- Standard Hann window (`0.5 * (1 - cos(2πi/(N-1)))`)
- Slaney mel scale with linear region below 1000Hz
- Column-major power spectrum `[n_freqs × n_frames]`
- log10 normalization with Whisper clamping

**Verification:** Post-fix transcription for segment 0:
- Before: "是你打开**冰箱**每天可以吃多少？冰箱是不管放在哪..." (wrong)
- After: "是你打开**B站**，每天可以看到推荐视频。不管他们在讲内容多么..." (correct!)

**Reference:** `references/qwen3-asr-rs/src/mel.rs` lines 1-213

---

## Remaining Issues

### Generation Repetition Degeneracy

### Generation Repetition Degeneracy

**Symptom:** After the first few tokens, the model gets stuck in repetition loops (e.g., "所以，所以，所以..." or "没有见过，没有见过，没有见过..."). Most segments generate exactly 512 tokens (max_new) without hitting EOS.

**Status:** Partially mitigated by post-processing `fix_repetitions()`, but the root cause is the mel spectrogram mismatch (see above). With correct mel features, the Python reference generates ~128 tokens of clean text without any repetition.

**Note:** The Python reference applies `detect_and_fix_repetitions()` as a post-processing step, confirming that repetition can occur even with correct features in some edge cases. A generation-time `no_repeat_ngram_size=3` filter could also help.

---

## Debugging Methodology

The root cause was identified through layer-by-layer comparison:

1. **Pre-fill logits** → Top-1 token matched (id=11528, "language") but values differed by ~3.6
2. **Audio encoder output** → Different values: Token 0[0] = 0.027 (Rust) vs 0.0015 (Python)
3. **Mel spectrogram** → mean_diff = 0.37, confirming the issue is in the feature extraction stage

This narrowed down the problem to `src/audio.rs` — specifically the mel filterbank and STFT implementation.
