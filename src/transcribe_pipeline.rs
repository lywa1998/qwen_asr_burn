use std::rc::Rc;

use burn::tensor::DType;
use burn::tensor::{Device, Int, Tensor};
use burn_store::{ModuleAdapter, TensorSnapshot};

use crate::models::qwen_asr::config::{GenerationConfig, ModelConfig, PreprocessorConfig};
use crate::models::qwen_asr::{self as model, create_mrope, KvCache, Qwen3ASR, Qwen3ASRConfig};
use crate::utils::audio::{self, MelSpectrogram};
use crate::utils::tokenizer::Qwen2Tokenizer;
use crate::utils::vad;

/// Converts BF16 weights to F32 during loading (for backends that don't support BF16).
#[derive(Clone)]
pub(crate) struct Bf16ToF32Adapter;

impl ModuleAdapter for Bf16ToF32Adapter {
    fn adapt(&self, snapshot: &TensorSnapshot) -> TensorSnapshot {
        if snapshot.dtype != DType::BF16 {
            return snapshot.clone();
        }
        let original = snapshot.clone_data_fn();
        let cast = Rc::new(move || {
            let data = original()?;
            Ok(data.convert_dtype(DType::F32))
        });
        TensorSnapshot::from_closure(
            cast,
            DType::F32,
            snapshot.shape.clone(),
            snapshot.path_stack.clone().unwrap_or_default(),
            snapshot.container_stack.clone().unwrap_or_default(),
            snapshot.tensor_id.unwrap_or_default(),
        )
    }

    fn clone_box(&self) -> Box<dyn ModuleAdapter> {
        Box::new(self.clone())
    }
}

pub struct TranscribePipeline {
    model: Qwen3ASR,
    tokenizer: Qwen2Tokenizer,
    mel_extractor: MelSpectrogram,
    mrope: model::Qwen3ASRMRoPE,
    device: Device,
    eos_token_ids: Vec<u32>,
    audio_start_token_id: u32,
    audio_end_token_id: u32,
    audio_token_id: u32,
}

impl TranscribePipeline {
    pub fn new(model_dir: &str, device: Device) -> anyhow::Result<Self> {
        let model_config = ModelConfig::from_dir(model_dir)?;
        let preprocessor_config = PreprocessorConfig::from_dir(model_dir)?;
        let generation_config = GenerationConfig::from_dir(model_dir)?;

        let audio_config = model_config.thinker_config.audio_config.clone();
        let text_config = model_config.thinker_config.text_config.clone();

        let burn_config = Qwen3ASRConfig::from_configs(audio_config, text_config.clone());
        let mut model = burn_config.init(&device);

        let weights_path = format!("{}/model.safetensors", model_dir);
        {
            use burn_store::{
                ChainAdapter, ModuleSnapshot, PyTorchToBurnAdapter, SafetensorsStore,
            };
            // Model weights are stored as BF16.
            #[cfg(not(feature = "bf16"))]
            let adapter = ChainAdapter::new(PyTorchToBurnAdapter, Bf16ToF32Adapter);
            #[cfg(feature = "bf16")]
            let adapter = PyTorchToBurnAdapter;
            let mut store = SafetensorsStore::from_file(&weights_path).with_from_adapter(adapter);
            let result = model.load_from(&mut store)?;
            log::info!(
                "Weight loading: {} applied, {} errors",
                result.applied.len(),
                result.errors.len()
            );
        }

        let tokenizer = Qwen2Tokenizer::from_dir(model_dir)?;
        let mel_extractor = MelSpectrogram::new(
            preprocessor_config.n_fft,
            preprocessor_config.hop_length,
            preprocessor_config.feature_size,
            16000,
        );
        let mrope = create_mrope(&text_config);

        Ok(Self {
            model,
            tokenizer,
            mel_extractor,
            mrope,
            device,
            eos_token_ids: generation_config.eos_token_id,
            audio_start_token_id: model_config.thinker_config.audio_start_token_id,
            audio_end_token_id: model_config.thinker_config.audio_end_token_id,
            audio_token_id: model_config.thinker_config.audio_token_id,
        })
    }

    pub fn transcribe(
        &self,
        wav_path: &str,
        language: Option<&str>,
        context: &str,
    ) -> anyhow::Result<(Vec<String>, Vec<vad::VoiceSegment>)> {
        log::info!("Loading audio: {}", wav_path);

        let samples = audio::load_wav_samples(wav_path)?;
        let segments = vad::detect_segments(&samples);

        if segments.is_empty() {
            let audio_features = self.preprocess_audio(&samples)?;
            let text = self.infer_segment(&audio_features, language, context)?;
            return Ok((vec![text], vec![]));
        }

        log::info!("Transcribing {} voice segments", segments.len());

        // Phase 1: extract + pad audio for all segments, compute mel, batch through
        // audio encoder. The audio encoder is non-autoregressive so batching
        // significantly reduces total wall time.
        let batch_size = segments.len();
        let mut all_mel_flat: Vec<f32> = Vec::with_capacity(batch_size * 128 * 3000);
        for seg in &segments {
            let start_sample = (seg.start_secs * 16_000.0) as usize;
            let end_sample = (seg.end_secs * 16_000.0) as usize;
            let seg_samples =
                &samples[start_sample.min(samples.len())..end_sample.min(samples.len())];
            let padded = audio::pad_to_30s(seg_samples);
            let mel = self.mel_extractor.compute(&padded);
            all_mel_flat.extend(mel.into_iter().flatten());
        }
        log::info!(
            "Batch audio encoding: {} segments, {} mel frames",
            batch_size,
            all_mel_flat.len() / 128
        );
        // Load WhisperFE mel directly for definitive comparison
        let use_whisper_mel = std::fs::read("/tmp/whisper_mel.bin")
            .map(|bytes| {
                let floats: Vec<f32> = bytes
                    .chunks_exact(4)
                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect();
                floats
            })
            .ok()
            .filter(|f| f.len() == 128 * 3000);

        if let Some(ref whisper_flat) = use_whisper_mel {
            log::info!("Using WhisperFE mel directly (bypassing Rust mel computation)");
            // Run audio encoder on WhisperFE mel
            let mel_tensor = Tensor::<1>::from_floats(whisper_flat.as_slice(), &self.device)
                .reshape([batch_size, 128, 3000]);
            let all_audio_features = self.model.thinker.audio_tower.forward(mel_tensor);
            let [_, num_audio_tokens, feat_dim] = all_audio_features.dims();

            // Debug audio encoder output
            {
                let af0 =
                    all_audio_features
                        .clone()
                        .slice([0..1, 0..num_audio_tokens, 0..feat_dim]);
                let flat = af0
                    .reshape([num_audio_tokens * feat_dim])
                    .cast(burn::tensor::DType::F32)
                    .into_data();
                let vals: &[f32] = flat.as_slice().unwrap();
                let mean = vals.iter().sum::<f32>() / vals.len() as f32;
                let std = (vals.iter().map(|v| (v - mean).powi(2)).sum::<f32>()
                    / vals.len() as f32)
                    .sqrt();
                log::info!("Audio enc (WhisperFE mel): tokens={}, token0[0..4]={:?}, mean={:.4}, std={:.4}",
                    num_audio_tokens, &vals[..4], mean, std);
            }

            // Generate text using WhisperFE mel
            let af = all_audio_features;
            let mut texts: Vec<String> = Vec::new();
            for (i, seg) in segments.iter().enumerate() {
                let audio_features = af
                    .clone()
                    .slice([i..i + 1, 0..num_audio_tokens, 0..feat_dim]);
                log::info!(
                    "Segment {}/{}: {:.2}s-{:.2}s",
                    i + 1,
                    batch_size,
                    seg.start_secs,
                    seg.end_secs
                );
                let text = self.infer_segment(&audio_features, None, "")?;
                texts.push(text);
            }
            return Ok((texts, segments));
        }

        let mel_tensor = Tensor::<1>::from_floats(all_mel_flat.as_slice(), &self.device)
            .reshape([batch_size, 128, 3000]);
        let all_audio_features = self.model.thinker.audio_tower.forward(mel_tensor);
        let [_, num_audio_tokens, feat_dim] = all_audio_features.dims();

        // Debug audio encoder output (compare with Python reference)
        {
            let af0 = all_audio_features
                .clone()
                .slice([0..1, 0..num_audio_tokens, 0..feat_dim]);
            let flat = af0
                .reshape([num_audio_tokens * feat_dim])
                .cast(burn::tensor::DType::F32)
                .into_data();
            let vals: &[f32] = flat.as_slice().unwrap();
            let mean = vals.iter().sum::<f32>() / vals.len() as f32;
            let std =
                (vals.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / vals.len() as f32).sqrt();
            log::info!(
                "Audio enc: tokens={}, token0[0..4]={:?}, mean={:.4}, std={:.4}",
                num_audio_tokens,
                &vals[..4],
                mean,
                std
            );
        }

        // Phase 2: text generation per segment (autoregressive, cannot batch)
        // Note: Tensor::clone() is shallow (Arc ref-count), so cloning the
        // full batch for each slice is cheap — no data copy.
        let mut texts: Vec<String> = Vec::new();
        let af = all_audio_features;
        for (i, seg) in segments.iter().enumerate() {
            let audio_features = af
                .clone()
                .slice([i..i + 1, 0..num_audio_tokens, 0..feat_dim]);
            log::info!(
                "Segment {}/{}: {:.2}s-{:.2}s",
                i + 1,
                batch_size,
                seg.start_secs,
                seg.end_secs,
            );
            let text = self.infer_segment(&audio_features, language, context)?;
            texts.push(text);
        }

        Ok((texts, segments))
    }

    /// Compute mel + audio encoder features for raw samples. Used when VAD
    /// finds no voice segments (full-audio fallback).
    fn preprocess_audio(&self, samples: &[f32]) -> anyhow::Result<Tensor<3>> {
        let padded = audio::pad_to_30s(samples);
        let mel = self.mel_extractor.compute(&padded);
        let (n_mels, n_frames) = (mel.len(), mel[0].len());
        let flat: Vec<f32> = mel.into_iter().flatten().collect();

        // Load WhisperFE mel for comparison if available
        if let Ok(whisper_bytes) = std::fs::read("/tmp/whisper_mel.bin") {
            if whisper_bytes.len() == n_mels * n_frames * 4 {
                let whisper_f32: Vec<f32> = whisper_bytes
                    .chunks_exact(4)
                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect();
                let mut max_diff = 0.0f32;
                let mut sum_diff = 0.0f64;
                for (&a, &b) in flat.iter().zip(whisper_f32.iter()) {
                    let d = (a - b).abs();
                    max_diff = max_diff.max(d);
                    sum_diff += d as f64;
                }
                log::info!(
                    "Mel diff (Rust vs WhisperFE): max={:.4}, mean={:.4}",
                    max_diff,
                    sum_diff / flat.len() as f64
                );
            }
        }

        let mel_tensor =
            Tensor::<1>::from_floats(flat.as_slice(), &self.device).reshape([1, n_mels, n_frames]);
        let af = self.model.thinker.audio_tower.forward(mel_tensor);

        // Compare audio encoder output with Python reference
        {
            let [_, num_tokens, feat_dim] = af.dims();
            let f32_data = af
                .clone()
                .reshape([num_tokens * feat_dim])
                .cast(burn::tensor::DType::F32)
                .into_data();
            let vals: &[f32] = f32_data.as_slice().unwrap();
            log::info!(
                "Audio enc: {} tokens, dim={}, token0[0..4]={:?}, mean={:.4}, std={:.4}",
                num_tokens,
                feat_dim,
                &vals[..4],
                vals.iter().sum::<f32>() / vals.len() as f32,
                (vals
                    .iter()
                    .map(|v| (v - vals.iter().sum::<f32>() / vals.len() as f32).powi(2))
                    .sum::<f32>()
                    / vals.len() as f32)
                    .sqrt(),
            );
        }

        Ok(af)
    }

    /// Run text generation on a single segment with pre-computed audio features.
    fn infer_segment(
        &self,
        audio_features: &Tensor<3>,
        language: Option<&str>,
        context: &str,
    ) -> anyhow::Result<String> {
        let [_, num_audio_tokens, _feat_dim] = audio_features.dims();

        // Build full prompt: prefix + audio_pad * N + suffix, then replace
        // audio_pad embeddings with audio encoder features.
        // Matches Qwen3 chat template:
        //   <|im_start|>system\n...<|im_end|>\n<|im_start|>user\n<|audio_start|>
        //   <|audio_pad|> * N
        //   <|audio_end|><|im_end|>\n<|im_start|>assistant\n
        let prefix_ids = self.build_prefix_ids(context);
        let mut suffix_ids = self.build_suffix_ids();

        // When language is forced, append "language X<asr_text>" to the prompt
        // so the model skips language detection and outputs only the transcription.
        if let Some(lang) = language {
            let force_text = format!("language {}<asr_text>", lang);
            suffix_ids.extend(self.tokenizer.encode(&force_text));
        }

        let before_len = prefix_ids.len();
        let after_start = before_len + num_audio_tokens;

        let mut prompt_ids = prefix_ids;
        prompt_ids.extend(std::iter::repeat_n(self.audio_token_id, num_audio_tokens));
        prompt_ids.extend(suffix_ids);

        let before_ids = &prompt_ids[..before_len];
        let after_ids = &prompt_ids[after_start..];

        let before_t = int_tensor_2d(before_ids, &self.device);
        let after_t = int_tensor_2d(after_ids, &self.device);
        let before_embeds = self.model.thinker.model.embed_tokens.forward(before_t);
        let after_embeds = self.model.thinker.model.embed_tokens.forward(after_t);

        let current_embeds =
            Tensor::cat(vec![before_embeds, audio_features.clone(), after_embeds], 1);

        let max_new = 256;
        let seq_len = current_embeds.dims()[1];
        let total_positions: Vec<usize> = (0..(seq_len + max_new)).collect();
        let mut kv_cache = KvCache::new(self.model.thinker.model.layers.len());
        let (prefill_cos, prefill_sin) = self
            .mrope
            .compute_cos_sin_from_positions(&total_positions[..seq_len], &self.device);
        let causal_mask = model::create_causal_mask(seq_len, 0, &self.device);
        let hidden_states = self.model.thinker.model.forward_embeds(
            current_embeds,
            &prefill_cos,
            &prefill_sin,
            Some(causal_mask),
            Some(&mut kv_cache),
        );
        let logits = self.model.thinker.lm_head.forward(hidden_states);
        let vocab_size = logits.dims()[2];
        let mut last_logits = logits.narrow(1, seq_len - 1, 1).reshape([1, vocab_size]);

        let mut generated: Vec<u32> = Vec::new();
        let mut current_pos = seq_len;
        for step in 0..max_new {
            let token_data = last_logits.argmax(1).reshape([1]).into_data();
            let next_token_id = i32::from_le_bytes([
                token_data.bytes[0],
                token_data.bytes[1],
                token_data.bytes[2],
                token_data.bytes[3],
            ]) as u32;

            // EOS tokens per generation_config.json: 151643 (<|endoftext|>), 151645 (<|im_end|>)
            if self.eos_token_ids.contains(&next_token_id) || next_token_id == self.tokenizer.im_end
            {
                break;
            }

            // Simple repetition detection: if the last 5 tokens form the same
            // 2-token bigram, stop generation to prevent loops.
            if generated.len() >= 10 {
                let last5: Vec<u32> = generated[generated.len() - 5..].to_vec();
                let (a, b, c, d, e) = (last5[0], last5[1], last5[2], last5[3], last5[4]);
                if a == c && c == e && b == d && a == next_token_id {
                    break;
                }
            }

            generated.push(next_token_id);
            let next_ids = int_tensor_2d(&[next_token_id], &self.device);
            let next_embed = self.model.thinker.model.embed_tokens.forward(next_ids);
            let (step_cos, step_sin) = self
                .mrope
                .compute_cos_sin_from_positions(&[current_pos], &self.device);
            let step_mask = model::create_causal_mask(1, kv_cache.seq_len(), &self.device);
            let hidden_states = self.model.thinker.model.forward_embeds(
                next_embed,
                &step_cos,
                &step_sin,
                Some(step_mask),
                Some(&mut kv_cache),
            );
            let logits = self.model.thinker.lm_head.forward(hidden_states);
            last_logits = logits.reshape([1, vocab_size]);
            current_pos += 1;

            // Check for pattern repetition every 16 steps
            if step > 0 && step % 16 == 0 && generated.len() >= 20 {
                let half = generated.len() / 2;
                let first = &generated[..half];
                let second = &generated[half..half * 2];
                if first == second && first.len() > 3 {
                    // Full pattern repeat → stop
                    log::info!("Pattern repetition detected at step {step}, stopping");
                    generated.truncate(half);
                    break;
                }
            }
        }

        let mut text = self.extract_text(&generated);
        text = self.fix_repetitions(&text);
        log::info!("Segment: {} tokens → {}", generated.len(), text.trim());
        Ok(text.trim().to_string())
    }

    /// Extract transcription text from generated tokens, stripping the
    /// "language X<asr_text>" prefix that Qwen3-ASR automatically emits.
    fn extract_text(&self, generated: &[u32]) -> String {
        // <asr_text> token id (151704) — the separator between language tag and actual text.
        // This token is only in the forced-aligner's tokenizer.json, so fall back
        // to string-based parsing when the token is not in this tokenizer's vocab.
        const ASR_TEXT_SEP_ID: u32 = 151704;
        if let Some(sep_pos) = generated.iter().position(|&id| id == ASR_TEXT_SEP_ID) {
            // Split: before separator = language tag, after = actual text
            let text_ids = &generated[sep_pos + 1..];
            self.tokenizer.decode(text_ids)
        } else {
            // Fallback: try string-based parsing
            let raw = self.tokenizer.decode(generated).trim().to_string();
            if let Some(rest) = raw.strip_prefix("language ") {
                if let Some(pos) = rest.find("<asr_text>") {
                    return rest[pos + "<asr_text>".len()..].trim().to_string();
                }
                // Find first non-alphabetic char as language/text boundary
                if let Some(cut) = rest.find(|c: char| c.is_whitespace() || !c.is_alphabetic()) {
                    if cut > 0 {
                        return rest[cut..].trim().to_string();
                    }
                }
            }
            raw
        }
    }

    /// Fix repetition loops common in greedy ASR decoding.
    /// Ported from Qwen3-ASR Python reference (detect_and_fix_repetitions).
    fn fix_repetitions(&self, text: &str) -> String {
        let char_repeat_threshold: usize = 20;
        let chars: Vec<char> = text.chars().collect();
        let n = chars.len();
        if n < char_repeat_threshold * 2 {
            return text.to_string();
        }

        // Phase 1: fix single-character repeats
        let mut fixed_chars: Vec<char> = Vec::with_capacity(n);
        let mut i = 0;
        while i < n {
            let mut count = 1;
            while i + count < n && chars[i + count] == chars[i] {
                count += 1;
            }
            if count > char_repeat_threshold {
                fixed_chars.push(chars[i]);
                i += count;
            } else {
                for j in 0..count {
                    fixed_chars.push(chars[i + j]);
                }
                i += count;
            }
        }

        // Phase 2: fix pattern repeats at character level
        let pattern_repeat_threshold = 15usize;
        let max_pat = 8usize;
        let s = fixed_chars;
        let n = s.len();
        if n < pattern_repeat_threshold * 2 {
            return s.iter().collect();
        }

        let mut result: Vec<char> = Vec::with_capacity(n);
        let mut i = 0;
        while i + pattern_repeat_threshold * 2 <= n {
            let mut found = false;
            for k in 1..=max_pat {
                if i + k * pattern_repeat_threshold > n {
                    break;
                }
                let pattern = &s[i..i + k];
                let mut all_match = true;
                for rep in 1..pattern_repeat_threshold {
                    let start = i + rep * k;
                    if &s[start..start + k] != pattern {
                        all_match = false;
                        break;
                    }
                }
                if all_match {
                    result.extend_from_slice(pattern);
                    i += pattern_repeat_threshold * k;
                    found = true;
                    break;
                }
            }
            if !found {
                if i < n {
                    result.push(s[i]);
                }
                i += 1;
            }
        }
        if i < n {
            result.extend_from_slice(&s[i..]);
        }

        result.iter().collect()
    }

    fn build_prefix_ids(&self, context: &str) -> Vec<u32> {
        let mut prefix_ids = vec![self.tokenizer.im_start];
        prefix_ids.extend(self.tokenizer.encode("system\n"));
        if !context.is_empty() {
            prefix_ids.extend(self.tokenizer.encode(context));
        }
        prefix_ids.push(self.tokenizer.im_end);
        prefix_ids.extend(self.tokenizer.encode("\n"));
        prefix_ids.push(self.tokenizer.im_start);
        prefix_ids.extend(self.tokenizer.encode("user\n"));
        prefix_ids.push(self.audio_start_token_id);
        prefix_ids
    }

    fn build_suffix_ids(&self) -> Vec<u32> {
        // Matches Qwen3 chat template: <|audio_end|><|im_end|>\n<|im_start|>assistant\n
        let mut suffix_ids = vec![self.audio_end_token_id, self.tokenizer.im_end];
        suffix_ids.extend(self.tokenizer.encode("\n"));
        suffix_ids.push(self.tokenizer.im_start);
        suffix_ids.extend(self.tokenizer.encode("assistant\n"));
        suffix_ids
    }
}

fn int_tensor_2d(ids: &[u32], device: &Device) -> Tensor<2, Int> {
    let ints: Vec<i32> = ids.iter().map(|&id| id as i32).collect();
    Tensor::<1, Int>::from_ints(ints.as_slice(), device).unsqueeze_dim::<2>(0)
}
