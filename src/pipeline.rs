use burn::tensor::backend::Backend;
use burn::tensor::{Int, Tensor};

#[cfg(feature = "metal")]
use std::rc::Rc;
#[cfg(feature = "metal")]
use burn::tensor::DType;
#[cfg(feature = "metal")]
use burn_store::{ModuleAdapter, TensorSnapshot};

use crate::audio::{self, MelSpectrogram};
use crate::config::{GenerationConfig, ModelConfig, PreprocessorConfig};
use crate::model::{self, create_mrope, KvCache, Qwen3ASR, Qwen3ASRConfig};
use crate::tokenizer::Qwen2Tokenizer;
use crate::vad;

/// Converts BF16 weights to F32 during loading (for backends that don't support BF16).
#[cfg(feature = "metal")]
#[derive(Clone)]
pub(crate) struct Bf16ToF32Adapter;

#[cfg(feature = "metal")]
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

pub struct AsrPipeline<B: Backend> {
    model: Qwen3ASR<B>,
    tokenizer: Qwen2Tokenizer,
    mel_extractor: MelSpectrogram,
    mrope: model::MRoPE,
    device: B::Device,
    eos_token_ids: Vec<u32>,
    audio_start_token_id: u32,
    audio_end_token_id: u32,
    audio_token_id: u32,
}

impl<B: Backend> AsrPipeline<B> {
    pub fn new(model_dir: &str, device: B::Device) -> anyhow::Result<Self> {
        let model_config = ModelConfig::from_dir(model_dir)?;
        let preprocessor_config = PreprocessorConfig::from_dir(model_dir)?;
        let generation_config = GenerationConfig::from_dir(model_dir)?;

        let audio_config = model_config.thinker_config.audio_config.clone();
        let text_config = model_config.thinker_config.text_config.clone();

        let burn_config = Qwen3ASRConfig::from_configs(audio_config, text_config.clone());
        let mut model = burn_config.init(&device);

        let weights_path = format!("{}/model.safetensors", model_dir);
        {
            use burn_store::{ChainAdapter, ModuleSnapshot, PyTorchToBurnAdapter, SafetensorsStore};
            #[cfg(feature = "metal")]
            let adapter = ChainAdapter::new(PyTorchToBurnAdapter, Bf16ToF32Adapter);
            #[cfg(not(feature = "metal"))]
            let adapter = PyTorchToBurnAdapter;
            let mut store = SafetensorsStore::from_file(&weights_path)
                .with_from_adapter(adapter);
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

    pub fn transcribe(&self, wav_path: &str) -> anyhow::Result<(String, Vec<vad::VoiceSegment>)> {
        log::info!("Loading audio: {}", wav_path);

        let samples = audio::load_wav_samples(wav_path)?;
        let segments = vad::detect_segments(&samples);

        if segments.is_empty() {
            // Fallback: no voice detected, still try full audio
            let text = self.infer_segment(&samples)?;
            log::info!("Transcription complete: {} chars", text.len());
            return Ok((text, vec![]));
        }

        log::info!("Transcribing {} voice segments", segments.len());
        let mut texts: Vec<String> = Vec::new();

        for (i, seg) in segments.iter().enumerate() {
            let start_sample = (seg.start_secs * 16_000.0) as usize;
            let end_sample = (seg.end_secs * 16_000.0) as usize;
            let seg_samples = &samples[start_sample.min(samples.len())..end_sample.min(samples.len())];

            log::info!(
                "Segment {}/{}: {:.2}s-{:.2}s ({} samples)",
                i + 1,
                segments.len(),
                seg.start_secs,
                seg.end_secs,
                seg_samples.len()
            );

            let text = self.infer_segment(seg_samples)?;
            texts.push(text);
        }

        let combined = texts.join("\n");
        log::info!("Transcription complete: {} segments, {} chars", texts.len(), combined.len());
        Ok((combined, segments))
    }

    /// Run ASR inference on a single audio segment (f32 16kHz mono samples).
    fn infer_segment(&self, samples: &[f32]) -> anyhow::Result<String> {
        // Pad audio to exactly 30 seconds (480000 samples @ 16kHz) to match
        // WhisperFeatureExtractor behavior. The model expects exactly 3000 mel
        // frames per segment.
        const TARGET_SAMPLES: usize = 480_000; // 30s @ 16kHz
        let padded: Vec<f32> = if samples.len() < TARGET_SAMPLES {
            let mut v = samples.to_vec();
            v.resize(TARGET_SAMPLES, 0.0);
            v
        } else if samples.len() > TARGET_SAMPLES {
            samples[..TARGET_SAMPLES].to_vec()
        } else {
            samples.to_vec()
        };
        let mel_spec = self.mel_extractor.compute(&padded);
        let n_mels = mel_spec.len();
        let n_frames = mel_spec[0].len();
        let flat: Vec<f32> = mel_spec.into_iter().flatten().collect();
        log::info!("Mel: {n_mels} bins x {n_frames} frames (padded to {TARGET_SAMPLES} samples)");
        let mel_tensor = Tensor::<B, 1>::from_floats(flat.as_slice(), &self.device)
            .reshape([1, n_mels, n_frames]);

        let audio_features = self.model.thinker.audio_tower.forward(mel_tensor);
        let [_, num_audio_tokens, feat_dim] = audio_features.dims();
        log::info!("Audio encoder output: {num_audio_tokens} tokens, dim={feat_dim}");


        // Build full prompt: prefix + audio_pad * N + suffix, then replace
        // audio_pad embeddings with audio encoder features.
        // Matches Qwen3 chat template:
        //   <|im_start|>system\n...<|im_end|>\n<|im_start|>user\n<|audio_start|>
        //   <|audio_pad|> * N
        //   <|audio_end|><|im_end|>\n<|im_start|>assistant\n
        let prefix_ids = self.build_prefix_ids();
        let suffix_ids = self.build_suffix_ids();
        let before_len = prefix_ids.len();
        let after_start = before_len + num_audio_tokens;

        let mut prompt_ids = prefix_ids;
        prompt_ids.extend(std::iter::repeat_n(self.audio_token_id, num_audio_tokens));
        prompt_ids.extend(suffix_ids);

        let before_ids = &prompt_ids[..before_len];
        let after_ids = &prompt_ids[after_start..];

        let before_t = int_tensor_2d::<B>(before_ids, &self.device);
        let after_t = int_tensor_2d::<B>(after_ids, &self.device);
        let before_embeds = self.model.thinker.model.embed_tokens.forward(before_t);
        let after_embeds = self.model.thinker.model.embed_tokens.forward(after_t);

        let current_embeds =
            Tensor::cat(vec![before_embeds, audio_features, after_embeds], 1);

        let max_new = 512;
        let seq_len = current_embeds.dims()[1];
        let total_positions: Vec<usize> = (0..(seq_len + max_new)).collect();
        let mut kv_cache = KvCache::new(self.model.thinker.model.layers.len());
        let (prefill_cos, prefill_sin) = self
            .mrope
            .compute_cos_sin_from_positions(&total_positions[..seq_len], &self.device);
        let causal_mask = model::create_causal_mask::<B>(seq_len, 0, &self.device);
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
                    // Detected bigram loop → stop
                    break;
                }
            }

            generated.push(next_token_id);
            let next_ids = int_tensor_2d::<B>(&[next_token_id], &self.device);
            let next_embed = self.model.thinker.model.embed_tokens.forward(next_ids);
            let (step_cos, step_sin) = self
                .mrope
                .compute_cos_sin_from_positions(&[current_pos], &self.device);
            let step_mask = model::create_causal_mask::<B>(1, kv_cache.seq_len(), &self.device);
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

            if step < 4 {
                log::info!("Token {step}: id={next_token_id}");
            }
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
        let threshold: usize = 20;
        let chars: Vec<char> = text.chars().collect();
        let n = chars.len();
        if n < threshold * 2 {
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
            if count > threshold {
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
        let threshold = 15usize;
        let max_pat = 8usize;
        let s = fixed_chars;
        let n = s.len();
        if n < threshold * 2 {
            return s.iter().collect();
        }

        let mut result: Vec<char> = Vec::with_capacity(n);
        let mut i = 0;
        while i + threshold * 2 <= n {
            let mut found = false;
            for k in 1..=max_pat {
                if i + k * threshold > n {
                    break;
                }
                let pattern = &s[i..i + k];
                let mut all_match = true;
                for rep in 1..threshold {
                    let start = i + rep * k;
                    if &s[start..start + k] != pattern {
                        all_match = false;
                        break;
                    }
                }
                if all_match {
                    result.extend_from_slice(pattern);
                    i += threshold * k;
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

    fn build_prefix_ids(&self) -> Vec<u32> {
        let mut prefix_ids = vec![self.tokenizer.im_start];
        prefix_ids.extend(self.tokenizer.encode("system\n"));
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

fn int_tensor_2d<B: Backend>(ids: &[u32], device: &B::Device) -> Tensor<B, 2, Int> {
    let ints: Vec<i32> = ids.iter().map(|&id| id as i32).collect();
    Tensor::<B, 1, Int>::from_ints(ints.as_slice(), device).unsqueeze_dim::<2>(0)
}
