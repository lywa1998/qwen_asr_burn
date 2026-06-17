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
            return Ok((String::new(), vec![]));
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
        let mel_spec = self.mel_extractor.compute(samples);
        let n_mels = mel_spec.len();
        let n_frames = mel_spec[0].len();
        let flat: Vec<f32> = mel_spec.into_iter().flatten().collect();
        log::info!("Mel: {n_mels} bins x {n_frames} frames");
        let mel_tensor = Tensor::<B, 1>::from_floats(flat.as_slice(), &self.device)
            .reshape([1, n_mels, n_frames]);

        let audio_features = self.model.thinker.audio_tower.forward(mel_tensor);
        let [_, audio_len, feat_dim] = audio_features.dims();
        log::info!("Audio encoder output: {audio_len} tokens, dim={feat_dim}");

        let prefix_ids = self.build_prefix_ids();
        let suffix_ids = self.build_suffix_ids();
        let prefix_ids_tensor = int_tensor_2d::<B>(&prefix_ids, &self.device);
        let suffix_ids_tensor = int_tensor_2d::<B>(&suffix_ids, &self.device);

        let prefix_embeds = self
            .model
            .thinker
            .model
            .embed_tokens
            .forward(prefix_ids_tensor);
        let suffix_embeds = self
            .model
            .thinker
            .model
            .embed_tokens
            .forward(suffix_ids_tensor);
        let current_embeds = Tensor::cat(vec![prefix_embeds, audio_features, suffix_embeds], 1);

        let max_new = 256;
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

            if self.eos_token_ids.contains(&next_token_id) || next_token_id == self.tokenizer.im_end
            {
                break;
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

            if step < 2 {
                log::info!("Token {step}: id={next_token_id}");
            }
        }

        let text = self.tokenizer.decode(&generated);
        log::info!("Segment: {} tokens → {}", generated.len(), text.trim());
        Ok(text.trim().to_string())
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
        let mut suffix_ids = vec![self.audio_end_token_id, self.tokenizer.im_end];
        suffix_ids.extend(self.tokenizer.encode("\nassistant\n"));
        suffix_ids
    }

    #[allow(dead_code)]
    fn build_audio_token_ids(&self, audio_len: usize) -> Vec<u32> {
        std::iter::repeat_n(self.audio_token_id, audio_len).collect()
    }
}

fn int_tensor_2d<B: Backend>(ids: &[u32], device: &B::Device) -> Tensor<B, 2, Int> {
    let ints: Vec<i32> = ids.iter().map(|&id| id as i32).collect();
    Tensor::<B, 1, Int>::from_ints(ints.as_slice(), device).unsqueeze_dim::<2>(0)
}
