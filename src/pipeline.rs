use burn::tensor::backend::Backend;
use burn::tensor::{Int, Tensor};

use crate::audio::MelSpectrogram;
use crate::config::{GenerationConfig, ModelConfig, PreprocessorConfig};
use crate::model::{self, create_mrope, KvCache, Qwen3ASR, Qwen3ASRConfig};
use crate::tokenizer::Qwen2Tokenizer;

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
            use burn_store::{ModuleSnapshot, PyTorchToBurnAdapter, SafetensorsStore};
            let mut store =
                SafetensorsStore::from_file(&weights_path).with_from_adapter(PyTorchToBurnAdapter);
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

    pub fn transcribe(&self, wav_path: &str) -> anyhow::Result<String> {
        log::info!("Loading audio: {}", wav_path);

        let mel_spec = self.mel_extractor.compute_from_wav(wav_path)?;
        let n_mels = mel_spec.len();
        let n_frames = mel_spec[0].len();
        let flat: Vec<f32> = mel_spec.into_iter().flatten().collect();
        log::info!(
            "Mel: {n_mels} bins x {n_frames} frames, mean={:.4}, min={:.4}, max={:.4}",
            flat.iter().sum::<f32>() / flat.len() as f32,
            flat.iter().fold(f32::MAX, |a, &b| a.min(b)),
            flat.iter().fold(f32::MIN, |a, &b| a.max(b))
        );
        let mel_tensor = Tensor::<B, 1>::from_floats(flat.as_slice(), &self.device)
            .reshape([1, n_mels, n_frames]);

        let audio_features = self.model.thinker.audio_tower.forward(mel_tensor);
        let [_, audio_len, feat_dim] = audio_features.dims();
        log::info!("Audio encoder output: {audio_len} tokens, dim={feat_dim}");

        let feat_flat = audio_features.clone().flatten::<1>(0, 2);
        let feat_data = feat_flat.into_data();
        let n_floats = feat_data.bytes.len() / 4;
        let mut sum = 0.0f64;
        let mut sum_sq = 0.0f64;
        let mut lo = f32::MAX;
        let mut hi = f32::MIN;
        for i in 0..n_floats.min(1000) {
            let v = f32::from_le_bytes([
                feat_data.bytes[i * 4],
                feat_data.bytes[i * 4 + 1],
                feat_data.bytes[i * 4 + 2],
                feat_data.bytes[i * 4 + 3],
            ]);
            sum += v as f64;
            sum_sq += (v as f64) * (v as f64);
            lo = lo.min(v);
            hi = hi.max(v);
        }
        let n = n_floats.min(1000) as f64;
        let mean = sum / n;
        let std = (sum_sq / n - mean * mean).sqrt();
        log::info!("Audio features stats: mean={mean:.4}, std={std:.4}, range=[{lo:.4}, {hi:.4}]");

        let feat_data = audio_features.clone().flatten::<1>(0, 2).into_data();
        let bytes = &feat_data.bytes;
        if bytes.len() >= 4 {
            let f0 = f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
            let fn_ = f32::from_le_bytes([
                bytes[bytes.len() - 4],
                bytes[bytes.len() - 3],
                bytes[bytes.len() - 2],
                bytes[bytes.len() - 1],
            ]);
            log::info!("Audio features: first={f0:.4}, last={fn_:.4}");
        }

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
                log::info!("EOS token {next_token_id} at step {step}");
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

            if step < 3 {
                let raw = &token_data.bytes;
                log::info!(
                    "Token {step}: bytes={:02x?}, id={next_token_id}, text={}",
                    &raw[..4.min(raw.len())],
                    self.tokenizer.decode(&[next_token_id])
                );
            }
        }

        let text = self.tokenizer.decode(&generated);
        log::info!("Generated {} tokens, text: {}", generated.len(), text);
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
