use burn::tensor::{Device, Int, Tensor};

use crate::models::qwen_asr::config::{ForcedAlignerConfig, PreprocessorConfig};
use crate::models::qwen_asr::{self as model, create_mrope, Qwen3ASR, Qwen3ASRConfig};
#[cfg(feature = "metal")]
use crate::transcribe_pipeline::Bf16ToF32Adapter;
use crate::utils::audio::{self, MelSpectrogram};
use crate::utils::text_processor::{self, TimestampItem};
use crate::utils::tokenizer::Qwen2Tokenizer;

pub struct AlignPipeline {
    model: Qwen3ASR,
    tokenizer: Qwen2Tokenizer,
    mel_extractor: MelSpectrogram,
    mrope: model::Qwen3ASRMRoPE,
    device: Device,
    timestamp_segment_time: u32,
}

impl AlignPipeline {
    pub fn new(model_dir: &str, device: Device) -> anyhow::Result<Self> {
        let aligner_config = ForcedAlignerConfig::from_dir(model_dir)?;
        let thinker_cfg = &aligner_config.thinker_config;
        let audio_config = thinker_cfg.audio_config.clone();
        let text_config = thinker_cfg.text_config.clone();
        let classify_num = thinker_cfg.classify_num;
        let timestamp_segment_time = aligner_config.timestamp_segment_time;

        let burn_config =
            Qwen3ASRConfig::from_aligner_configs(audio_config, text_config.clone(), classify_num);
        let mut model = burn_config.init(&device);

        let weights_path = format!("{}/model.safetensors", model_dir);
        {
            #[cfg(feature = "metal")]
            use burn_store::ChainAdapter;
            use burn_store::{ModuleSnapshot, PyTorchToBurnAdapter, SafetensorsStore};
            #[cfg(feature = "metal")]
            let adapter = ChainAdapter::new(PyTorchToBurnAdapter, Bf16ToF32Adapter);
            #[cfg(not(feature = "metal"))]
            let adapter = PyTorchToBurnAdapter;
            let mut store = SafetensorsStore::from_file(&weights_path)
                .with_from_adapter(adapter)
                .allow_partial(true);
            let result = model.load_from(&mut store)?;
            log::info!(
                "Weight loading: {} applied, {} errors",
                result.applied.len(),
                result.errors.len()
            );
            if !result.errors.is_empty() {
                log::warn!("First 10 weight load errors:");
                for err in result.errors.iter().take(10) {
                    log::warn!("  {:?}", err);
                }
            }
        }

        let tokenizer = Qwen2Tokenizer::from_dir(model_dir)?;

        let preprocessor_config = PreprocessorConfig::from_dir(model_dir).unwrap_or_else(|_| {
            log::warn!("No preprocessor_config.json found, using defaults");
            PreprocessorConfig {
                feature_size: 128,
                n_fft: 400,
                hop_length: 160,
                n_samples: 480000,
                nb_max_frames: 3000,
                chunk_length: 30.0,
                padding_value: 0.0,
                return_attention_mask: false,
            }
        });
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
            timestamp_segment_time,
        })
    }

    pub fn align(
        &self,
        audio_path: &str,
        text: &str,
        language: &str,
    ) -> anyhow::Result<Vec<TimestampItem>> {
        // 1. Load and resample audio. Pad to a 30s multiple (no truncation)
        //    so tail words still get valid mel frames.
        let samples = audio::load_wav_samples(audio_path)?;
        let padded = audio::pad_to_30s_multiple(&samples);
        let mel_spec = self.mel_extractor.compute(&padded);
        let n_mels = mel_spec.len();
        let n_frames = mel_spec[0].len();
        let flat: Vec<f32> = mel_spec.into_iter().flatten().collect();
        log::info!(
            "Mel: {n_mels} bins x {n_frames} frames (padded {:.1}s → {:.1}s)",
            samples.len() as f32 / 16_000.0,
            padded.len() as f32 / 16_000.0
        );
        let mel_tensor =
            Tensor::<1>::from_floats(flat.as_slice(), &self.device).reshape([1, n_mels, n_frames]);

        // 2. Run audio encoder
        let audio_features = self.model.thinker.audio_tower.forward(mel_tensor);
        let t_audio = audio_features.dims()[1];
        log::info!("Audio encoder output: {} tokens", t_audio);

        // 3. Format text with timestamp markers
        let (word_list, formatted_text) = text_processor::encode_timestamp(text, language);

        // 4. Build prompt: audio placeholder + formatted text
        let audio_placeholder = format!(
            "<|audio_start|>{}<|audio_end|>",
            "<|audio_pad|>".repeat(t_audio)
        );
        let full_text = format!("{}{}", audio_placeholder, formatted_text);

        // 5. Tokenize
        let input_ids = self.tokenizer.encode(&full_text);
        let seq_len = input_ids.len();

        // 6. Embed input tokens
        let input_ids_i32: Vec<i32> = input_ids.iter().map(|&id| id as i32).collect();
        let input_ids_tensor = Tensor::<1, Int>::from_ints(input_ids_i32.as_slice(), &self.device)
            .unsqueeze_dim::<2>(0);
        let inputs_embeds = self
            .model
            .thinker
            .model
            .embed_tokens
            .forward(input_ids_tensor);

        // 7. Replace audio_pad positions with encoder features
        let audio_pad_id = self.tokenizer.audio_pad;
        let pad_positions: Vec<usize> = input_ids
            .iter()
            .enumerate()
            .filter(|(_, &id)| id == audio_pad_id)
            .map(|(i, _)| i)
            .collect();

        if pad_positions.len() != t_audio {
            anyhow::bail!(
                "Audio pad count mismatch: {} tokens in prompt vs {} encoder output frames",
                pad_positions.len(),
                t_audio
            );
        }

        let start = pad_positions[0];
        let end = pad_positions[pad_positions.len() - 1] + 1;

        let hidden = inputs_embeds.dims()[2];
        let prefix = inputs_embeds.clone().slice([0..1, 0..start, 0..hidden]);
        let suffix = inputs_embeds.clone().slice([0..1, end..seq_len, 0..hidden]);
        let replaced_embeds = Tensor::cat(vec![prefix, audio_features, suffix], 1);

        // 8. Single forward pass with causal mask
        let total_positions: Vec<usize> = (0..replaced_embeds.dims()[1]).collect();
        let (cos, sin) = self
            .mrope
            .compute_cos_sin_from_positions(&total_positions, &self.device);
        let causal_mask = model::create_causal_mask(replaced_embeds.dims()[1], 0, &self.device);

        let hidden_states = self.model.thinker.model.forward_embeds(
            replaced_embeds,
            &cos,
            &sin,
            Some(causal_mask),
            None, // no KV-cache: single-pass forward
        );
        let logits = self.model.thinker.lm_head.forward(hidden_states);
        let logit_seq_len = logits.dims()[1];

        // 9. Extract logits at timestamp positions
        let timestamp_id = *self.tokenizer.timestamp_id.as_ref().ok_or_else(|| {
            anyhow::anyhow!("tokenizer is missing <timestamp> (required for alignment)")
        })?;
        let timestamp_positions: Vec<usize> = input_ids
            .iter()
            .enumerate()
            .filter(|(_, &id)| id == timestamp_id)
            .map(|(i, _)| i)
            .collect();

        if timestamp_positions.is_empty() {
            anyhow::bail!("No timestamp tokens found in input");
        }

        // 10. Get class predictions for each timestamp position
        let classify_num = logits.dims()[2];
        let timestamp_classes: Vec<f64> = timestamp_positions
            .iter()
            .filter(|&&pos| pos < logit_seq_len)
            .map(|&pos| {
                let logit_slice = logits
                    .clone()
                    .slice([0..1, pos..pos + 1, 0..classify_num])
                    .reshape([classify_num]);
                let token_data = logit_slice.argmax(0).reshape([1]).into_data();
                let class_id = i32::from_le_bytes([
                    token_data.bytes[0],
                    token_data.bytes[1],
                    token_data.bytes[2],
                    token_data.bytes[3],
                ]) as f64;
                class_id * self.timestamp_segment_time as f64
            })
            .collect();

        log::info!(
            "Predicted {} timestamps from {} positions",
            timestamp_classes.len(),
            timestamp_positions.len()
        );

        // 11. Monotonicity fix
        let fixed_timestamps = text_processor::fix_timestamp(&timestamp_classes);

        // 12. Pair with words
        let items = text_processor::parse_timestamp(&word_list, &fixed_timestamps);

        Ok(items)
    }
}
