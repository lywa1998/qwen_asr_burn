//! Translation pipeline using the Hy-MT (Hunyuan Machine Translation) model.

use burn::tensor::{Device, Int, Tensor};
#[cfg(feature = "metal")]
use burn_store::ChainAdapter;
use burn_store::{ModuleSnapshot, PyTorchToBurnAdapter, SafetensorsStore};
use tokenizers::Tokenizer;

use crate::models::hy_mt::config::HYV3Config;
use crate::models::hy_mt::{HYV3ForCausalLM, HYV3RotaryEmbedding};
#[cfg(feature = "metal")]
use crate::transcribe_pipeline::Bf16ToF32Adapter;

#[derive(Debug, Clone)]
pub struct GenerationConfig {
    pub max_new_tokens: usize,
    pub temperature: f64,
    pub top_k: usize,
    pub top_p: f64,
    pub do_sample: bool,
    pub eos_token_id: usize,
}

impl Default for GenerationConfig {
    fn default() -> Self {
        // Hy-MT 1.8B/7B recommended sampling params from the model README.
        Self {
            max_new_tokens: 4096,
            temperature: 0.7,
            top_k: 20,
            top_p: 0.6,
            do_sample: true,
            eos_token_id: 120_020,
        }
    }
}

impl GenerationConfig {
    pub fn from_model_config(config: &HYV3Config) -> Self {
        Self {
            eos_token_id: config.eos_token_id,
            ..Default::default()
        }
    }
}

pub struct TranslatePipeline {
    model: HYV3ForCausalLM,
    rope: HYV3RotaryEmbedding,
    tokenizer: Tokenizer,
    gen_config: GenerationConfig,
    device: Device,
}

impl TranslatePipeline {
    pub fn new(model_dir: &str, device: Device) -> anyhow::Result<Self> {
        let config_path = format!("{model_dir}/config.json");
        let model_config = HYV3Config::from_file(&config_path)?;
        log::info!(
            "Loaded Hy-MT config: hidden_size={}",
            model_config.hidden_size
        );

        let tokenizer_path = format!("{model_dir}/tokenizer.json");
        let mut tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {e}"))?;
        let _ = tokenizer.with_padding(Some(tokenizers::PaddingParams {
            strategy: tokenizers::PaddingStrategy::Fixed(0),
            pad_id: model_config.pad_token_id as u32,
            ..Default::default()
        }));

        let mut model = HYV3ForCausalLM::new(&model_config, &device);

        let safetensors_path = format!("{model_dir}/model.safetensors");
        log::info!("Loading Hy-MT weights from {safetensors_path}");

        #[cfg(feature = "metal")]
        let adapter = ChainAdapter::new(PyTorchToBurnAdapter, Bf16ToF32Adapter);
        #[cfg(not(feature = "metal"))]
        let adapter = PyTorchToBurnAdapter;
        let mut store = SafetensorsStore::from_file(&safetensors_path)
            .with_from_adapter(adapter)
            .allow_partial(true);
        let result = model
            .load_from(&mut store)
            .map_err(|e| anyhow::anyhow!("Failed to load Hy-MT weights: {e}"))?;
        log::info!(
            "Hy-MT weights: {} applied, {} errors",
            result.applied.len(),
            result.errors.len()
        );

        // Models with `tie_word_embeddings: true` omit `lm_head.weight` from
        // safetensors; copy from embed_tokens. This is the standard HuggingFace
        // pattern documented in PreTrainedModel._tied_weights_keys.
        if model_config.tie_word_embeddings {
            model.tie_lm_head_to_embeddings();
            log::info!("Hy-MT: tied lm_head.weight ← embed_tokens.weight");
        }

        let rope = HYV3RotaryEmbedding::new(&model_config);
        let gen_config = GenerationConfig::from_model_config(&model_config);

        Ok(Self {
            model,
            rope,
            tokenizer,
            gen_config,
            device,
        })
    }

    /// Translate `text` into `target_lang` (e.g. "English", "中文").
    pub fn translate(&self, text: &str, target_lang: &str) -> anyhow::Result<String> {
        let prompt = build_prompt(target_lang, text);
        self.generate(&prompt)
    }

    /// Run autoregressive generation on the formatted prompt.
    /// Returns only the newly generated text (prompt stripped).
    fn generate(&self, prompt: &str) -> anyhow::Result<String> {
        let encoding = self
            .tokenizer
            .encode(prompt, false)
            .map_err(|e| anyhow::anyhow!("Tokenization failed: {e}"))?;
        let mut token_ids: Vec<i32> = encoding.get_ids().iter().map(|&id| id as i32).collect();
        let prompt_len = token_ids.len();

        for _step in 0..self.gen_config.max_new_tokens {
            let len = token_ids.len();
            let input =
                Tensor::<1, Int>::from_ints(token_ids.as_slice(), &self.device).reshape([1, len]);

            let logits = self.model.forward(input, &self.rope);
            let vocab_size = logits.dims()[2];
            let last_logits = logits
                .clone()
                .slice([0..1, (len - 1)..len, 0..vocab_size])
                .reshape([vocab_size]);

            let last_logits_f32: Vec<f32> = last_logits
                .into_data()
                .to_vec()
                .map_err(|e| anyhow::anyhow!("Failed to get logits data: {}", e))?;

            let next_token = if self.gen_config.do_sample && self.gen_config.temperature > 0.0 {
                sample_token(&last_logits_f32, &self.gen_config)
            } else {
                argmax_token(&last_logits_f32)
            } as i32;

            if next_token == self.gen_config.eos_token_id as i32 {
                break;
            }

            token_ids.push(next_token);
        }

        let generated_ids: Vec<u32> = token_ids[prompt_len..]
            .iter()
            .map(|&id| id as u32)
            .collect();
        let output = self
            .tokenizer
            .decode(&generated_ids, true)
            .map_err(|e| anyhow::anyhow!("Decoding failed: {e}"))?;

        Ok(output)
    }
}

// ── Prompt formatting ──────────────────────────────────────────────────────

/// Hy-MT chat-template formatter. Per the model README, Hy-MT has no default
/// system prompt — only a user turn is used. The recommended translation
/// instruction template is hard-coded here.
///
/// chat_template.jinja (no system branch):
///   <｜hy_begin▁of▁sentence｜><｜hy_User｜>{content}<｜hy_Assistant｜>
fn build_prompt(target_lang: &str, source_text: &str) -> String {
    let user_content = format!(
        "Translate the following text into {target_lang}. Note that you should only output the translated result without any additional explanation:\n\n{source_text}"
    );
    let mut p = String::with_capacity(user_content.len() + 64);
    p.push_str("<｜hy_begin▁of▁sentence｜>");
    p.push_str("<｜hy_User｜>");
    p.push_str(&user_content);
    p.push_str("<｜hy_Assistant｜>");
    p
}

// ── Token selection helpers ────────────────────────────────────────────────

fn argmax_token(logits: &[f32]) -> usize {
    logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap_or(0)
}

fn sample_token(logits: &[f32], config: &GenerationConfig) -> usize {
    let temperature = config.temperature as f32;
    let scaled: Vec<f32> = logits.iter().map(|x| x / temperature).collect();

    let mut indexed: Vec<(usize, f32)> = scaled.iter().copied().enumerate().collect();
    indexed
        .sort_unstable_by(|(_, a), (_, b)| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));

    if config.top_k > 0 && config.top_k < indexed.len() {
        let threshold = indexed[config.top_k - 1].1;
        indexed.retain(|(_, v)| *v >= threshold);
    }

    if config.top_p < 1.0 {
        indexed.sort_unstable_by(|(_, a), (_, b)| {
            b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal)
        });
        let max_val = indexed[0].1;
        let exps: Vec<f32> = indexed.iter().map(|(_, v)| (v - max_val).exp()).collect();
        let sum_exps: f32 = exps.iter().sum();
        let probs: Vec<f32> = exps.iter().map(|v| v / sum_exps).collect();

        let mut cumsum = 0.0f32;
        let mut cutoff = indexed.len();
        for (i, p) in probs.iter().enumerate() {
            cumsum += p;
            if cumsum > config.top_p as f32 {
                cutoff = i + 1;
                break;
            }
        }
        indexed.truncate(cutoff);
    }

    let max_val = indexed[0].1;
    let exps: Vec<f32> = indexed.iter().map(|(_, v)| (v - max_val).exp()).collect();
    let sum_exps: f32 = exps.iter().sum();
    let probs: Vec<f32> = exps.iter().map(|v| v / sum_exps).collect();

    let r: f32 = rand::random();
    let mut cumsum = 0.0f32;
    for (i, p) in probs.iter().enumerate() {
        cumsum += p;
        if r < cumsum {
            return indexed[i].0;
        }
    }

    indexed.last().map(|(i, _)| *i).unwrap_or(0)
}
