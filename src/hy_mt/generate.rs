use burn::tensor::backend::Backend;
use burn::tensor::{Int, Tensor};
use tokenizers::Tokenizer;

use crate::hy_mt::config::ModelConfig;
use crate::hy_mt::model::{HunYuanModel, RotaryEmbedding};

#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct GenerationConfig {
    pub max_new_tokens: usize,
    pub temperature: f64,
    pub top_k: usize,
    pub top_p: f64,
    pub repetition_penalty: f64,
    pub do_sample: bool,
    pub eos_token_id: usize,
    pub pad_token_id: usize,
}

impl Default for GenerationConfig {
    fn default() -> Self {
        Self {
            max_new_tokens: 512,
            temperature: 0.7,
            top_k: 20,
            top_p: 0.8,
            repetition_penalty: 1.05,
            do_sample: true,
            eos_token_id: 120_020,
            pad_token_id: 120_002,
        }
    }
}

impl GenerationConfig {
    pub fn from_model_config(config: &ModelConfig) -> Self {
        Self {
            eos_token_id: config.eos_token_id,
            pad_token_id: config.pad_token_id,
            ..Default::default()
        }
    }
}

/// Generate tokens autoregressively. Returns only the newly generated text.
pub fn generate<B: Backend>(
    model: &HunYuanModel<B>,
    rope: &RotaryEmbedding,
    tokenizer: &Tokenizer,
    prompt: &str,
    gen_config: &GenerationConfig,
    device: &B::Device,
) -> anyhow::Result<String> {
    let encoding = tokenizer
        .encode(prompt, false)
        .map_err(|e| anyhow::anyhow!("Tokenization failed: {e}"))?;
    let mut token_ids: Vec<i32> = encoding.get_ids().iter().map(|&id| id as i32).collect();
    let prompt_len = token_ids.len();

    for _step in 0..gen_config.max_new_tokens {
        let len = token_ids.len();
        let input =
            Tensor::<B, 1, Int>::from_ints(token_ids.as_slice(), device).reshape([1, len]);

        let logits = model.forward(input, rope);
        let vocab_size = logits.dims()[2];
        let last_logits = logits
            .clone()
            .slice([0..1, (len - 1)..len, 0..vocab_size])
            .reshape([vocab_size]);

        let last_logits_f32: Vec<f32> = last_logits
            .into_data()
            .to_vec()
            .map_err(|e| anyhow::anyhow!("Failed to get logits data: {}", e))?;

        let next_token = if gen_config.do_sample && gen_config.temperature > 0.0 {
            sample_token(&last_logits_f32, gen_config)
        } else {
            argmax_token(&last_logits_f32)
        } as i32;

        if next_token == gen_config.eos_token_id as i32 {
            break;
        }

        token_ids.push(next_token);
    }

    let generated_ids: Vec<u32> = token_ids[prompt_len..]
        .iter()
        .map(|&id| id as u32)
        .collect();
    let output = tokenizer
        .decode(&generated_ids, true)
        .map_err(|e| anyhow::anyhow!("Decoding failed: {e}"))?;

    Ok(output)
}

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
    indexed.sort_unstable_by(|(_, a), (_, b)| {
        b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal)
    });

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
