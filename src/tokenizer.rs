use std::path::Path;
use tokenizers::Tokenizer;

pub struct Qwen2Tokenizer {
    pub tokenizer: Tokenizer,
    pub audio_start: u32,
    pub audio_end: u32,
    pub audio_pad: u32,
    pub im_start: u32,
    pub im_end: u32,
    pub eos_token_id: u32,
}

impl Qwen2Tokenizer {
    pub fn from_dir(model_dir: &str) -> anyhow::Result<Self> {
        let tokenizer_path = Path::new(model_dir).join("tokenizer.json");
        let tokenizer = Tokenizer::from_file(tokenizer_path.to_str().unwrap())
            .map_err(|e| anyhow::anyhow!("Failed to load tokenizer.json: {}", e))?;

        let audio_pad_id = required_token_id(&tokenizer, "<|audio_pad|>")?;
        let audio_start_id = required_token_id(&tokenizer, "<|audio_start|>")?;
        let audio_end_id = required_token_id(&tokenizer, "<|audio_end|>")?;
        let im_start_id = required_token_id(&tokenizer, "<|im_start|>")?;
        let im_end_id = required_token_id(&tokenizer, "<|im_end|>")?;
        let eos_id = required_token_id(&tokenizer, "<|endoftext|>")?;

        log::info!("Audio tokens: start={audio_start_id}, end={audio_end_id}, pad={audio_pad_id}");
        log::info!("Chat tokens: im_start={im_start_id}, im_end={im_end_id}, eos={eos_id}");

        Ok(Self {
            tokenizer,
            audio_start: audio_start_id,
            audio_end: audio_end_id,
            audio_pad: audio_pad_id,
            im_start: im_start_id,
            im_end: im_end_id,
            eos_token_id: eos_id,
        })
    }

    pub fn encode(&self, text: &str) -> Vec<u32> {
        match self.tokenizer.encode(text, false) {
            Ok(encoding) => encoding.get_ids().to_vec(),
            Err(e) => {
                log::warn!("Tokenization error: {}", e);
                vec![]
            }
        }
    }

    pub fn decode(&self, ids: &[u32]) -> String {
        match self.tokenizer.decode(ids, true) {
            Ok(s) => s,
            Err(e) => {
                log::warn!("Decode error: {}", e);
                String::new()
            }
        }
    }
}

fn required_token_id(tokenizer: &Tokenizer, token: &str) -> anyhow::Result<u32> {
    tokenizer
        .token_to_id(token)
        .ok_or_else(|| anyhow::anyhow!("missing required special token: {}", token))
}
