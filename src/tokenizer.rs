use std::path::Path;
use tokenizers::models::bpe::BPE;
use tokenizers::pre_tokenizers::byte_level::ByteLevel;
use tokenizers::decoders::byte_level::ByteLevel as ByteLevelDecoder;
use tokenizers::{AddedToken, Tokenizer};

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
        let vocab_path = Path::new(model_dir).join("vocab.json");
        let merges_path = Path::new(model_dir).join("merges.txt");

        let bpe = BPE::from_file(
            vocab_path.to_str().unwrap(),
            merges_path.to_str().unwrap(),
        )
        .build()
        .map_err(|e| anyhow::anyhow!("Failed to load BPE model: {}", e))?;

        let mut tokenizer = Tokenizer::new(bpe);
        tokenizer.with_pre_tokenizer(Some(ByteLevel::default()));
        tokenizer.with_decoder(Some(ByteLevelDecoder::default()));

        let special_tokens: Vec<(&str, bool, bool)> = vec![
            ("<|endoftext|>", true, false),
            ("<|im_start|>", true, false),
            ("<|im_end|>", true, false),
            ("<|object_ref_start|>", true, false),
            ("<|object_ref_end|>", true, false),
            ("<|box_start|>", true, false),
            ("<|box_end|>", true, false),
            ("<|quad_start|>", true, false),
            ("<|quad_end|>", true, false),
            ("<|vision_start|>", true, false),
            ("<|vision_end|>", true, false),
            ("<|vision_pad|>", true, false),
            ("<|image_pad|>", true, false),
            ("<|video_pad|>", true, false),
            ("<tool_call>", false, false),
            ("</tool_call>", false, false),
            ("<|fim_prefix|>", false, false),
            ("<|fim_middle|>", false, false),
            ("<|fim_suffix|>", false, false),
            ("<|fim_pad|>", false, false),
            ("<|repo_name|>", false, false),
            ("<|file_sep|>", false, false),
            ("<tool_response>", false, false),
            ("</tool_response>", false, false),
            ("<think>", false, false),
            ("</think>", false, false),
            ("<|audio_start|>", true, false),
            ("<|audio_end|>", true, false),
            ("<tts_pad>", true, false),
            ("<tts_text_bos>", true, false),
            ("<tts_text_eod>", true, false),
            ("<tts_text_bos_single>", true, false),
            ("<non_speech>", false, false),
            ("<|audio_pad|>", true, false),
            ("<blank1>", true, false),
            ("<blank2>", true, false),
            ("<blank3>", true, false),
            ("<blank4>", true, false),
            ("<blank5>", true, false),
            ("<blank6>", true, false),
            ("<blank7>", true, false),
            ("<blank8>", true, false),
            ("<blank9>", true, false),
            ("<blank10>", true, false),
            ("<blank11>", true, false),
            ("<blank12>", true, false),
            ("<blank13>", true, false),
            ("<blank14>", true, false),
            ("<blank15>", true, false),
            ("<blank16>", true, false),
            ("<blank17>", true, false),
            ("<blank18>", true, false),
            ("<blank19>", true, false),
            ("<blank20>", true, false),
            ("<blank21>", true, false),
            ("<blank22>", true, false),
            ("<blank23>", true, false),
            ("<blank24>", true, false),
            ("<blank25>", true, false),
            ("<blank26>", true, false),
            ("<blank27>", true, false),
            ("<asr_text>", false, false),
        ];

        let added_tokens: Vec<AddedToken> = special_tokens
            .into_iter()
            .map(|(content, special, single_word)| {
                AddedToken::from(content, special)
                    .single_word(single_word)
                    .rstrip(false)
                    .lstrip(false)
                    .normalized(false)
            })
            .collect();

        let count = tokenizer.add_special_tokens(&added_tokens);
        log::info!("Added {} special tokens", count);

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
        let special_tokens: &[&str] = &[
            "<|endoftext|>", "<|im_start|>", "<|im_end|>",
            "<|object_ref_start|>", "<|object_ref_end|>",
            "<|box_start|>", "<|box_end|>",
            "<|quad_start|>", "<|quad_end|>",
            "<|vision_start|>", "<|vision_end|>", "<|vision_pad|>",
            "<|image_pad|>", "<|video_pad|>",
            "<tool_call>", "</tool_call>",
            "<|fim_prefix|>", "<|fim_middle|>", "<|fim_suffix|>", "<|fim_pad|>",
            "<|repo_name|>", "<|file_sep|>",
            "<tool_response>", "</tool_response>",
            "<think>", "</think>",
            "<|audio_start|>", "<|audio_end|>",
            "<tts_pad>", "<tts_text_bos>", "<tts_text_eod>", "<tts_text_bos_single>",
            "<non_speech>", "<|audio_pad|>",
            "<blank1>", "<blank2>", "<blank3>", "<blank4>", "<blank5>",
            "<blank6>", "<blank7>", "<blank8>", "<blank9>", "<blank10>",
            "<blank11>", "<blank12>", "<blank13>", "<blank14>", "<blank15>",
            "<blank16>", "<blank17>", "<blank18>", "<blank19>", "<blank20>",
            "<blank21>", "<blank22>", "<blank23>", "<blank24>", "<blank25>",
            "<blank26>", "<blank27>", "<asr_text>",
        ];

        let mut positions: Vec<(usize, usize, u32)> = Vec::new();
        for token_str in special_tokens {
            let token_id = match self.tokenizer.token_to_id(token_str) {
                Some(id) => id,
                None => continue,
            };
            let mut start = 0;
            while let Some(pos) = text[start..].find(token_str) {
                let abs_pos = start + pos;
                positions.push((abs_pos, abs_pos + token_str.len(), token_id));
                start = abs_pos + token_str.len();
            }
        }
        positions.sort_by_key(|(s, _, _)| *s);

        let mut filtered: Vec<(usize, usize, u32)> = Vec::new();
        for (s, e, id) in positions {
            let overlaps = filtered.iter().any(|(fs, fe, _)| !(e <= *fs || s >= *fe));
            if !overlaps {
                filtered.push((s, e, id));
            }
        }

        let mut space_map: std::collections::HashMap<u32, u32> = std::collections::HashMap::new();
        let mut result: Vec<u32> = Vec::new();
        let mut cursor = 0;
        for (start, end, token_id) in &filtered {
            if cursor < *start {
                let segment = &text[cursor..*start];
                if let Ok(encoding) = self.tokenizer.encode(segment, false) {
                    let mut ids: Vec<u32> = encoding.get_ids().to_vec();
                    if !ids.is_empty() {
                        let first_id = ids[0];
                        if let Some(&fixed_id) = space_map.get(&first_id) {
                            ids[0] = fixed_id;
                        } else {
                            let decoded = self.tokenizer.decode(&[first_id], true).unwrap_or_default();
                            if decoded.starts_with(' ') {
                                let unspaced = decoded.trim_start_matches(' ');
                                if !unspaced.is_empty() {
                                    if let Ok(enc) = self.tokenizer.encode(unspaced, false) {
                                        let new_ids: Vec<u32> = enc.get_ids().to_vec();
                                        if !new_ids.is_empty() {
                                            space_map.insert(first_id, new_ids[0]);
                                            ids[0] = new_ids[0];
                                        }
                                    }
                                }
                            }
                        }
                    }
                    result.extend_from_slice(&ids);
                }
            }
            result.push(*token_id);
            cursor = *end;
        }
        if cursor < text.len() {
            let segment = &text[cursor..];
            if let Ok(encoding) = self.tokenizer.encode(segment, false) {
                let mut ids: Vec<u32> = encoding.get_ids().to_vec();
                if !ids.is_empty() {
                    let first_id = ids[0];
                    if let Some(&fixed_id) = space_map.get(&first_id) {
                        ids[0] = fixed_id;
                    }
                }
                result.extend_from_slice(&ids);
            }
        }

        if result.is_empty() {
            match self.tokenizer.encode(text, false) {
                Ok(encoding) => encoding.get_ids().iter().map(|&id| id).collect(),
                Err(e) => {
                    log::warn!("Tokenization error: {}", e);
                    vec![]
                }
            }
        } else {
            result
        }
    }

    pub fn decode(&self, ids: &[u32]) -> String {
        let ids_vec: Vec<u32> = ids.to_vec();
        match self.tokenizer.decode(&ids_vec, true) {
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
