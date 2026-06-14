// Helper: convert Qwen3 tokenizer files → tokenizer.json
use tokenizers::models::bpe::BPE;
use tokenizers::pre_tokenizers::byte_level::ByteLevel;
use tokenizers::decoders::byte_level::ByteLevel as ByteLevelDecoder;
use tokenizers::{AddedToken, Tokenizer};

fn main() {
    let model_dir = "Qwen3-ASR-0.6B";
    let bpe = BPE::from_file(
        &format!("{}/vocab.json", model_dir),
        &format!("{}/merges.txt", model_dir),
    )
    .build()
    .map_err(|e| format!("BPE build failed: {}", e))
    .unwrap();

    let mut tokenizer = Tokenizer::new(bpe);
    tokenizer.with_pre_tokenizer(Some(ByteLevel::default()));
    tokenizer.with_decoder(Some(ByteLevelDecoder::default()));

    // Add all 62 special tokens from tokenizer_config.json in correct order (151643-151704)
    let special_tokens: Vec<AddedToken> = vec![
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
    ]
    .into_iter()
    .map(|s| {
        let is_special = !matches!(s,
            "<tool_call>" | "</tool_call>" | "<|fim_prefix|>" | "<|fim_middle|>" |
            "<|fim_suffix|>" | "<|fim_pad|>" | "<|repo_name|>" | "<|file_sep|>" |
            "<tool_response>" | "</tool_response>" | "<think>" | "</think>" |
            "<non_speech>" | "<asr_text>"
        );
        AddedToken::from(s, is_special)
    })
    .collect();

    tokenizer.add_special_tokens(&special_tokens);

    tokenizer
        .save(&format!("{}/tokenizer.json", model_dir), false)
        .map_err(|e| format!("Save failed: {}", e))
        .unwrap();

    println!("Created tokenizer.json in {} with {} added tokens", model_dir, special_tokens.len());
}
