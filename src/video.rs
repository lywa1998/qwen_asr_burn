use avio::{AudioDecoder, SampleFormat};

/// Extract audio from a video file, returning 16kHz mono f32 PCM samples.
pub fn extract_audio(input_path: &str) -> anyhow::Result<Vec<f32>> {
    let mut decoder = AudioDecoder::open(input_path)
        .output_format(SampleFormat::F32)
        .output_sample_rate(16_000)
        .build()
        .map_err(|e| anyhow::anyhow!("Failed to open audio stream in '{input_path}': {e}"))?;

    let channels = decoder.channels() as usize;
    let src_rate = decoder.sample_rate();
    log::info!(
        "Extracting audio: src_rate={src_rate} Hz, channels={channels}, codec={}",
        decoder.stream_info().codec_name()
    );

    let mut samples = Vec::new();

    for result in &mut decoder {
        let frame = result.map_err(|e| anyhow::anyhow!("Audio decode error: {e}"))?;
        let pcm = frame.to_f32_interleaved();

        if channels == 1 {
            samples.extend_from_slice(&pcm);
        } else {
            for chunk in pcm.chunks(channels) {
                let mono = chunk.iter().sum::<f32>() / channels as f32;
                samples.push(mono);
            }
        }
    }

    let duration = samples.len() as f64 / 16_000.0;
    log::info!(
        "Extracted {} samples ({:.2}s) at 16kHz mono",
        samples.len(),
        duration
    );
    Ok(samples)
}

/// Save mono f32 samples as a 16kHz WAV file.
pub fn save_audio_wav(samples: &[f32], output_path: &str) -> anyhow::Result<()> {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: 16_000,
        bits_per_sample: 32,
        sample_format: hound::SampleFormat::Float,
    };
    let mut writer = hound::WavWriter::create(output_path, spec)?;
    for &s in samples {
        writer.write_sample(s)?;
    }
    writer.finalize()?;
    log::info!("Saved WAV: {output_path} ({:.2}s)", samples.len() as f64 / 16_000.0);
    Ok(())
}
