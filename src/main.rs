#![cfg_attr(feature = "metal", recursion_limit = "256")]

mod align_pipeline;
mod audio;
mod config;
mod model;
mod pipeline;
mod srt;
mod text_processor;
mod tokenizer;
mod vad;
mod video;

use align_pipeline::AlignPipeline;
use clap::Parser;
use pipeline::AsrPipeline;

#[cfg(feature = "cuda")]
use burn::{backend::{cuda::CudaDevice, Cuda}, tensor::bf16};

#[cfg(feature = "metal")]
use burn::backend::{wgpu::WgpuDevice, Wgpu};

#[cfg(feature = "cuda")]
type Backend = Cuda<bf16, i32>;

#[cfg(feature = "metal")]
type Backend = Wgpu<f32, i32>;

#[derive(Parser)]
#[command(
    name = "qwen-asr",
    version,
    about = "Qwen3-ASR with Burn (BF16)"
)]
struct Cli {
    #[arg(short, long, default_value = "Qwen3-ASR-0.6B")]
    model_dir: String,

    #[command(subcommand)]
    command: Command,
}

#[derive(clap::Subcommand)]
enum Command {
    /// Transcribe speech to text from a WAV file
    Transcribe {
        /// Input WAV file (16kHz mono recommended)
        input: String,
        /// Output text file (default: <input_stem>_transcript.txt)
        #[arg(short, long)]
        output: Option<String>,
        /// Force language (e.g. "Chinese", "English"). Skips language detection.
        #[arg(short, long)]
        language: Option<String>,
        /// Context string for the system prompt (e.g. "You are a transcription expert.")
        #[arg(short, long, default_value = "")]
        context: String,
        /// Save SRT subtitle file alongside transcript
        #[arg(long)]
        save_srt: bool,
    },
    /// Force-align text to audio, producing word-level timestamps
    Align {
        /// Input WAV file
        #[arg(short, long)]
        input: String,
        /// Text to align with the audio
        #[arg(short, long)]
        text: String,
        /// Language for word splitting (Chinese, English, etc.)
        #[arg(short, long, default_value = "English")]
        language: String,
        /// Output format: "text" or "json"
        #[arg(short = 'F', long, default_value = "text")]
        format: String,
    },
    /// Extract 16kHz mono audio from a video file
    Extract {
        /// Input video file (mp4, mkv, avi, etc.)
        input: String,
        /// Output WAV file (default: <input_stem>.wav)
        #[arg(short, long)]
        output: Option<String>,
    },
}

fn main() -> anyhow::Result<()> {
    env_logger::init();
    let cli = Cli::parse();

    #[cfg(feature = "cuda")]
    let device = CudaDevice::default();
    #[cfg(feature = "metal")]
    let device = WgpuDevice::default();

    match cli.command {
        Command::Transcribe { input, output, language, context, save_srt } => {
            log::info!("Initializing Qwen3-ASR...");
            let pipeline = AsrPipeline::<Backend>::new(&cli.model_dir, device)?;
            log::info!("Transcribing: {} (language={:?})", input, language);
            let (texts, segments) = pipeline.transcribe(&input, language.as_deref(), &context)?;
            let combined = texts.join("\n");
            println!("{combined}");

            let stem = std::path::Path::new(&input)
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("output");
            let out_path = output.unwrap_or_else(|| format!("{stem}_transcript.txt"));
            std::fs::write(&out_path, &combined)?;
            log::info!("Wrote transcript to {out_path}");

            if !segments.is_empty() {
                let seg_path = format!("{stem}_segments.json");
                std::fs::write(&seg_path, serde_json::to_string_pretty(&segments)?)?;
                log::info!("Wrote {} segments to {seg_path}", segments.len());
            }

            if save_srt && !segments.is_empty() {
                let srt_path = format!("{stem}.srt");
                srt::write_srt(&segments, &texts, &srt_path)?;
            }
        }
        Command::Align {
            input,
            text,
            language,
            format,
        } => {
            log::info!("Initializing Qwen3-ForcedAligner...");
            let pipeline = AlignPipeline::<Backend>::new(&cli.model_dir, device)?;
            log::info!("Aligning: {} with language={}", input, language);
            let results = pipeline.align(&input, &text, &language)?;
            match format.as_str() {
                "json" => println!("{}", serde_json::to_string_pretty(&results)?),
                _ => {
                    for item in &results {
                        println!(
                            "{:.3}s-{:.3}s  {}",
                            item.start_time, item.end_time, item.text
                        );
                    }
                }
            }
        }
        Command::Extract { input, output } => {
            let output_path = output.unwrap_or_else(|| {
                let stem = std::path::Path::new(&input)
                    .file_stem()
                    .and_then(|s| s.to_str())
                    .unwrap_or("audio");
                format!("{stem}.wav")
            });
            log::info!("Extracting audio from: {input}");
            let samples = video::extract_audio(&input)?;
            video::save_audio_wav(&samples, &output_path)?;
            let duration = samples.len() as f64 / 16_000.0;
            println!("Extracted {:.2}s audio → {output_path}", duration);
        }
    }
    Ok(())
}
