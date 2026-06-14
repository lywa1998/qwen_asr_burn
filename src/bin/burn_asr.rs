//! Burn implementation binary for Qwen3-ASR.
//! Run with: cargo run --bin burn-asr -- transcribe <file.wav>

use burn::backend::Cuda;
use clap::Parser;

#[path = "../audio.rs"]
mod audio;
#[path = "../config.rs"]
mod config;
#[path = "../model.rs"]
mod model;
#[path = "../pipeline.rs"]
mod pipeline;
#[path = "../tokenizer.rs"]
mod tokenizer;

use pipeline::AsrPipeline;

#[derive(Parser)]
#[command(
    name = "burn-asr",
    about = "Qwen3-ASR Burn implementation (development)"
)]
struct Cli {
    #[arg(short, long, default_value = "Qwen3-ASR-0.6B")]
    model_dir: String,

    #[command(subcommand)]
    command: Command,
}

#[derive(clap::Subcommand)]
enum Command {
    Transcribe {
        input: String,
        #[arg(short, long)]
        output: Option<String>,
    },
}

type Backend = Cuda;

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    log::info!("Burn ASR: CUDA backend...");
    let device = burn::backend::cuda::CudaDevice::default();
    let pipeline = AsrPipeline::<Backend>::new(&cli.model_dir, device)?;

    match cli.command {
        Command::Transcribe { input, output } => {
            let text = pipeline.transcribe(&input)?;
            println!("{text}");
            if let Some(p) = output {
                std::fs::write(&p, &text)?;
            }
        }
    }
    Ok(())
}
