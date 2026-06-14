mod audio;
mod config;
mod model;
mod pipeline;
mod tokenizer;

use burn::backend::Cuda;
use burn::tensor::bf16;
use clap::Parser;
use pipeline::AsrPipeline;

#[derive(Parser)]
#[command(name = "qwen-asr", version, about = "Qwen3-ASR with Burn + CUDA (BF16)")]
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

type Backend = Cuda<bf16, i32>;

fn main() -> anyhow::Result<()> {
    env_logger::init();
    let cli = Cli::parse();

    log::info!("Initializing Qwen3-ASR with CUDA backend...");
    let device = burn::backend::cuda::CudaDevice::default();
    let pipeline = AsrPipeline::<Backend>::new(&cli.model_dir, device)?;

    match cli.command {
        Command::Transcribe { input, output } => {
            log::info!("Transcribing: {}", input);
            let text = pipeline.transcribe(&input)?;
            println!("{text}");
            if let Some(out_path) = output {
                std::fs::write(&out_path, &text)?;
            }
        }
    }
    Ok(())
}
