mod align_pipeline;
mod audio;
mod config;
mod model;
mod pipeline;
mod text_processor;
mod tokenizer;

use align_pipeline::AlignPipeline;
use burn::backend::{cuda::CudaDevice, Cuda};
use burn::tensor::bf16;
use clap::Parser;
use pipeline::AsrPipeline;

#[derive(Parser)]
#[command(
    name = "qwen-asr",
    version,
    about = "Qwen3-ASR with Burn + CUDA (BF16)"
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
    Align {
        #[arg(short, long)]
        input: String,
        #[arg(short, long)]
        text: String,
        #[arg(short, long, default_value = "English")]
        language: String,
        #[arg(short = 'F', long, default_value = "text")]
        format: String,
    },
}

type Backend = Cuda<bf16, i32>;

fn main() -> anyhow::Result<()> {
    env_logger::init();
    let cli = Cli::parse();

    let device = CudaDevice::default();

    match cli.command {
        Command::Transcribe { input, output } => {
            log::info!("Initializing Qwen3-ASR...");
            let pipeline = AsrPipeline::<Backend>::new(&cli.model_dir, device)?;
            log::info!("Transcribing: {}", input);
            let text = pipeline.transcribe(&input)?;
            println!("{text}");
            if let Some(out_path) = output {
                std::fs::write(&out_path, &text)?;
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
    }
    Ok(())
}
