# qwen-subtitle — Design Document

长视频（1-2h）字幕翻译 CLI 工具。提取视频音频 → ASR 转录 → 强制对齐生成 SRT → 翻译 → 软字幕视频。

## Pipeline Overview

```
video.mp4
  │ avio (decode + resample → 16kHz f32 mono)
  ▼
PCM f32 (16kHz mono)
  │ earshot (VAD)
  ▼
voice segments [(t0, t1), (t1, t2), ...]
  │
  │  ┌─ segment audio ── Qwen3-ASR-0.6B ── text ──┐
  │  │   (Burn, CUDA BF16)                          │
  │  └──────────────────────────────────────────────┘
  │  ┌─ segment audio + text ── Qwen3-ForcedAligner-0.6B ── timestamps ──┐
  │  │   (Burn, CUDA BF16)                                                  │
  │  └──────────────────────────────────────────────────────────────────────┘
  │
  ▼
source.srt (word-level timestamps, source language)
  │
  │ per-subtitle batch ── Hy-MT-1.8B ── translated text ──┐
  │   (Burn, CUDA BF16)                                     │
  └─────────────────────────────────────────────────────────┘
  │
  ▼
bilingual.srt (source + target)
  │ avio (mux soft subtitle)
  ▼
video_with_subtitle.mkv (软字幕，不重编码)
```

## Technology Stack

| 层 | Crate | 说明 |
|---|---|---|
| 视频解码+重采样 | `avio` (0.14) | FFmpeg libswresample，直接出 16kHz f32 单声道 |
| VAD | `earshot` (1.1) | 纯 Rust 嵌入式神经网络，RTF 0.0007 |
| 重采样（WAV 输入） | `rubato` (0.16) | 独立 `transcribe`/`align` 命令的 WAV 重采样，已有 |
| 特征提取 | `rustfft` (6.3) | STFT + mel filterbank，已有实现 |
| ASR 推理 | `burn` (0.21) | CUDA BF16 / Metal F32，Qwen3-ASR-0.6B |
| 强制对齐 | `burn` (0.21) | CUDA BF16 / Metal F32，Qwen3-ForcedAligner-0.6B |
| 翻译 | `burn` (0.21) | CUDA BF16 / Metal F32，Hy-MT-1.8B（待迁移 Burn 实现） |
| 软字幕压入 | `avio` | 视频拷贝 + SRT 轨道混流，不重编码 |
| 分词 | `tokenizers` (0.22) | HuggingFace BPE |
| CLI | `clap` (4) | derive-based |

纯 Rust 全链路，唯一非 Rust 依赖是 GPU 驱动（CUDA 或 Metal）和 FFmpeg 共享库（avio 底层）。

## Model Loading Strategy

按 Command 按需加载，不预加载全部模型：

| Command | 需要的模型 | GPU 显存 |
|---|---|---|
| `transcribe` | Qwen3-ASR-0.6B | ~2.5 GB |
| `align` | Qwen3-ForcedAligner-0.6B | ~3.0 GB |
| `translate`（待实现） | Hy-MT-1.8B | ~5.0 GB |
| `run`（全流水线） | 依次加载三个模型 | 峰值 ~5.0 GB |

每个命令运行时只加载当前步骤所需的模型，命令结束后释放。`run` 全流水线按步骤串行执行，每个步骤独立加载/释放，因此峰值显存为最大单模型的需求（~5.0 GB），而非三个模型总和。

目标设备 NVIDIA 4070 Super 16GB，余量充足。

## VAD Segmentation

earshot 逐帧推理（16ms/帧，256 samples @ 16kHz），输出 0-1 语音概率。

后处理规则：
- 连续语音帧合并为一个语音段
- 间距 < 0.5s 的相邻段合并（同说话人）
- 单段最大 30s，超出在最近静音点附近切开（1s overlap）
- 单段最小 0.5s，短于阈值的静音间隙忽略

段的绝对时间戳（相对原始视频）在 ASR 和对齐阶段持续传递，确保 SRT 时间正确。

## CLI Design

### 全流水线

```bash
qwen-subtitle run video.mp4 --source zh --target en [-o output.mkv]
```

### 分步执行

```bash
qwen-subtitle extract video.mp4 -o audio.wav
  # 提取 16kHz mono 音频

qwen-subtitle transcribe audio.wav --segments segments.json -o transcript.txt
  # VAD 分段 + ASR 转录

qwen-subtitle align audio.wav -t transcript.txt --segments segments.json -o source.srt
  # 强制对齐 → 源语言 SRT

qwen-subtitle translate source.srt --source zh --target en -o bilingual.srt
  # 翻译 → 双语 SRT

qwen-subtitle embed video.mp4 bilingual.srt -o video_subtitled.mkv
  # 压入软字幕
```

### 公共参数

- `-m, --model-dir <PATH>`：模型根目录，包含 `Qwen3-ASR-0.6B/`、`Qwen3-ForcedAligner-0.6B/`、`Hy-MT-1.8B/` 三个子目录
- `--device <ID>`：CUDA 设备编号（`cuda` feature，默认 0）

### Feature Flags

编译时选择后端：

```bash
# CUDA（默认）
cargo build --features cuda

# Metal（macOS Apple Silicon）
cargo build --no-default-features --features metal
```

### 断点续跑

每步输出中间文件到工作目录。`run` 命令自动检测已有文件跳过对应步骤：

| 步骤 | 检查点文件 | 跳过条件 |
|---|---|---|
| 音频提取 | `audio.wav` | 文件存在 |
| ASR 转录 | `segments.json` + `transcript.txt` | 文件存在 |
| 强制对齐 | `source.srt` | 文件存在 |
| 翻译 | `bilingual.srt` | 文件存在 |
| 软字幕 | `output.mkv` | 文件存在（`--force` 覆盖） |

`-f, --force` 参数强制重跑全部步骤。

## Module Design

```
src/
  main.rs                  — CLI entry point，clap derive
  pipeline_orchestrator.rs — 顶层编排，管理模型生命周期和步骤调度
  pipeline.rs              — AsrPipeline<B>，音频 → 文本（已有，加分段批处理）
  align_pipeline.rs        — AlignPipeline<B>，音频+文本 → 词级时间戳（已有，加分段+偏移累积）
  translate_pipeline.rs    — TranslatePipeline<B>，文本 → 翻译文本（新建，Hy-MT Burn 推理）
  model.rs                 — NN 模块定义（已有，扩展 Hy-MT decoder 模块）
  audio.rs                 — mel 特征提取（已有）
  tokenizer.rs             — HuggingFace 分词器封装（已有）
  config.rs                — JSON 配置反序列化（已有）
  text_processor.rs        — 文本处理、时间戳修正（已有，加 SRT 格式化）
  vad.rs                   — earshot VAD 封装 + 段合并逻辑（新建）
  srt.rs                   — SRT 文件读写、双语合并（新建）
  video.rs                 — avio 封装：音频解码 + 软字幕压入（新建）
```

## SRT Format

```
1
00:00:01,200 --> 00:00:02,500
今天天气真好
The weather is really nice today.

2
00:00:02,800 --> 00:00:05,100
我们去公园散步吧
Let's go for a walk in the park.
```

双语 SRT：每段包含原文和译文，中间以空行分隔。

## Translation Strategy

以 SRT 字幕条目为单位翻译。每 3-5 句构成一个翻译 batch，提供前后句作为上下文：

```
System: 你是专业的字幕翻译。将用户输入翻译为英语。只输出译文，不要解释。
Previous: I went to the park yesterday.
Current: 我们在湖边散步。
Next: 看到了很多漂亮的鸟。
```

单句最大 token 数有限（字幕句子短），无需 KV cache 复用，逐 batch 前向传播即可。

## Error Handling

- avio 解码失败 → 提示缺少 FFmpeg 或文件格式不支持
- VAD 无语音段 → 提示无有效音频
- ASR 输出空文本 → 标记该段为空，不送对齐
- 对齐 LIS 修正异常 > 30% → 标记该段为低置信度
- 翻译失败 → 保留原文，译文留空
- 中间文件异常 → 用户可删除对应文件后重跑
