---
name: pipeline-srt
description: Generate SRT subtitles from project video/audio by running the current Rust CLI in explicit stages: extract -> transcribe -> align -> translate. Use this skill whenever the user asks for video subtitles, SRT files, bilingual captions, subtitle translation, aligned timestamps, mp4-to-srt, audio-to-srt, 字幕, 双语字幕, or asks to process media through this repository. Prefer the staged CLI workflow and do not create a monolithic all-in-one pipeline script.
---

# Pipeline SRT Skill

Use this skill to turn a video or audio file into an SRT subtitle file with the repository's current commands. The workflow is intentionally staged:

```text
video/audio
  -> extract      # video -> 16 kHz mono WAV; skip for suitable audio input
  -> transcribe   # WAV -> transcript lines + VAD segments
  -> align        # per segment WAV + matching text -> word/char timestamps
  -> translate    # optional per subtitle phrase -> target-language text
  -> assemble SRT # use the staged outputs; do not re-run the full pipeline inside a script
```

## Core rule: stage the work

Run the project subcommands as separate, observable steps. Do **not** create a single integrated script that performs extract, transcribe, align, and translate end-to-end.

Why: the model commands are slow and each stage produces artifacts that need inspection (`.wav`, `_segments.json`, `_transcript.txt`, align JSON, translated text). Staging makes failures recoverable and avoids hiding model or timestamp problems inside a wrapper.

Allowed helper code:
- Small, one-purpose helpers for mechanical file handling are OK, such as reading existing align JSON files and writing SRT, or formatting timestamps.
- Helpers must not call all model subcommands as an end-to-end pipeline.
- Prefer direct CLI commands for model inference stages.

## Models and defaults

Use explicit model directories unless the user provides different paths:

```bash
ASR_MODEL=models/Qwen3-ASR-0.6B
ALIGN_MODEL=models/Qwen3-ForcedAligner-0.6B
MT_MODEL=models/Hy-MT2-1.8B
```

Ask for missing media path, source language, or target language only when they are not inferable. Default target language for bilingual subtitles is `English`.

## Stage 1 — Extract audio

If the input is a video (`.mp4`, `.mkv`, `.mov`, `.avi`, etc.), extract WAV first:

```bash
cargo run -- extract input.mp4 -o input.wav
```

Current behavior:
- Output is 16 kHz mono WAV.
- If `-o` is omitted, output defaults to `<input_stem>.wav`.

If the user already provides a WAV/audio file, skip extraction unless conversion is needed. The project WAV loader can resample for transcribe/align, but using a clean 16 kHz mono WAV keeps artifacts predictable.

## Stage 2 — Transcribe

Run ASR on the WAV and save the staged transcript artifacts:

```bash
cargo run -- --model-dir "$ASR_MODEL" \
  transcribe input.wav \
  --language Chinese
```

Options to use as needed:

```bash
-o input_transcript.txt        # override transcript output path
--language English             # force source language
-c "domain/context prompt"     # context prompt if the user provides one
--save-srt                     # optional rough segment-level SRT only
```

Expected outputs:
- `<stem>_transcript.txt`: one transcript line per VAD segment when speech is detected.
- `<stem>_segments.json`: VAD segment timings.
- `<stem>.srt`: only if `--save-srt` is passed.

Important: `--save-srt` produces a quick segment-level SRT based on VAD boundaries. For the aligned subtitle workflow, continue to Stage 3 instead of treating this as the final result.

## Stage 3 — Align per segment

Use `_segments.json` and `_transcript.txt` together. Each transcript line corresponds to the matching VAD segment in order.

For each segment:

1. Read its `start_secs` and `end_secs` from `<stem>_segments.json`.
2. Take the matching non-empty transcript line from `<stem>_transcript.txt`.
3. Slice the segment audio with `ffmpeg` or another single-purpose audio command:

```bash
ffmpeg -y -i input.wav \
  -ss <start_secs> \
  -to <end_secs> \
  -ac 1 -ar 16000 \
  work/segment_000.wav
```

4. Run forced alignment for that segment and save stdout as JSON:

```bash
cargo run -- --model-dir "$ALIGN_MODEL" \
  align \
  -i work/segment_000.wav \
  -t "<matching transcript line>" \
  -l Chinese \
  -F json > work/segment_000.align.json
```

Current align behavior:
- `align` prints to stdout; it has no `-o` flag.
- JSON items look like `{ "text": "...", "start_time": 0.0, "end_time": 0.42 }`.
- Times are relative to the segment WAV. Add the segment's `start_secs` to get absolute SRT timestamps.
- For `Chinese`, `Japanese`, and `Korean`, alignment is mostly character-level; for other languages it is word-level.

## Stage 4 — Group aligned tokens into SRT phrases

Build readable SRT entries from the aligned tokens. Target phrase-level readability, not one token per caption.

Suggested grouping rules:
- Never merge across VAD segment boundaries.
- Start time = first token start + segment offset.
- End time = last token end + segment offset.
- Break on sentence punctuation: `。！？.!?;；` and usually `，,` when the phrase is already long enough.
- Also break after about 1–4 seconds, or when a line becomes too long to read comfortably.
- Drop empty or obviously invalid align items.

SRT timestamp format:

```text
HH:MM:SS,mmm --> HH:MM:SS,mmm
```

Monolingual entry:

```srt
12
00:00:19,260 --> 00:00:22,540
这个POCKE4P的画质比我们想的要好一些
```

## Stage 5 — Translate, if bilingual output is requested

Translate after phrase grouping so each translated line matches one SRT entry.

For exact entry-to-entry bilingual subtitles, call the translate command per phrase:

```bash
cargo run -- translate "<source phrase>" \
  -t English \
  -M "$MT_MODEL"
```

If saving each translation separately:

```bash
cargo run -- translate "<source phrase>" \
  -t English \
  -M "$MT_MODEL" \
  -o work/entry_001.en.txt
```

For many entries, batching is only acceptable if you preserve a clear delimiter and verify that the number of translated entries matches the source entries. If the count does not match, fall back to per-entry translation for the affected range.

Bilingual SRT entry format:

```srt
12
00:00:19,260 --> 00:00:22,540
这个POCKE4P的画质比我们想的要好一些
The image quality of the POCKE4P is better than we thought.
```

## Recommended artifact layout

Keep intermediate files so the user can inspect or resume a failed stage:

```text
input.wav
input_transcript.txt
input_segments.json
work/
  segment_000.wav
  segment_000.align.json
  segment_001.wav
  segment_001.align.json
input.aligned.srt
input.bilingual.srt
```

## Quality checks before reporting done

Check the outputs at each stage:

1. Extraction: WAV duration is plausible for the source video.
2. Transcription: transcript line count matches or is safely no greater than segment count; investigate large mismatches.
3. Alignment: each non-empty segment has align JSON; timestamps are non-decreasing after offsetting.
4. SRT assembly: entries have increasing `HH:MM:SS,mmm` times and no empty captions.
5. Translation: bilingual entries preserve source text and add the translation below it; entry counts match.

When reporting results, include the final SRT path(s), mention any skipped stage, and state whether the output is segment-level (`--save-srt`) or forced-aligned phrase-level.
