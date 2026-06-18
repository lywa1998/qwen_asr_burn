#!/usr/bin/env python3
"""Validate Qwen3-ASR Python reference against Rust implementation.

Usage:
    python3 tools/validate_asr.py                          # transcribe only
    python3 tools/validate_asr.py --align                  # transcribe + align
    python3 tools/validate_asr.py --model Qwen3-ASR-1.7B   # use 1.7B
"""
import torch, argparse, json, os

from transformers import AutoConfig, AutoModel, AutoProcessor
from qwen_asr.core.transformers_backend import (
    Qwen3ASRConfig, Qwen3ASRForConditionalGeneration, Qwen3ASRProcessor
)
AutoConfig.register("qwen3_asr", Qwen3ASRConfig)
AutoModel.register(Qwen3ASRConfig, Qwen3ASRForConditionalGeneration)
AutoProcessor.register(Qwen3ASRConfig, Qwen3ASRProcessor)

from qwen_asr import Qwen3ASRModel, Qwen3ForcedAligner
from qwen_asr.inference.qwen3_forced_aligner import Qwen3ForceAlignProcessor


def run_transcribe(audio_path: str, model_dir: str) -> list:
    print(f"Loading ASR: {model_dir}")
    model = Qwen3ASRModel.from_pretrained(
        model_dir, dtype=torch.float32, device_map="cpu", max_new_tokens=256,
    )
    print(f"Transcribing: {audio_path}")
    return model.transcribe(audio=audio_path, language="Chinese")


def run_align(audio_path: str, text: str, aligner_dir: str) -> list:
    print(f"Loading aligner: {aligner_dir}")
    model = AutoModel.from_pretrained(aligner_dir, dtype=torch.float32, device_map="cpu")
    processor = AutoProcessor.from_pretrained(aligner_dir)
    aligner = Qwen3ForcedAligner(
        model=model, processor=processor,
        aligner_processor=Qwen3ForceAlignProcessor()
    )
    print(f"Aligning: {text[:60]}...")
    return aligner.align(audio=audio_path, text=text, language="Chinese")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="models/Qwen3-ASR-0.6B")
    p.add_argument("--aligner", default="models/Qwen3-ForcedAligner-0.6B")
    p.add_argument("--audio", default="assets/test_30s.wav")
    p.add_argument("--text", help="Text to align (default: use transcribe output)")
    p.add_argument("--align", action="store_true")
    args = p.parse_args()

    results = run_transcribe(args.audio, args.model)
    r = results[0]
    print(f"\n=== Python Transcribe ({len(r.text)} chars) ===")
    print(r.text)
    print(f"Language: {r.language}")

    if not args.align:
        return

    text = args.text or r.text
    if not text.strip():
        print("Empty text, skipping align")
        return

    align_results = run_align(args.audio, text, args.aligner)
    items = list(align_results[0])
    print(f"\n=== Python Align ({len(items)} words) ===")
    for item in items[:20]:
        print(f"  {item.start_time:.3f}s-{item.end_time:.3f}s  {item.text}")
    if len(items) > 20:
        print(f"  ... ({len(items) - 20} more)")

    out = [{"text": it.text, "start_time": it.start_time, "end_time": it.end_time}
           for it in items]
    stem = os.path.splitext(os.path.basename(args.audio))[0]
    path = f"{stem}_python_align.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: {path}")


if __name__ == "__main__":
    main()
