#!/usr/bin/env python3
import torch

# Fix for PyTorch 2.6+ safe deserialization issues with pyannote checkpoints
torch.serialization.add_safe_globals([
    __import__("omegaconf").OmegaConf,
    __import__("omegaconf.listconfig").listconfig.ListConfig,
])

import os
import sys
import json
import argparse
from typing import List, Dict, Any, Optional


def format_timestamp(seconds: float) -> str:
    # Format seconds as HH:MM:SS.mmm
    if seconds is None:
        return "00:00:00.000"
    total_ms = int(round(float(seconds) * 1000))
    total_seconds, ms = divmod(total_ms, 1000)
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{ms:03d}"


def print_environment() -> None:
    # Print basic info about Python, Torch, and GPU availability.
    print("=== Environment info ===")
    print(f"Python version : {sys.version.split()[0]}")
    print(f"Torch version  : {torch.__version__}")
    cuda_version = getattr(torch.version, "cuda", None)
    if cuda_version:
        print(f"Torch CUDA     : {cuda_version}")
    else:
        print("Torch CUDA     : not compiled with CUDA support")

    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        print(f"CUDA available : YES ({num_gpus} GPU(s) detected)")
        for idx in range(num_gpus):
            name = torch.cuda.get_device_name(idx)
            print(f"  GPU {idx}: {name}")
    else:
        print("CUDA available : NO (running on CPU)")


def choose_device(requested: Optional[str] = None) -> str:
    # Select device string ('cuda' or 'cpu').
    if requested:
        dev = requested
    else:
        dev = "cuda" if torch.cuda.is_available() else "cpu"

    if dev.startswith("cuda") and not torch.cuda.is_available():
        print("WARNING: CUDA requested but no GPU is available. Falling back to CPU.")
        dev = "cpu"

    print(f"Using device   : {dev}")
    return dev


def get_hf_token(skip_diarization: bool) -> Optional[str]:
    # Read HF_TOKEN from environment and handle errors.
    token = os.getenv("HF_TOKEN")
    if token:
        print("HF_TOKEN       : detected (not printed for security)")
        return token

    msg = (
        "HF_TOKEN environment variable is not set.\n"
        "It is required to download pyannote speaker-diarization models from Hugging Face.\n"
        "Create a token at https://huggingface.co/settings/tokens with 'read' access,\n"
        "accept the terms for pyannote models, then set it, for example:\n"
        "  Linux/macOS (bash):  export HF_TOKEN=hf_xxx\n"
        "  Windows (CMD)      :  set HF_TOKEN=hf_xxx\n"
        "  Windows (PowerShell): $env:HF_TOKEN=\"hf_xxx\""
    )

    if skip_diarization:
        print("WARNING: " + msg)
        print("Proceeding without diarization because --skip_diarization was provided.")
        return None
    else:
        print("ERROR: " + msg)
        sys.exit(1)


def create_pyannote_pipeline(hf_token: str, device: str):
    # Create pyannote speaker diarization pipeline on the requested device.
    from pyannote.audio import Pipeline  # imported lazily

    print("Loading pyannote speaker-diarization-3.1 pipeline ...")
    try:
        try:
            pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token=hf_token,
            )
        except TypeError:
            # Newer pyannote versions use 'token' instead of 'use_auth_token'
            pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                token=hf_token,
            )
    except Exception as e:
        print("ERROR: failed to load pyannote speaker-diarization-3.1 pipeline.")
        print("       Check that HF_TOKEN is valid and has access to the model.")
        print(f"       Original exception: {e}")
        sys.exit(1)

    try:
        pipeline.to(torch.device(device))
    except Exception as e:
        print(f"WARNING: could not move pyannote pipeline to device '{device}'. It will run on CPU.")
        print(f"         Reason: {e}")
    return pipeline


def assign_speakers_to_words(diarization, aligned_result: Dict[str, Any]) -> List[Dict[str, Any]]:
    # Assign speaker labels from pyannote diarization to WhisperX word-level timestamps.
    diar_segments: List[Dict[str, Any]] = []
    # diarization is a pyannote.core.Annotation
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        diar_segments.append({
            "start": float(turn.start),
            "end": float(turn.end),
            "speaker": str(speaker),
        })
    diar_segments.sort(key=lambda s: s["start"])

    words: List[Dict[str, Any]] = []
    for seg in aligned_result.get("segments", []):
        for w in seg.get("words", []):
            if "start" not in w or "end" not in w:
                # Skip words without timing (should not normally happen)
                continue
            words.append({
                "word": w.get("word", "").strip(),
                "start": float(w["start"]),
                "end": float(w["end"]),
            })
    words.sort(key=lambda w: w["start"])

    def find_speaker_for_time(t: float) -> str:
        for seg in diar_segments:
            if seg["start"] <= t < seg["end"]:
                return seg["speaker"]
        return "SPEAKER_UNKNOWN"

    for w in words:
        mid = 0.5 * (w["start"] + w["end"])
        w["speaker"] = find_speaker_for_time(mid)

    return words


def build_speaker_segments(words: List[Dict[str, Any]], max_gap: float = 1.0) -> List[Dict[str, Any]]:
    # Group consecutive words by speaker into segments.
    if not words:
        return []

    segments: List[Dict[str, Any]] = []
    current = {
        "speaker": words[0]["speaker"],
        "start": words[0]["start"],
        "end": words[0]["end"],
        "words": [words[0]],
    }

    for prev, w in zip(words, words[1:]):
        gap = w["start"] - prev["end"]
        if w["speaker"] != current["speaker"] or gap > max_gap:
            current["end"] = prev["end"]
            current["text"] = " ".join([x["word"] for x in current["words"]]).strip()
            segments.append(current)
            current = {
                "speaker": w["speaker"],
                "start": w["start"],
                "end": w["end"],
                "words": [w],
            }
        else:
            current["words"].append(w)

    last = words[-1]
    current["end"] = last["end"]
    current["text"] = " ".join([x["word"] for x in current["words"]]).strip()
    segments.append(current)
    return segments


def save_text_result(segments: List[Dict[str, Any]], path: str) -> None:
    # Save human-readable transcript with speakers to a .txt file.
    lines = []
    for seg in segments:
        start_ts = format_timestamp(seg["start"])
        end_ts = format_timestamp(seg["end"])
        speaker = seg["speaker"]
        text = seg.get("text", "").strip()
        lines.append(f"[{start_ts} --> {end_ts}] {speaker}: {text}")
    content = "\n".join(lines)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"Wrote text transcript to {path}")


def save_json_result(audio_path: str, language: Optional[str], segments: List[Dict[str, Any]], path: str) -> None:
    # Save structured JSON with segments & words.
    out: Dict[str, Any] = {
        "audio_path": os.path.abspath(audio_path),
        "language": language,
        "segments": segments,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"Wrote JSON transcript to {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Automatic transcription (WhisperX large-v2) with pyannote diarization."
    )
    parser.add_argument(
        "--audio_path",
        type=str,
        default=os.path.join("audio", "input.wav"),
        help="Path to input audio file (wav, mp3, m4a, flac, ...). Default: audio/input.wav",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use: 'cuda' or 'cpu'. Default: auto-detect.",
    )
    parser.add_argument(
        "--compute_type",
        type=str,
        default=None,
        help="WhisperX compute type, e.g. 'float16', 'float32', 'int8'. "
             "Default: 'float16' on GPU, 'int8' on CPU.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size for WhisperX transcription.",
    )
    parser.add_argument(
        "--skip_diarization",
        action="store_true",
        help="Run only ASR (no diarization). result.txt/json will not contain speakers.",
    )
    return parser.parse_args()


def main() -> None:
    print_environment()
    args = parse_args()
    device = choose_device(args.device)

    if args.compute_type is not None:
        compute_type = args.compute_type
    else:
        compute_type = "float16" if device.startswith("cuda") else "int8"
    print(f"Compute type   : {compute_type}")

    audio_path = args.audio_path
    if not os.path.isfile(audio_path):
        print(f"ERROR: audio file not found: {audio_path}")
        sys.exit(1)

    # HF token (may exit if missing and diarization is required)
    hf_token = get_hf_token(skip_diarization=args.skip_diarization)

    # Import heavy dependencies lazily after basic checks
    import whisperx

    # 1. Load and transcribe audio with WhisperX
    print(f"Loading audio : {audio_path}")
    audio = whisperx.load_audio(audio_path)

    print("Loading WhisperX ASR model (large-v2) ...")
    model = whisperx.load_model(
        "large-v2",
        device=device,
        compute_type=compute_type,
    )

    print("Running transcription ...")
    asr_result = model.transcribe(audio, batch_size=args.batch_size)
    language = asr_result.get("language", None)
    if language:
        print(f"Detected language: {language}")

    # 2. Alignment for accurate word-level timestamps
    print("Loading alignment model ...")
    align_model, metadata = whisperx.load_align_model(
        language_code=language,
        device=device,
    )

    print("Running alignment ...")
    aligned_result = whisperx.align(
        asr_result["segments"],
        align_model,
        metadata,
        audio,
        device,
        return_char_alignments=False,
    )

    if args.skip_diarization:
        print("Skipping diarization. All text will be tagged as SPEAKER_UNKNOWN.")
        # Fake speaker assignment
        words: List[Dict[str, Any]] = []
        for seg in aligned_result.get("segments", []):
            for w in seg.get("words", []):
                if "start" not in w or "end" not in w:
                    continue
                words.append({
                    "word": w.get("word", "").strip(),
                    "start": float(w["start"]),
                    "end": float(w["end"]),
                    "speaker": "SPEAKER_UNKNOWN",
                })
        words.sort(key=lambda w: w["start"])
        speaker_segments = build_speaker_segments(words)
    else:
        # 3. Diarization with pyannote
        pipeline = create_pyannote_pipeline(hf_token, device)
        print("Running pyannote diarization ...")
        diarization = pipeline(audio_path)

        print("Assigning speakers to words ...")
        words = assign_speakers_to_words(diarization, aligned_result)
        speaker_segments = build_speaker_segments(words)

    # 4. Save outputs
    save_text_result(speaker_segments, "result.txt")
    save_json_result(audio_path, language, speaker_segments, "result.json")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted by user.")
        sys.exit(1)
