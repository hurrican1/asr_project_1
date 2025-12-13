#!/usr/bin/env python3

# === Required by user: PyTorch 2.6+ safe unpickling fix (must be at top) ===
import torch
torch.serialization.add_safe_globals([
    __import__("omegaconf").OmegaConf,
    __import__("omegaconf.listconfig").listconfig.ListConfig,
])

import os
import sys
import json
import argparse
import shutil
import subprocess
import tempfile
import uuid
from typing import Any, Dict, List, Optional


# --- RunPod / HF download robustness ---
# If the environment forces hf_transfer but the package is not installed, downloads fail.
# We disable it for stability (normal download will be used).
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

# Lightning (used by pyannote) can be blocked by torch.load(weights_only=True) in torch>=2.6.
# This env var makes lightning force weights_only=False when not explicitly passed.
# (We download official HF checkpoints, so it's typically trusted.)
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")


# --------------------------- utils ---------------------------

def print_environment() -> None:
    print("=== Environment info ===")
    print(f"Python version : {sys.version.split()[0]}")
    print(f"Torch version  : {torch.__version__}")
    cuda_ver = getattr(torch.version, "cuda", None)
    print(f"Torch CUDA     : {cuda_ver if cuda_ver else 'not available'}")

    cudnn_ver = None
    try:
        cudnn_ver = torch.backends.cudnn.version()
    except Exception:
        cudnn_ver = None
    print(f"cuDNN version  : {cudnn_ver if cudnn_ver else 'unknown'}")

    if torch.cuda.is_available():
        n = torch.cuda.device_count()
        print(f"CUDA available : YES ({n} GPU(s) detected)")
        for i in range(n):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("CUDA available : NO (running on CPU)")


def choose_device(requested: Optional[str]) -> str:
    if requested:
        dev = requested.strip().lower()
    else:
        dev = "cuda" if torch.cuda.is_available() else "cpu"

    if dev.startswith("cuda") and not torch.cuda.is_available():
        print("WARNING: CUDA requested but not available. Falling back to CPU.")
        dev = "cpu"

    print(f"Using device   : {dev}")
    return dev


def require_ffmpeg() -> None:
    if shutil.which("ffmpeg") is None:
        print("ERROR: ffmpeg not found in PATH.")
        print("Install it:")
        print("  Ubuntu/Debian: sudo apt update && sudo apt install -y ffmpeg")
        print("  macOS: brew install ffmpeg")
        sys.exit(1)


def convert_to_wav_16k_mono(input_path: str) -> (str, bool):
    """
    Convert any input audio into a temporary 16kHz mono WAV file.
    This makes pyannote pipelines stable (they expect 16k mono) and avoids codec issues.
    Returns: (wav_path, is_temporary)
    """
    require_ffmpeg()

    # Always convert to ensure 16k/mono for pyannote and consistent timings
    tmp_wav = os.path.join(tempfile.gettempdir(), f"asr_tmp_{uuid.uuid4().hex}.wav")
    cmd = [
        "ffmpeg", "-y", "-nostdin",
        "-i", input_path,
        "-ac", "1",
        "-ar", "16000",
        "-vn",
        tmp_wav
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        err = e.stderr.decode(errors="ignore") if e.stderr else str(e)
        print("ERROR: ffmpeg conversion failed.")
        print(err)
        sys.exit(1)

    return tmp_wav, True


def load_dotenv_if_exists(env_path: str = ".env") -> None:
    """
    Optional convenience: if .env exists and HF_TOKEN not set, read HF_TOKEN=... from it.
    .env should be in .gitignore (do NOT commit secrets).
    """
    if os.getenv("HF_TOKEN"):
        return
    if not os.path.isfile(env_path):
        return

    try:
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                k = k.strip()
                v = v.strip().strip('"').strip("'")
                if k == "HF_TOKEN" and v:
                    os.environ["HF_TOKEN"] = v
                    return
    except Exception:
        # If .env parsing fails, just ignore and rely on env var
        return


def get_hf_token(skip_diarization: bool) -> Optional[str]:
    load_dotenv_if_exists(".env")

    token = os.getenv("HF_TOKEN")
    if token:
        print("HF_TOKEN       : detected (not printed for security)")
        return token

    msg = (
        "HF_TOKEN environment variable is not set.\n"
        "It is required to download pyannote speaker-diarization models from Hugging Face.\n"
        "Create a token at https://huggingface.co/settings/tokens with 'read' access,\n"
        "accept terms for pyannote models, then set it:\n"
        "  export HF_TOKEN=hf_xxx\n"
        "Or create a local .env file (NOT committed) with:\n"
        "  HF_TOKEN=hf_xxx\n"
    )

    if skip_diarization:
        print("WARNING: " + msg)
        print("Proceeding without diarization because --skip_diarization was provided.")
        return None

    print("ERROR: " + msg)
    sys.exit(1)


def format_ts(seconds: float) -> str:
    ms = int(round(float(seconds) * 1000))
    s, ms = divmod(ms, 1000)
    m, s = divmod(s, 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


# --------------------------- diarization helpers ---------------------------

def create_pyannote_pipeline(hf_token: str, requested_device: str):
    """
    Loads pyannote speaker diarization pipeline and tries to move it to GPU.
    If GPU move fails (e.g., cuDNN mismatch), it will FORCE a clean CPU state to avoid mixed-device crashes.
    """
    from pyannote.audio import Pipeline

    print("Loading pyannote speaker-diarization-3.1 pipeline ...")

    try:
        try:
            pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token=hf_token,
            )
        except TypeError:
            pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                token=hf_token,
            )
    except Exception as e:
        print("ERROR: Failed to load pyannote pipeline.")
        print("Make sure token is valid and model terms are accepted on Hugging Face.")
        print(f"Original error: {e}")
        sys.exit(1)

    # Always start on CPU (clean baseline)
    try:
        pipeline.to(torch.device("cpu"))
    except Exception as e:
        print("ERROR: Could not move pyannote pipeline to CPU.")
        print(f"Original error: {e}")
        sys.exit(1)

    # Try GPU if requested and available
    if requested_device.startswith("cuda") and torch.cuda.is_available():
        try:
            pipeline.to(torch.device(requested_device))
            print(f"pyannote device : {requested_device}")
        except Exception as e:
            print(f"WARNING: could not move pyannote pipeline to device '{requested_device}'. Falling back to CPU.")
            print(f"         Reason: {e}")
            # Force back to CPU to avoid mixed-device error later
            pipeline.to(torch.device("cpu"))
            print("pyannote device : cpu")
    else:
        print("pyannote device : cpu")

    return pipeline


def diarize_with_retry(pipeline, wav_path: str):
    """
    Run diarization. If a mixed-device runtime error happens, force CPU and retry once.
    """
    try:
        return pipeline(wav_path)
    except RuntimeError as e:
        msg = str(e)
        if "Expected all tensors to be on the same device" in msg:
            print("WARNING: pyannote device mismatch detected. Forcing CPU diarization and retrying once ...")
            try:
                pipeline.to(torch.device("cpu"))
            except Exception:
                pass
            return pipeline(wav_path)
        raise


def assign_word_speakers(diarization, aligned: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Assign speaker label to each word by matching word mid-time with diarization segments.
    Returns a flat list of words {word,start,end,speaker}.
    """
    # Build diarization intervals
    diar = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        diar.append((float(turn.start), float(turn.end), str(speaker)))
    diar.sort(key=lambda x: x[0])

    def speaker_at(t: float) -> str:
        for s, e, spk in diar:
            if s <= t < e:
                return spk
        return "SPEAKER_UNKNOWN"

    words: List[Dict[str, Any]] = []
    for seg in aligned.get("segments", []) or []:
        for w in (seg.get("words") or []):
            if "start" not in w or "end" not in w:
                continue
            word_text = (w.get("word") or "").strip()
            if not word_text:
                continue
            start = float(w["start"])
            end = float(w["end"])
            mid = 0.5 * (start + end)
            words.append({
                "word": word_text,
                "start": start,
                "end": end,
                "speaker": speaker_at(mid),
            })
    words.sort(key=lambda x: x["start"])
    return words


def merge_words_into_speaker_segments(words: List[Dict[str, Any]], max_gap: float = 1.0) -> List[Dict[str, Any]]:
    """
    Merge consecutive words into speaker segments.
    """
    if not words:
        return []

    out: List[Dict[str, Any]] = []
    cur = {
        "speaker": words[0]["speaker"],
        "start": words[0]["start"],
        "end": words[0]["end"],
        "words": [words[0]],
    }

    for prev, w in zip(words, words[1:]):
        gap = w["start"] - prev["end"]
        if w["speaker"] != cur["speaker"] or gap > max_gap:
            cur["end"] = prev["end"]
            cur["text"] = " ".join(x["word"] for x in cur["words"]).strip()
            out.append(cur)
            cur = {
                "speaker": w["speaker"],
                "start": w["start"],
                "end": w["end"],
                "words": [w],
            }
        else:
            cur["words"].append(w)

    cur["end"] = words[-1]["end"]
    cur["text"] = " ".join(x["word"] for x in cur["words"]).strip()
    out.append(cur)
    return out


# --------------------------- output ---------------------------

def write_result_txt(segments: List[Dict[str, Any]], path: str = "result.txt") -> None:
    lines = []
    for s in segments:
        lines.append(f"[{format_ts(s['start'])} --> {format_ts(s['end'])}] {s['speaker']}: {s.get('text','')}")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Wrote {path}")


def write_result_json(audio_path: str, language: Optional[str], segments: List[Dict[str, Any]], path: str = "result.json") -> None:
    payload = {
        "audio_path": os.path.abspath(audio_path),
        "language": language,
        "segments": segments,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"Wrote {path}")


# --------------------------- cli ---------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="WhisperX large-v2 + pyannote diarization 3.1")
    p.add_argument("--audio_path", type=str, required=True, help="Path to input audio file")
    p.add_argument("--device", type=str, default=None, help="ASR device: cuda/cpu (default auto)")
    p.add_argument("--compute_type", type=str, default=None, help="float16/float32/int8 (default auto)")
    p.add_argument("--batch_size", type=int, default=16, help="WhisperX batch size")
    p.add_argument("--language", type=str, default=None, help="Language code (e.g., ru, en). If not set -> detect")
    p.add_argument("--skip_diarization", action="store_true", help="Run ASR only (no diarization)")
    p.add_argument(
        "--asr_vad",
        type=str,
        default="silero",
        choices=["silero", "pyannote"],
        help="VAD method inside WhisperX for ASR segmentation. Default silero (recommended).",
    )
    p.add_argument(
        "--diarization_device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Diarization device preference. auto=use cuda if available else cpu.",
    )
    p.add_argument(
        "--max_speaker_gap",
        type=float,
        default=1.0,
        help="Max silence gap (sec) to keep words in the same speaker segment.",
    )
    return p.parse_args()


# --------------------------- main ---------------------------

def main() -> None:
    args = parse_args()
    print_environment()

    if not os.path.isfile(args.audio_path):
        print(f"ERROR: audio file not found: {args.audio_path}")
        sys.exit(1)

    asr_device = choose_device(args.device)
    compute_type = args.compute_type or ("float16" if asr_device.startswith("cuda") else "int8")
    print(f"Compute type   : {compute_type}")

    # HF token needed for diarization
    hf_token = get_hf_token(skip_diarization=args.skip_diarization)

    # Convert to stable format for both ASR and diarization
    wav_path, is_tmp = convert_to_wav_16k_mono(args.audio_path)

    try:
        import whisperx

        print(f"Loading audio : {wav_path}")
        audio = whisperx.load_audio(wav_path)

        print("Loading WhisperX ASR model (large-v2) ...")
        model = whisperx.load_model(
            "large-v2",
            device=asr_device,
            compute_type=compute_type,
            vad_method=args.asr_vad,  # IMPORTANT: default silero to avoid legacy pyannote VAD issues
        )

        print("Running transcription ...")
        result = model.transcribe(audio, batch_size=args.batch_size, language=args.language)
        language = result.get("language", args.language)
        if language:
            print(f"Detected language: {language}")

        # Alignment (word-level)
        aligned = result
        word_level_ok = False
        try:
            print("Loading alignment model ...")
            align_model, metadata = whisperx.load_align_model(language_code=language, device=asr_device)
            print("Running alignment ...")
            aligned = whisperx.align(
                result["segments"],
                align_model,
                metadata,
                audio,
                asr_device,
                return_char_alignments=False,
            )
            word_level_ok = True
        except Exception as e:
            print("WARNING: alignment failed, fallback to segment-level.")
            print(f"         Reason: {e}")
            aligned = result
            word_level_ok = False

        # Build output segments
        if args.skip_diarization:
            print("Skipping diarization. Speakers will be SPEAKER_UNKNOWN.")
            segments_out: List[Dict[str, Any]] = []
            for s in aligned.get("segments", []) or []:
                segments_out.append({
                    "speaker": "SPEAKER_UNKNOWN",
                    "start": float(s.get("start", 0.0)),
                    "end": float(s.get("end", 0.0)),
                    "text": (s.get("text") or "").strip(),
                    "words": s.get("words") or [],
                })
        else:
            # diarization device preference
            if args.diarization_device == "cpu":
                diar_dev = "cpu"
            elif args.diarization_device == "cuda":
                diar_dev = "cuda"
            else:
                diar_dev = "cuda" if torch.cuda.is_available() else "cpu"

            pipeline = create_pyannote_pipeline(hf_token, diar_dev)
            print("Running pyannote diarization ...")
            diarization = diarize_with_retry(pipeline, wav_path)

            # Assign speakers
            if word_level_ok and any((seg.get("words") for seg in aligned.get("segments", []) or [])):
                words = assign_word_speakers(diarization, aligned)
                segments_out = merge_words_into_speaker_segments(words, max_gap=float(args.max_speaker_gap))
            else:
                # segment-level fallback
                segments_out = []
                for s in aligned.get("segments", []) or []:
                    start = float(s.get("start", 0.0))
                    end = float(s.get("end", start))
                    # map by midpoint
                    mid = 0.5 * (start + end)
                    spk = "SPEAKER_UNKNOWN"
                    for turn, _, speaker in diarization.itertracks(yield_label=True):
                        if float(turn.start) <= mid < float(turn.end):
                            spk = str(speaker)
                            break
                    segments_out.append({
                        "speaker": spk,
                        "start": start,
                        "end": end,
                        "text": (s.get("text") or "").strip(),
                        "words": s.get("words") or [],
                    })

        # Save outputs
        write_result_txt(segments_out, "result.txt")
        write_result_json(args.audio_path, language, segments_out, "result.json")

    finally:
        if is_tmp:
            try:
                os.remove(wav_path)
            except Exception:
                pass


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted by user.")
        sys.exit(1)
