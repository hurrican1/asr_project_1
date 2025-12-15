#!/usr/bin/env python3
# === Required by user: PyTorch 2.6+ safe unpickling fix (must be at top) ===
import torch

torch.serialization.add_safe_globals(
    [
        __import__("omegaconf").OmegaConf,
        __import__("omegaconf.listconfig").listconfig.ListConfig,
    ]
)

import os
import sys
import json
import argparse
import shutil
import subprocess
import tempfile
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict

import numpy as np

# --- RunPod / HF download robustness ---
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

VOICE_EMBED_MODEL_ID = "pyannote/wespeaker-voxceleb-resnet34-LM"


# --------------------------- utils ---------------------------
def print_environment() -> None:
    print("=== Environment info ===")
    print(f"Python version : {sys.version.split()[0]}")
    print(f"Torch version : {torch.__version__}")
    cuda_ver = getattr(torch.version, "cuda", None)
    print(f"Torch CUDA : {cuda_ver if cuda_ver else 'not available'}")
    cudnn_ver = None
    try:
        cudnn_ver = torch.backends.cudnn.version()
    except Exception:
        cudnn_ver = None
    print(f"cuDNN version : {cudnn_ver if cudnn_ver else 'unknown'}")

    if torch.cuda.is_available():
        n = torch.cuda.device_count()
        print(f"CUDA available : YES ({n} GPU(s) detected)")
        for i in range(n):
            print(f" GPU {i}: {torch.cuda.get_device_name(i)}")
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

    print(f"Using device : {dev}")
    return dev


def require_ffmpeg() -> None:
    if shutil.which("ffmpeg") is None:
        print("ERROR: ffmpeg not found in PATH.")
        print("Install it:")
        print(" Ubuntu/Debian: sudo apt update && sudo apt install -y ffmpeg")
        print(" macOS: brew install ffmpeg")
        sys.exit(1)


def convert_to_wav_16k_mono(input_path: str) -> Tuple[str, bool]:
    """
    Convert any input audio into a temporary 16kHz mono WAV file.
    This makes pyannote pipelines stable (they expect 16k mono) and avoids codec issues.
    Returns: (wav_path, is_temporary)
    """
    require_ffmpeg()
    tmp_wav = os.path.join(tempfile.gettempdir(), f"asr_tmp_{uuid.uuid4().hex}.wav")
    cmd = [
        "ffmpeg",
        "-y",
        "-nostdin",
        "-i",
        input_path,
        "-ac",
        "1",
        "-ar",
        "16000",
        "-vn",
        tmp_wav,
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
        return


def get_hf_token(skip_diarization: bool) -> Optional[str]:
    load_dotenv_if_exists(".env")
    token = os.getenv("HF_TOKEN")
    if token:
        print("HF_TOKEN : detected (not printed for security)")
        return token

    msg = (
        "HF_TOKEN environment variable is not set.\n"
        "It is required to download pyannote speaker-diarization models from Hugging Face.\n"
        "Create a token at https://huggingface.co/settings/tokens with 'read' access,\n"
        "accept terms for pyannote models, then set it:\n"
        " export HF_TOKEN=hf_xxx\n"
        "Or create a local .env file (NOT committed) with:\n"
        " HF_TOKEN=hf_xxx\n"
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
    If GPU move fails, it will force CPU to avoid mixed-device crashes.
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
    pipeline.to(torch.device("cpu"))

    # Try GPU if requested and available
    if requested_device.startswith("cuda") and torch.cuda.is_available():
        try:
            pipeline.to(torch.device(requested_device))
            print(f"pyannote device : {requested_device}")
        except Exception as e:
            print(f"WARNING: could not move pyannote pipeline to '{requested_device}'. Falling back to CPU.")
            print(f" Reason: {e}")
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
            words.append(
                {"word": word_text, "start": start, "end": end, "speaker": speaker_at(mid)}
            )

    words.sort(key=lambda x: x["start"])
    return words


def merge_words_into_speaker_segments(
    words: List[Dict[str, Any]], max_gap: float = 1.0
) -> List[Dict[str, Any]]:
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
            cur = {"speaker": w["speaker"], "start": w["start"], "end": w["end"], "words": [w]}
        else:
            cur["words"].append(w)
            cur["end"] = w["end"]

    cur["text"] = " ".join(x["word"] for x in cur["words"]).strip()
    out.append(cur)
    return out


# --------------------------- speaker identification ---------------------------
def create_embedding_inference(hf_token: Optional[str], device: str):
    """
    Uses pyannote/wespeaker-voxceleb-resnet34-LM (compatible with pyannote.audio 3.x).
    """
    from pyannote.audio import Model, Inference

    # Prefer token if available, but model is usually public
    model = None
    if hf_token:
        try:
            model = Model.from_pretrained(VOICE_EMBED_MODEL_ID, use_auth_token=hf_token)
        except TypeError:
            model = Model.from_pretrained(VOICE_EMBED_MODEL_ID, token=hf_token)
        except Exception:
            model = None
    if model is None:
        model = Model.from_pretrained(VOICE_EMBED_MODEL_ID)

    inference = Inference(model, window="whole")

    if device.startswith("cuda") and torch.cuda.is_available():
        inference.to(torch.device(device))
    else:
        inference.to(torch.device("cpu"))

    return inference


def compute_cluster_embedding(
    inference,
    wav_path: str,
    segments: List[Tuple[float, float]],
    *,
    min_segment_sec: float = 1.0,
    max_crop_sec: float = 3.0,
    max_total_sec: float = 30.0,
) -> Optional[np.ndarray]:
    """
    Вычисляет эмбеддинг "спикера" (кластера) как среднее из эмбеддингов по нескольким
    самым длинным фрагментам речи.
    """
    from pyannote.core import Segment

    if not segments:
        return None

    # Берём сначала самые длинные куски
    segs = sorted(segments, key=lambda x: (x[1] - x[0]), reverse=True)

    embs: List[np.ndarray] = []
    total = 0.0

    for start, end in segs:
        dur = end - start
        if dur < min_segment_sec:
            continue

        crop_end = min(end, start + max_crop_sec)
        if crop_end <= start:
            continue

        emb = inference.crop(wav_path, Segment(start, crop_end))
        vec = np.asarray(emb, dtype=np.float32).reshape(-1)
        embs.append(vec)

        total += (crop_end - start)
        if total >= max_total_sec:
            break

    if not embs:
        return None

    return np.mean(np.stack(embs, axis=0), axis=0)


def identify_speakers(
    diarization,
    wav_path: str,
    inference,
    voice_db: Dict[str, Any],
    threshold: float,
) -> Tuple[Dict[str, str], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Возвращает:
    - speaker_map: diar_label -> name_or_UNKNOWN_k
    - unknowns: [{id,label,embedding,best_score}, ...]
    - knowns:   [{label,name,score}, ...]
    """
    from voice_db import match_speaker

    spk_segments: Dict[str, List[Tuple[float, float]]] = defaultdict(list)
    for turn, _, spk in diarization.itertracks(yield_label=True):
        spk_segments[str(spk)].append((float(turn.start), float(turn.end)))

    speaker_map: Dict[str, str] = {}
    unknowns: List[Dict[str, Any]] = []
    knowns: List[Dict[str, Any]] = []

    unknown_idx = 1

    for label in sorted(spk_segments.keys()):
        emb = compute_cluster_embedding(inference, wav_path, spk_segments[label])
        if emb is None:
            unk_id = f"UNKNOWN_{unknown_idx}"
            unknown_idx += 1
            speaker_map[label] = unk_id
            unknowns.append(
                {"id": unk_id, "label": label, "embedding": None, "best_score": None}
            )
            continue

        emb_list = emb.astype(np.float32).tolist()
        name, score = match_speaker(voice_db, emb_list, threshold)

        if name is not None:
            speaker_map[label] = name
            knowns.append({"label": label, "name": name, "score": score})
        else:
            unk_id = f"UNKNOWN_{unknown_idx}"
            unknown_idx += 1
            speaker_map[label] = unk_id
            unknowns.append(
                {"id": unk_id, "label": label, "embedding": emb_list, "best_score": score}
            )

    return speaker_map, unknowns, knowns


def apply_speaker_map(segments: List[Dict[str, Any]], speaker_map: Dict[str, str]) -> None:
    """
    Модифицирует segments in-place:
    - сохраняет исходную метку в speaker_label
    - заменяет speaker на имя/UNKNOWN_k
    """
    for s in segments:
        orig = s.get("speaker")
        s["speaker_label"] = orig
        if orig in speaker_map:
            s["speaker"] = speaker_map[orig]


# --------------------------- output ---------------------------
def write_result_txt(segments: List[Dict[str, Any]], path: Path) -> None:
    lines = []
    for s in segments:
        lines.append(
            f"[{format_ts(s['start'])} --> {format_ts(s['end'])}] {s['speaker']}: {s.get('text','')}"
        )
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {path}")


def write_result_json(
    audio_path: str,
    language: Optional[str],
    segments: List[Dict[str, Any]],
    path: Path,
    *,
    speaker_map: Optional[Dict[str, str]] = None,
    unknown_speakers: Optional[List[Dict[str, Any]]] = None,
    known_speakers: Optional[List[Dict[str, Any]]] = None,
    speaker_threshold: Optional[float] = None,
    voice_embed_model: Optional[str] = None,
) -> None:
    payload: Dict[str, Any] = {
        "audio_path": os.path.abspath(audio_path),
        "language": language,
        "segments": segments,
    }
    if speaker_map is not None:
        payload["speaker_map"] = speaker_map
    if unknown_speakers is not None:
        payload["unknown_speakers"] = unknown_speakers
    if known_speakers is not None:
        payload["known_speakers"] = known_speakers
    if speaker_threshold is not None:
        payload["speaker_threshold"] = speaker_threshold
    if voice_embed_model is not None:
        payload["voice_embed_model"] = voice_embed_model

    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {path}")


# --------------------------- cli ---------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="WhisperX large-v2 + pyannote diarization 3.1 (+ speaker id)")
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

    # NEW:
    p.add_argument("--out_dir", type=str, default=".", help="Output directory (per-job folder recommended).")
    p.add_argument("--voice_db_path", type=str, default="voice_db.json", help="Path to voice DB (JSON).")
    p.add_argument("--speaker_threshold", type=float, default=0.75, help="Cosine similarity threshold for speaker ID.")
    p.add_argument("--disable_speaker_id", action="store_true", help="Disable mapping to known speaker names.")

    return p.parse_args()


# --------------------------- main ---------------------------
def main() -> None:
    args = parse_args()
    print_environment()

    if not os.path.isfile(args.audio_path):
        print(f"ERROR: audio file not found: {args.audio_path}")
        sys.exit(1)

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_txt = out_dir / "result.txt"
    out_json = out_dir / "result.json"

    asr_device = choose_device(args.device)
    compute_type = args.compute_type or ("float16" if asr_device.startswith("cuda") else "int8")
    print(f"Compute type : {compute_type}")

    hf_token = get_hf_token(skip_diarization=args.skip_diarization)

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
            vad_method=args.asr_vad,
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
            print(f" Reason: {e}")
            aligned = result
            word_level_ok = False

        # Build output segments
        speaker_map = None
        unknown_speakers = None
        known_speakers = None

        if args.skip_diarization:
            print("Skipping diarization. Speakers will be SPEAKER_UNKNOWN.")
            segments_out: List[Dict[str, Any]] = []
            for s in aligned.get("segments", []) or []:
                segments_out.append(
                    {
                        "speaker": "SPEAKER_UNKNOWN",
                        "start": float(s.get("start", 0.0)),
                        "end": float(s.get("end", 0.0)),
                        "text": (s.get("text") or "").strip(),
                        "words": s.get("words") or [],
                    }
                )

        else:
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
                    mid = 0.5 * (start + end)

                    spk = "SPEAKER_UNKNOWN"
                    for turn, _, speaker in diarization.itertracks(yield_label=True):
                        if float(turn.start) <= mid < float(turn.end):
                            spk = str(speaker)
                            break

                    segments_out.append(
                        {
                            "speaker": spk,
                            "start": start,
                            "end": end,
                            "text": (s.get("text") or "").strip(),
                            "words": s.get("words") or [],
                        }
                    )

            # NEW: speaker identification
            if not args.disable_speaker_id:
                from voice_db import load_db

                voice_db = load_db(Path(args.voice_db_path))
                inference = create_embedding_inference(hf_token, diar_dev)
                speaker_map, unknown_speakers, known_speakers = identify_speakers(
                    diarization=diarization,
                    wav_path=wav_path,
                    inference=inference,
                    voice_db=voice_db,
                    threshold=float(args.speaker_threshold),
                )
                apply_speaker_map(segments_out, speaker_map)

        # Save outputs
        write_result_txt(segments_out, out_txt)
        write_result_json(
            args.audio_path,
            language,
            segments_out,
            out_json,
            speaker_map=speaker_map,
            unknown_speakers=unknown_speakers,
            known_speakers=known_speakers,
            speaker_threshold=None if args.disable_speaker_id else float(args.speaker_threshold),
            voice_embed_model=None if args.disable_speaker_id else VOICE_EMBED_MODEL_ID,
        )

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
