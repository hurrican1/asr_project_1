#!/usr/bin/env python3
"""
run_asr.py — WhisperX + alignment + pyannote diarization + speaker embedding export

Основная цель этого файла:
1) Принять аудиофайл (путь) + язык (опционально) + output_dir (опционально)
2) Привести аудио к 16kHz mono WAV (через ffmpeg)
3) Запустить WhisperX ASR + alignment
4) Запустить diarization (pyannote) и назначить спикеров сегментам
5) Попробовать распознать известных спикеров по voice_db.json (косинусная близость embeddings)
6) Остальных обозначить как UNKNOWN_1, UNKNOWN_2, ... и сохранить их embeddings в result.json
7) Сохранить result.json и result.txt в output_dir

Заметка по VAD:
По умолчанию используется pyannote VAD (WHISPERX_VAD_METHOD=pyannote),
чтобы избежать torch.hub загрузки silero-vad из GitHub.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


LOG = logging.getLogger("run_asr")


def env_str(name: str, default: Optional[str] = None) -> Optional[str]:
    v = os.getenv(name)
    if v is None:
        return default
    v = v.strip()
    return v if v else default


def env_int(name: str, default: int) -> int:
    v = env_str(name)
    if v is None:
        return default
    try:
        return int(v)
    except Exception:
        return default


def env_float(name: str, default: float) -> float:
    v = env_str(name)
    if v is None:
        return default
    try:
        return float(v)
    except Exception:
        return default


def format_time(seconds: float) -> str:
    if seconds is None:
        return "00:00"
    s = max(0.0, float(seconds))
    mm = int(s // 60)
    ss = int(s % 60)
    return f"{mm:02d}:{ss:02d}"


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def run_ffmpeg_to_wav16k_mono(src: Path, dst_wav: Path) -> None:
    """Convert any input audio/video to 16kHz mono WAV."""
    ensure_dir(dst_wav.parent)
    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(src),
        "-ac",
        "1",
        "-ar",
        "16000",
        "-vn",
        str(dst_wav),
    ]
    LOG.info("ffmpeg: %s", " ".join(cmd))
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p.returncode != 0:
        tail = (p.stderr or p.stdout or "")[-2000:]
        raise RuntimeError(f"ffmpeg failed (rc={p.returncode}). Tail:\n{tail}")


def load_voice_db(db_path: Path) -> Dict[str, Any]:
    """
    Expected schema (пример):
    {
      "version": 1,
      "model_id": "pyannote/wespeaker-voxceleb-resnet34-LM",
      "speakers": {
        "Иван": {"embeddings": [[...],[...]], "meta": {...}},
        ...
      }
    }
    """
    if not db_path.exists():
        return {"version": 1, "model_id": None, "speakers": {}}
    try:
        return json.loads(db_path.read_text(encoding="utf-8"))
    except Exception as e:
        LOG.warning("Failed to read voice db %s: %s", db_path, e)
        return {"version": 1, "model_id": None, "speakers": {}}


def l2_normalize(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32)
    n = float(np.linalg.norm(v) + 1e-12)
    return v / n


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = l2_normalize(a)
    b = l2_normalize(b)
    return float(np.dot(a, b))


def match_speaker(
    embedding: np.ndarray,
    voice_db: Dict[str, Any],
    threshold: float,
) -> Tuple[Optional[str], float]:
    """Returns: (best_name or None, best_score)"""
    if embedding is None:
        return None, 0.0
    speakers = (voice_db or {}).get("speakers") or {}
    best_name: Optional[str] = None
    best_score: float = -1.0

    for name, info in speakers.items():
        embs = (info or {}).get("embeddings") or []
        for e in embs:
            try:
                score = cosine_similarity(embedding, np.asarray(e, dtype=np.float32))
            except Exception:
                continue
            if score > best_score:
                best_score = score
                best_name = str(name)

    if best_name is not None and best_score >= threshold:
        return best_name, best_score
    return None, best_score if best_score > 0 else 0.0


def diarization_to_spans(diarization: Any) -> List[Tuple[float, float, str]]:
    """
    Convert diarization output to list of (start, end, speaker_label).
    Supports:
      - pandas.DataFrame (start/end/speaker)
      - pyannote.core.Annotation
    """
    spans: List[Tuple[float, float, str]] = []

    # pandas DataFrame-like
    if hasattr(diarization, "iterrows"):
        try:
            for _, row in diarization.iterrows():
                spans.append((float(row["start"]), float(row["end"]), str(row["speaker"])))
            return spans
        except Exception:
            pass

    # pyannote.core.Annotation-like
    if hasattr(diarization, "itertracks"):
        try:
            for segment, _, label in diarization.itertracks(yield_label=True):
                spans.append((float(segment.start), float(segment.end), str(label)))
            return spans
        except Exception:
            pass

    return spans


def pick_segments_for_embedding(
    spans: List[Tuple[float, float, str]],
    speaker: str,
    *,
    min_dur: float = 1.0,
    max_segments: int = 5,
    max_total_dur: float = 30.0,
) -> List[Tuple[float, float]]:
    """
    Choose segments for a speaker to compute a stable embedding:
    - take longest segments first
    - limit count and total duration
    """
    candidates = [(s, e) for (s, e, spk) in spans if spk == speaker and (e - s) >= min_dur]
    candidates.sort(key=lambda t: (t[1] - t[0]), reverse=True)

    chosen: List[Tuple[float, float]] = []
    total = 0.0
    for s, e in candidates:
        if len(chosen) >= max_segments:
            break
        dur = float(e - s)
        if total + dur > max_total_dur and chosen:
            break
        chosen.append((float(s), float(e)))
        total += dur

    # fallback: if nothing long enough, take first few short segments
    if not chosen:
        short = [(s, e) for (s, e, spk) in spans if spk == speaker]
        short.sort(key=lambda t: (t[1] - t[0]), reverse=True)
        for s, e in short[: max_segments]:
            chosen.append((float(s), float(e)))
    return chosen


def compute_speaker_embedding_pyannote(
    audio_wav: Path,
    spans: List[Tuple[float, float, str]],
    speaker_label: str,
    *,
    model_id: str,
    hf_token: str,
    device: str,
    min_dur: float = 1.0,
    max_segments: int = 5,
) -> Optional[np.ndarray]:
    """
    Compute speaker embedding using pyannote Model + Inference.crop.
    Uses averaging over a few segments.
    """
    try:
        import torch
        from pyannote.audio import Model, Inference
        from pyannote.core import Segment
    except Exception as e:
        LOG.warning("pyannote audio embedding imports failed: %s", e)
        return None

    segments = pick_segments_for_embedding(
        spans,
        speaker_label,
        min_dur=min_dur,
        max_segments=max_segments,
    )
    if not segments:
        return None

    try:
        model = Model.from_pretrained(model_id, use_auth_token=hf_token)
        inference = Inference(model, window="whole")
        inference.to(torch.device(device))
    except Exception as e:
        LOG.warning("Failed to init embedding inference (%s): %s", model_id, e)
        return None

    embs: List[np.ndarray] = []
    for s, e in segments:
        try:
            emb = inference.crop(str(audio_wav), Segment(float(s), float(e)))
            emb = np.asarray(emb, dtype=np.float32).reshape(-1)
            if emb.size > 0:
                embs.append(emb)
        except Exception:
            LOG.exception("Embedding crop failed for %s [%.2f..%.2f]", speaker_label, s, e)

    if not embs:
        return None

    avg = np.mean(np.stack(embs, axis=0), axis=0)
    return l2_normalize(avg)


def build_unknown_mapping(
    speaker_labels: List[str],
    known_map: Dict[str, str],
) -> Dict[str, str]:
    """speaker_label -> final_label"""
    mapping: Dict[str, str] = {}
    unknown_i = 1
    for spk in speaker_labels:
        if spk in known_map:
            mapping[spk] = known_map[spk]
        else:
            mapping[spk] = f"UNKNOWN_{unknown_i}"
            unknown_i += 1
    return mapping


def render_segments_to_text(segments: List[Dict[str, Any]]) -> str:
    lines: List[str] = []
    for seg in segments:
        start = format_time(float(seg.get("start") or 0.0))
        end = format_time(float(seg.get("end") or 0.0))
        speaker = str(seg.get("speaker") or "UNKNOWN")
        text = str(seg.get("text") or "").strip()
        if not text:
            continue
        lines.append(f"[{start} - {end}] {speaker}: {text}")
    return "\n".join(lines).strip() + "\n"


def compact_segments(segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Merge adjacent segments with same speaker when close in time."""
    if not segments:
        return []

    merged: List[Dict[str, Any]] = []
    cur = dict(segments[0])
    cur["text"] = str(cur.get("text") or "").strip()

    def flush():
        nonlocal cur
        if cur and str(cur.get("text") or "").strip():
            merged.append(
                {
                    "start": float(cur.get("start") or 0.0),
                    "end": float(cur.get("end") or 0.0),
                    "speaker": cur.get("speaker"),
                    "text": str(cur.get("text") or "").strip(),
                }
            )

    for seg in segments[1:]:
        spk = seg.get("speaker")
        text = str(seg.get("text") or "").strip()
        if not text:
            continue

        gap = float(seg.get("start") or 0.0) - float(cur.get("end") or 0.0)
        if spk == cur.get("speaker") and gap <= 0.8:
            cur["end"] = float(seg.get("end") or cur.get("end") or 0.0)
            cur["text"] = (str(cur.get("text") or "") + " " + text).strip()
        else:
            flush()
            cur = dict(seg)
            cur["text"] = text

    flush()
    return merged


def ensure_segment_speaker_from_words(seg: Dict[str, Any]) -> None:
    """If segment-level speaker missing but word-level exists, infer majority."""
    if seg.get("speaker"):
        return
    words = seg.get("words") or []
    speakers = [w.get("speaker") for w in words if isinstance(w, dict) and w.get("speaker")]
    if not speakers:
        return
    counts: Dict[str, int] = {}
    for s in speakers:
        counts[str(s)] = counts.get(str(s), 0) + 1
    best = max(counts.items(), key=lambda kv: kv[1])[0]
    seg["speaker"] = best


def parse_args(argv: List[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="WhisperX ASR + diarization + speaker embeddings.")
    p.add_argument("audio_path", type=str, help="Path to input audio")
    p.add_argument("language", nargs="?", default=env_str("ASR_LANG", "ru"), help="Language code (e.g., ru)")
    p.add_argument(
        "output_dir",
        nargs="?",
        default=env_str("ASR_OUTPUT_DIR", "."),
        help="Directory for result.txt/result.json",
    )

    p.add_argument("--model", default=env_str("WHISPERX_MODEL", "large-v2"))
    p.add_argument("--device", default=env_str("DEVICE", None))
    p.add_argument("--compute-type", default=env_str("WHISPERX_COMPUTE_TYPE", None))
    p.add_argument("--batch-size", type=int, default=env_int("WHISPERX_BATCH_SIZE", 16))

    p.add_argument("--vad-method", default=env_str("WHISPERX_VAD_METHOD", "pyannote"))

    p.add_argument("--min-speakers", type=int, default=env_int("DIARIZE_MIN_SPEAKERS", 0))
    p.add_argument("--max-speakers", type=int, default=env_int("DIARIZE_MAX_SPEAKERS", 0))

    p.add_argument("--voice-db", default=env_str("VOICE_DB_PATH", "voice_db.json"))
    p.add_argument("--sim-threshold", type=float, default=env_float("SPEAKER_SIM_THRESHOLD", 0.76))

    p.add_argument("--embedding-model", default=env_str("VOICE_EMBEDDING_MODEL", None))
    p.add_argument("--embedding-min-dur", type=float, default=env_float("VOICE_EMB_MIN_DUR", 1.0))
    p.add_argument("--embedding-max-seg", type=int, default=env_int("VOICE_EMB_MAX_SEG", 5))

    p.add_argument("--no-voice-matching", action="store_true", help="Do not match speakers from voice_db")
    p.add_argument("--no-embeddings", action="store_true", help="Do not compute embeddings (unknown_speakers empty)")

    p.add_argument("--log-level", default=env_str("LOG_LEVEL", "INFO"))

    return p.parse_args(argv)


def main(argv: List[str]) -> int:
    args = parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    audio_path = Path(args.audio_path).expanduser().resolve()
    out_dir = Path(args.output_dir).expanduser().resolve()
    ensure_dir(out_dir)

    if not audio_path.exists():
        LOG.error("Audio file not found: %s", audio_path)
        return 2

    # device
    device = args.device
    if not device:
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            device = "cpu"

    # compute_type
    compute_type = args.compute_type
    if not compute_type:
        compute_type = "float16" if device.startswith("cuda") else "int8"

    model_name = str(args.model)
    language = str(args.language) if args.language else None
    vad_method = str(args.vad_method).strip().lower()

    hf_token = env_str("HF_TOKEN", "")
    if not hf_token:
        LOG.error("HF_TOKEN is missing. Set HF_TOKEN in environment/.env for pyannote models.")
        return 3

    voice_db_path = Path(args.voice_db).expanduser()
    if not voice_db_path.is_absolute():
        voice_db_path = (Path.cwd() / voice_db_path).resolve()

    voice_db = load_voice_db(voice_db_path)
    db_model_id = (voice_db or {}).get("model_id")
    emb_model_id = args.embedding_model or db_model_id or "pyannote/wespeaker-voxceleb-resnet34-LM"

    if db_model_id and args.embedding_model and str(args.embedding_model) != str(db_model_id):
        LOG.warning(
            "VOICE_EMBEDDING_MODEL (%s) differs from voice_db.model_id (%s). Matching may degrade.",
            args.embedding_model,
            db_model_id,
        )

    LOG.info("Input: %s", audio_path)
    LOG.info("Output dir: %s", out_dir)
    LOG.info("Device: %s | compute_type: %s", device, compute_type)
    LOG.info("WhisperX model: %s | lang: %s | vad_method: %s", model_name, language, vad_method)
    LOG.info("Diarization: pyannote/speaker-diarization-3.1 (via WhisperX)")
    LOG.info(
        "Voice DB: %s | speakers: %s | sim_threshold: %.3f",
        voice_db_path,
        len((voice_db.get("speakers") or {})),
        args.sim_threshold,
    )
    LOG.info("Embedding model: %s", emb_model_id)

    # 1) Convert to wav 16k mono
    wav_path = out_dir / "audio_16k.wav"
    try:
        run_ffmpeg_to_wav16k_mono(audio_path, wav_path)
    except Exception:
        LOG.exception("ffmpeg conversion failed")
        return 10

    # 2) WhisperX transcription + alignment + diarization
    try:
        import whisperx
    except Exception as e:
        LOG.error("Missing dependencies: %s", e)
        return 11

    try:
        audio = whisperx.load_audio(str(wav_path))
    except Exception:
        LOG.exception("Failed to load audio with whisperx")
        return 12

    # Load ASR model (with pyannote VAD to avoid silero torch.hub)
    try:
        load_kwargs = dict(
            device=device,
            compute_type=compute_type,
            language=language,
            vad_method=vad_method,
        )
        # Для pyannote VAD нужен HF токен (модель сегментации часто gated).
        if vad_method == "pyannote":
            load_kwargs["vad_options"] = {"use_auth_token": hf_token}

        model = whisperx.load_model(
            model_name,
            **load_kwargs,
        )
    except TypeError:
        LOG.error("Your whisperx.load_model does not support vad_method/vad_options. Please upgrade whisperx to 3.7.4+.")
        return 13
    except Exception:
        LOG.exception("Failed to load WhisperX model")
        return 14

    try:
        result = model.transcribe(audio, batch_size=int(args.batch_size), language=language)
    except Exception:
        LOG.exception("ASR transcribe failed")
        return 15

    # Alignment
    try:
        align_model, metadata = whisperx.load_align_model(language_code=language or result.get("language"), device=device)
        result = whisperx.align(
            result.get("segments", []),
            align_model,
            metadata,
            audio,
            device,
            return_char_alignments=False,
        )
    except Exception:
        LOG.exception("Alignment failed (continuing without alignment)")

    # Diarization
    diarization = None
    diarize_min = int(args.min_speakers) if args.min_speakers else 0
    diarize_max = int(args.max_speakers) if args.max_speakers else 0

    try:
        try:
            diarize_pipeline = whisperx.DiarizationPipeline(use_auth_token=hf_token, device=device)
        except TypeError:
            diarize_pipeline = whisperx.DiarizationPipeline(token=hf_token, device=device)

        diar_kwargs = {}
        if diarize_min > 0:
            diar_kwargs["min_speakers"] = diarize_min
        if diarize_max > 0:
            diar_kwargs["max_speakers"] = diarize_max

        diarization = diarize_pipeline(str(wav_path), **diar_kwargs)
        result = whisperx.assign_word_speakers(diarization, result)
    except Exception:
        LOG.exception("Diarization failed (continuing with UNKNOWN speakers)")
        diarization = None

    # Ensure segment-level speaker exists
    segments: List[Dict[str, Any]] = []
    for seg in (result or {}).get("segments", []) or []:
        if not isinstance(seg, dict):
            continue
        ensure_segment_speaker_from_words(seg)
        segments.append(seg)

    # 3) Compute embeddings per diarization speaker + match to voice_db
    spans = diarization_to_spans(diarization) if diarization is not None else []

    # collect speaker labels from segments
    speaker_labels = []
    for seg in segments:
        spk = seg.get("speaker")
        if spk and str(spk) not in speaker_labels:
            speaker_labels.append(str(spk))

    # Deterministic order
    speaker_labels = sorted(speaker_labels)

    known_map: Dict[str, str] = {}
    speaker_scores: Dict[str, float] = {}
    unknown_speakers: List[Dict[str, Any]] = []
    embeddings_by_label: Dict[str, Optional[np.ndarray]] = {}

    if not args.no_embeddings and spans and speaker_labels:
        for spk in speaker_labels:
            emb = compute_speaker_embedding_pyannote(
                wav_path,
                spans,
                spk,
                model_id=str(emb_model_id),
                hf_token=hf_token,
                device=device,
                min_dur=float(args.embedding_min_dur),
                max_segments=int(args.embedding_max_seg),
            )
            embeddings_by_label[spk] = emb

            if emb is not None and (not args.no_voice_matching):
                name, score = match_speaker(emb, voice_db, threshold=float(args.sim_threshold))
                if name:
                    known_map[spk] = name
                    speaker_scores[spk] = score
    else:
        for spk in speaker_labels:
            embeddings_by_label[spk] = None

    final_map = build_unknown_mapping(speaker_labels, known_map)

    # Prepare unknown_speakers list (for later manual labeling in bot)
    for spk in speaker_labels:
        final = final_map.get(spk, spk)
        if final.startswith("UNKNOWN_"):
            emb = embeddings_by_label.get(spk)
            unknown_speakers.append(
                {
                    "id": final,
                    "source_speaker": spk,
                    "embedding": emb.tolist() if isinstance(emb, np.ndarray) else None,
                }
            )

    # 4) Apply mapping to segments & compact transcript
    mapped_segments: List[Dict[str, Any]] = []
    for seg in segments:
        spk = str(seg.get("speaker") or "")
        seg2 = {
            "start": float(seg.get("start") or 0.0),
            "end": float(seg.get("end") or 0.0),
            "speaker": final_map.get(spk, spk or "UNKNOWN"),
            "text": str(seg.get("text") or ""),
        }
        mapped_segments.append(seg2)

    mapped_segments = sorted(mapped_segments, key=lambda s: float(s.get("start") or 0.0))
    compact = compact_segments(mapped_segments)

    # 5) Save outputs
    result_txt_path = out_dir / "result.txt"
    result_json_path = out_dir / "result.json"

    transcript_text = render_segments_to_text(compact)
    result_txt_path.write_text(transcript_text, encoding="utf-8")

    payload: Dict[str, Any] = {
        "version": 2,
        "input_audio": str(audio_path),
        "audio_16k_wav": str(wav_path),
        "language": language or result.get("language"),
        "whisperx_model": model_name,
        "device": device,
        "compute_type": compute_type,
        "vad_method": vad_method,
        "diarization": {
            "pipeline": "pyannote/speaker-diarization-3.1",
            "min_speakers": diarize_min if diarize_min > 0 else None,
            "max_speakers": diarize_max if diarize_max > 0 else None,
        },
        "voice_db": {
            "path": str(voice_db_path),
            "model_id": db_model_id,
            "sim_threshold": float(args.sim_threshold),
        },
        "embedding_model": str(emb_model_id) if emb_model_id else None,
        "speaker_map": final_map,
        "speaker_scores": speaker_scores,
        "unknown_speakers": unknown_speakers,
        "segments": compact,
    }

    result_json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    LOG.info("Saved: %s", result_txt_path)
    LOG.info("Saved: %s", result_json_path)
    LOG.info("Unknown speakers: %s", len([u for u in unknown_speakers if u.get("id")]))

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main(sys.argv[1:]))
    except KeyboardInterrupt:
        raise SystemExit(130)
