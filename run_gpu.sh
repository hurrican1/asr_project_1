#!/usr/bin/env bash
set -euo pipefail

# =========================
# run_gpu.sh
# =========================
# Назначение:
# - стабилизировать CUDA/cuDNN
# - зафиксировать кеши (HF / torch / whisper)
# - запустить run_asr.py с корректным окружением
#
# Аргументы:
#   $1 = путь к аудио
#   $2 = язык (например ru)
#   $3 = output_dir (jobs/<job_id>)
# =========================

if [ "$#" -lt 3 ]; then
  echo "Usage: $0 <audio_path> <language> <output_dir>"
  exit 1
fi

AUDIO_PATH="$1"
LANG="$2"
OUT_DIR="$3"

echo "=== run_gpu.sh ==="
echo "Audio:    $AUDIO_PATH"
echo "Language: $LANG"
echo "Out dir:  $OUT_DIR"

# -------------------------
# 1) Python environment
# -------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [ ! -d ".venv" ]; then
  echo "ERROR: .venv not found. Activate/create venv first."
  exit 2
fi

source .venv/bin/activate

# -------------------------
# 2) CUDA / cuDNN fix
# -------------------------
# Собираем все libnvidia/cuDNN из site-packages,
# чтобы не ловить CUDNN_STATUS_SUBLIBRARY_VERSION_MISMATCH
NVIDIA_LIBS=$(python - << 'EOF'
import site, os
paths = []
for p in site.getsitepackages():
    cand = os.path.join(p, "nvidia")
    if os.path.isdir(cand):
        for root, dirs, files in os.walk(cand):
            if any(f.startswith("libcudnn") or f.startswith("libcuda") for f in files):
                paths.append(root)
print(":".join(sorted(set(paths))))
EOF
)

if [ -n "$NVIDIA_LIBS" ]; then
  export LD_LIBRARY_PATH="$NVIDIA_LIBS:${LD_LIBRARY_PATH:-}"
fi

echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH"

# -------------------------
# 3) Stable cache locations
# -------------------------
# ВАЖНО: persistent volume (/workspace)
export XDG_CACHE_HOME="/workspace/.cache"
export HF_HOME="/workspace/.cache/huggingface"
export TORCH_HOME="/workspace/.cache/torch"
export TRANSFORMERS_CACHE="/workspace/.cache/huggingface"
export WHISPERX_CACHE_DIR="/workspace/.cache/whisperx"

mkdir -p "$XDG_CACHE_HOME" "$HF_HOME" "$TORCH_HOME" "$WHISPERX_CACHE_DIR"

echo "Cache dirs:"
echo "  XDG_CACHE_HOME=$XDG_CACHE_HOME"
echo "  HF_HOME=$HF_HOME"
echo "  TORCH_HOME=$TORCH_HOME"

# -------------------------
# 4) Diagnostics (temporary)
# -------------------------
echo "Python: $(which python)"
python --version

echo "Env summary:"
echo "  WHISPERX_VAD_METHOD=${WHISPERX_VAD_METHOD:-unset}"
echo "  HF_TOKEN=${HF_TOKEN:+set}"
echo "  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-all}"

# -------------------------
# 5) Run ASR
# -------------------------
echo "Starting ASR..."
python run_asr.py "$AUDIO_PATH" "$LANG" "$OUT_DIR"

RC=$?
echo "ASR finished with code $RC"
exit $RC
