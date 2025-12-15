#!/usr/bin/env bash
set -euo pipefail

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$BASE_DIR"

# Activate venv if exists
if [[ -f "$BASE_DIR/.venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "$BASE_DIR/.venv/bin/activate"
fi

AUDIO_PATH="${1:-}"
LANG="${2:-ru}"
OUT_DIR="${3:-.}"

if [[ -z "$AUDIO_PATH" ]]; then
  echo "Usage: $0 <audio_path> [lang] [out_dir]"
  exit 2
fi

# Build LD_LIBRARY_PATH from NVIDIA libs that come with PyTorch wheels
NVIDIA_LIBS=$(python - <<'PY'
import site, glob, os
paths=[]
for base in site.getsitepackages():
    n=os.path.join(base, "nvidia")
    if os.path.isdir(n):
        for lib in glob.glob(os.path.join(n, "*", "lib")):
            paths.append(lib)
print(":".join(paths))
PY
)

if [[ -n "$NVIDIA_LIBS" ]]; then
  export LD_LIBRARY_PATH="$NVIDIA_LIBS"
fi

python run_asr.py --audio_path "$AUDIO_PATH" --language "$LANG" --out_dir "$OUT_DIR"
