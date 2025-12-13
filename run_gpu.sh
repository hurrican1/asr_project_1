{\rtf1\ansi\ansicpg1251\cocoartf2822
\cocoatextscaling0\cocoaplatform0{\fonttbl\f0\fswiss\fcharset0 Helvetica;}
{\colortbl;\red255\green255\blue255;}
{\*\expandedcolortbl;;}
\paperw11900\paperh16840\margl1440\margr1440\vieww11520\viewh8400\viewkind0
\pard\tx720\tx1440\tx2160\tx2880\tx3600\tx4320\tx5040\tx5760\tx6480\tx7200\tx7920\tx8640\pardirnatural\partightenfactor0

\f0\fs24 \cf0 #!/usr/bin/env bash\
set -euo pipefail\
\
cd "$(dirname "$0")"\
\
# Activate venv\
source .venv/bin/activate\
\
# Load HF_TOKEN from .env if present\
if [ -f .env ]; then\
  set -a\
  source .env\
  set +a\
fi\
\
# Build LD_LIBRARY_PATH from nvidia libs inside venv site-packages\
NVIDIA_LIBS="$(python - <<'PY'\
import site, glob, os\
paths=[]\
for base in site.getsitepackages():\
    n=os.path.join(base,'nvidia')\
    if os.path.isdir(n):\
        for lib in glob.glob(os.path.join(n,'*','lib')):\
            paths.append(lib)\
print(':'.join(paths))\
PY\
)"\
\
export LD_LIBRARY_PATH="$NVIDIA_LIBS"\
\
AUDIO_PATH="$\{1:-audio/rave.m4a\}"\
LANG="$\{2:-ru\}"\
\
python run_asr.py --audio_path "$AUDIO_PATH" --language "$LANG"\
}