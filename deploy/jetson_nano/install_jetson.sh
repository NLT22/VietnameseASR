#!/usr/bin/env bash
set -euo pipefail

run_sudo() {
  if sudo -n true 2>/dev/null; then
    sudo "$@"
  elif [[ -n "${SUDO_PASSWORD:-}" ]]; then
    printf '%s\n' "$SUDO_PASSWORD" | sudo -S -p "" "$@"
  else
    sudo "$@"
  fi
}

run_sudo apt-get update
run_sudo apt-get install -y \
  build-essential \
  curl \
  ffmpeg \
  libsndfile1 \
  python3-pip \
  python3-venv

python_bin="python3"
if apt-cache policy python3.8 2>/dev/null | awk '/Candidate:/ { found = ($2 != "(none)") } END { exit !found }'; then
  run_sudo apt-get install -y python3.8 python3.8-venv
  python_bin="python3.8"
fi

rm -rf .venv
"$python_bin" -m venv .venv
source .venv/bin/activate
python3 --version
python3 -m pip install --upgrade pip setuptools wheel
export PATH="$PWD/.venv/bin:$PATH"
python_minor="$(python3 - <<'PY'
import sys
print(f"{sys.version_info.major}.{sys.version_info.minor}")
PY
)"

# Our decoder (jetson_beam_decode.py / jetson_asr.py) needs plain onnxruntime
# + kaldi_native_fbank directly -- NOT sherpa-onnx, which bundles its own
# onnxruntime internally (no `import onnxruntime`) and gives worse WER (see
# deploy/jetson_nano/README.md section 2). Installing it wastes a slow
# from-source compile on py3.6 for nothing.
if [[ "$python_minor" == "3.6" ]]; then
  # JetPack 4 / L4T R32.7.6 pins, proven to work (no newer aarch64+py3.6
  # wheels exist): onnxruntime 1.10.0, numpy 1.19.5. kaldi_native_fbank has
  # no aarch64 py3.6 wheel either, so it builds from source -- needs a
  # modern-enough cmake (the system one on the Nano is often too old).
  python3 -m pip install "cmake<3.28"
  python3 -m pip install "numpy==1.19.5" "soundfile<0.13" "onnxruntime==1.10.0" kaldi_native_fbank
else
  python3 -m pip install numpy soundfile onnxruntime kaldi_native_fbank
fi

# Live UI deps (fastapi/uvicorn/websockets) here too, so `setup` fully
# prepares the machine in one pass — `startui`/`live-ui` no longer install
# anything on first use.
python3 -m pip install fastapi uvicorn websockets

echo
echo "Install complete. Activate with:"
echo "  source .venv/bin/activate"
echo
echo "Smoke test command:"
echo "  python3 transcribe_beam_wav.py --model-dir model_medium_epoch30_avg10 /path/to/16k_mono.wav"
