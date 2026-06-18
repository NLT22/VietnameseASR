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

if [[ "$python_minor" == "3.6" ]]; then
  python3 -m pip install "cmake<3.28"
  export SHERPA_ONNX_MAKE_ARGS="${SHERPA_ONNX_MAKE_ARGS:--j1}"
  python3 -m pip install "numpy<1.20" "soundfile<0.13" sherpa-onnx
else
  python3 -m pip install numpy soundfile sherpa-onnx
fi

echo
echo "Install complete. Activate with:"
echo "  source .venv/bin/activate"
echo
echo "Smoke test command:"
echo "  python3 transcribe_wav.py /path/to/16k_mono.wav"
