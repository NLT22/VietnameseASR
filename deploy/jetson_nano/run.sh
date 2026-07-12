#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# run.sh — run the VietnameseASR medium model ON THIS MACHINE (a Jetson Nano).
#
# One-time, from a fresh Jetson:
#   git clone git@github.com:NLT22/VietnameseASR.git && cd VietnameseASR/deploy/jetson_nano
#   bash run.sh setup                       # installs onnxruntime/numpy (~5-10 min)
#
# Then:
#   bash run.sh transcribe /path/to/audio.wav
#
# No PC, no SSH, no scp — everything the model needs is already in the repo
# (deploy/jetson_nano/model_medium_epoch30_avg10/, tracked in git).
#
# Browser mic UI (streaming, VAD), simplest path — one command, prints a URL
# to open from any laptop/phone (works through a tunnel, so LAN-only mic
# permission issues in the browser don't matter):
#   bash run.sh startui
#
# live-ui does the same but binds LAN only, no tunnel (http://<jetson-ip>:8100).
# =============================================================================

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$script_dir"

model_dir="${MODEL_DIR:-model_medium_epoch30_avg10}"
export OPENBLAS_CORETYPE=ARMV8   # numpy on the Nano segfaults ("Illegal instruction") without this
# global (not `local`): the EXIT trap in do_startui runs after the function
# that declares these returns, so they must still be in scope when it fires.
server_pid=""
tunnel_pid=""

usage() {
  awk '/^# ={5,}/{c++} c==1{print} c==2{exit}' "$0"
}

activate_venv() {
  if [ -x .venv/bin/python3 ]; then
    # shellcheck disable=SC1091
    . .venv/bin/activate
  fi
}

do_setup() {
  echo "==> Installing dependencies (first run only, ~5-10 min on a Nano)"
  bash install_jetson.sh
  echo "==> Done. Run: bash run.sh transcribe audio.wav"
}

do_transcribe() {
  local wav="${1:-}"
  [ -z "$wav" ] && { echo "ERROR: transcribe needs a wav path" >&2; exit 1; }
  [ -d ".venv" ] || { echo "ERROR: run 'bash run.sh setup' first" >&2; exit 1; }
  activate_venv
  python3 transcribe_beam_wav.py --model-dir "$model_dir" "$wav"
}

ensure_live_ui_deps() {
  [ -d ".venv" ] || { echo "ERROR: run 'bash run.sh setup' first" >&2; exit 1; }
  activate_venv
  python3 -m pip show fastapi >/dev/null 2>&1 || python3 -m pip install fastapi uvicorn websockets
}

ensure_cloudflared() {
  if ! command -v cloudflared >/dev/null 2>&1; then
    echo "==> Installing cloudflared (arm64, one-time)"
    mkdir -p "$HOME/.local/bin"
    curl -fL -o "$HOME/.local/bin/cloudflared" \
      https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-arm64
    chmod +x "$HOME/.local/bin/cloudflared"
    export PATH="$HOME/.local/bin:$PATH"
  fi
}

check_port_free() {
  if curl -sf "http://127.0.0.1:8100/health" >/dev/null 2>&1; then
    echo "ERROR: something is already listening on :8100 (another run.sh instance?)." >&2
    echo "       Stop it first, e.g.: pkill -f 'server.py.*port 8100'" >&2
    exit 1
  fi
}

launch_server() {
  local live_ui_dir="$script_dir/../../live_ui"
  local live_model="$script_dir/$model_dir"
  local spk_model="$live_ui_dir/models/speaker_embedding_campplus.onnx"
  local speaker_flag=""
  if [ ! -f "$spk_model" ]; then
    echo "==> Speaker-ID model missing (27M, optional) — download it with:" >&2
    echo "    curl -L -o '$spk_model' \\" >&2
    echo "      https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced.onnx" >&2
    echo "    Continuing without speaker-ID (--no-speaker-id)." >&2
    speaker_flag="--no-speaker-id"
  fi
  cd "$live_ui_dir"
  python3 server.py --model-dir "$live_model" --host 0.0.0.0 --port 8100 \
    $speaker_flag &
  server_pid=$!
}

do_live_ui() {
  ensure_live_ui_deps
  check_port_free
  echo "==> Starting live UI on this Jetson."
  echo "    From a laptop on the same LAN: http://<jetson-lan-ip>:8100"
  launch_server
  wait "$server_pid"
}

do_startui() {
  ensure_live_ui_deps
  check_port_free
  ensure_cloudflared
  launch_server
  if ! kill -0 "$server_pid" 2>/dev/null; then
    echo "ERROR: server failed to start — see output above." >&2
    exit 1
  fi
  trap 'kill "$server_pid" "$tunnel_pid" 2>/dev/null' EXIT INT TERM

  echo "==> Waiting for the server to come up..."
  for _ in $(seq 1 30); do
    curl -sf "http://127.0.0.1:8100/health" >/dev/null 2>&1 && break
    kill -0 "$server_pid" 2>/dev/null || { echo "ERROR: server exited early." >&2; exit 1; }
    sleep 1
  done

  echo "==> Starting cloudflared quick tunnel..."
  local log; log="$(mktemp)"
  cloudflared tunnel --url http://localhost:8100 >"$log" 2>&1 &
  tunnel_pid=$!

  local url=""
  for _ in $(seq 1 30); do
    url="$(grep -oE 'https://[a-zA-Z0-9.-]+trycloudflare\.com' "$log" | head -1 || true)"
    [ -n "$url" ] && break
    sleep 1
  done

  echo
  if [ -n "$url" ]; then
    echo "==> Open this from your laptop / phone:"
    echo "    $url"
  else
    echo "WARNING: tunnel URL not found yet, check log: $log"
  fi
  echo "==> Ctrl+C to stop the server and tunnel."
  echo
  wait "$server_pid"
}

[ $# -eq 0 ] && { usage; exit 1; }
cmd="$1"; shift || true

case "$cmd" in
  -h|--help) usage ;;
  setup) do_setup ;;
  transcribe) do_transcribe "${1:-}" ;;
  live-ui) do_live_ui ;;
  startui) do_startui ;;
  *)
    echo "Unknown command: $cmd" >&2
    usage
    exit 1
    ;;
esac
