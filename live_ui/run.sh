#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# run.sh — run the live streaming mic UI ON THIS MACHINE. Fully self-contained:
# clone the VietnameseASR repo and this is all you need, no other repo/venv.
#
#   bash run.sh startui       # local venv (first run) + server + cloudflared
#                              # tunnel, prints a URL you can open anywhere
#   bash run.sh serve         # server only, http://localhost:8100, no tunnel
#
# For the Jetson deployment (the real target), use deploy/jetson_nano/run.sh
# instead — this script is the desk/dev-testing equivalent.
# =============================================================================

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$script_dir"

model_dir="${MODEL_DIR:-$script_dir/../deploy/jetson_nano/model_medium_epoch30_avg10}"
venv_dir="$script_dir/.venv"
# global (not `local`): the EXIT trap in do_startui runs after the function
# that declares these returns, so they must still be in scope when it fires.
server_pid=""
tunnel_pid=""

usage() {
  awk '/^# ={5,}/{c++} c==1{print} c==2{exit}' "$0"
}

ensure_deps() {
  if [ ! -x "$venv_dir/bin/python3" ]; then
    echo "==> Creating local venv (first run only)"
    python3 -m venv "$venv_dir"
  fi
  "$venv_dir/bin/python3" -c "import fastapi, uvicorn, onnxruntime, kaldi_native_fbank" >/dev/null 2>&1 \
    || "$venv_dir/bin/python3" -m pip install -q -r requirements.txt
}

ensure_cloudflared() {
  if ! command -v cloudflared >/dev/null 2>&1; then
    echo "==> Installing cloudflared (one-time)"
    local arch; arch="$(uname -m)"
    case "$arch" in
      x86_64) arch="amd64" ;;
      aarch64|arm64) arch="arm64" ;;
      *) echo "ERROR: no cloudflared auto-install for arch '$arch'. Install manually: https://github.com/cloudflare/cloudflared/releases" >&2; exit 1 ;;
    esac
    mkdir -p "$HOME/.local/bin"
    curl -fL -o "$HOME/.local/bin/cloudflared" \
      "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-${arch}"
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
  local spk_model="$script_dir/models/speaker_embedding_campplus.onnx"
  local speaker_flag=""
  if [ ! -f "$spk_model" ]; then
    echo "==> Speaker-ID model missing (27M, optional) — download it with:" >&2
    echo "    curl -L -o '$spk_model' \\" >&2
    echo "      https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced.onnx" >&2
    echo "    Continuing without speaker-ID (--no-speaker-id)." >&2
    speaker_flag="--no-speaker-id"
  fi
  "$venv_dir/bin/python3" server.py --model-dir "$model_dir" --host 0.0.0.0 --port 8100 \
    $speaker_flag &
  server_pid=$!
}

do_serve() {
  ensure_deps
  check_port_free
  echo "==> Starting live UI locally: http://localhost:8100"
  launch_server
  wait "$server_pid"
}

do_startui() {
  ensure_deps
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
  serve) do_serve ;;
  startui) do_startui ;;
  *)
    echo "Unknown command: $cmd" >&2
    usage
    exit 1
    ;;
esac
