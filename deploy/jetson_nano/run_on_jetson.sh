#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/../.." && pwd)"
package="$repo_root/deploy/vietnamese_asr_jetson_nano.tar.gz"

jetson_host="${JETSON_HOST:-}"
jetson_dir="${JETSON_DIR:-vietnamese_asr_eval}"
manifest="$repo_root/transcripts/test.tsv"
audio_root="$repo_root"
threads="${THREADS:-2}"
provider="${PROVIDER:-cpu}"
remote_install_env=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --host) jetson_host="$2"; shift 2 ;;
    --dir) jetson_dir="$2"; shift 2 ;;
    --manifest) manifest="$2"; shift 2 ;;
    --audio-root) audio_root="$2"; shift 2 ;;
    --threads) threads="$2"; shift 2 ;;
    --provider) provider="$2"; shift 2 ;;
    *)
      echo "Unknown option: $1" >&2
      exit 1
      ;;
  esac
done

if [[ -z "$jetson_host" ]]; then
  cat >&2 <<'EOF'
Set the Jetson SSH target, for example:

  JETSON_HOST=trung@192.168.1.55 bash run_on_jetson.sh

Or:

  bash run_on_jetson.sh --host trung@192.168.1.55
EOF
  exit 1
fi

if [[ -n "${SUDO_PASSWORD:-}" ]]; then
  remote_install_env="SUDO_PASSWORD=$(printf '%q' "$SUDO_PASSWORD") "
fi

if [[ ! -f "$package" ]]; then
  echo "Missing package: $package" >&2
  echo "Create it from VietnameseASR/deploy with:" >&2
  echo "  tar -czf vietnamese_asr_jetson_nano.tar.gz jetson_nano" >&2
  exit 1
fi

tmp_package="/tmp/vietnamese_asr_jetson_nano.tar.gz"
remote_base="$(ssh "$jetson_host" "mkdir -p '$jetson_dir' && cd '$jetson_dir' && pwd")"
scp "$package" "$jetson_host:$tmp_package"
ssh "$jetson_host" "tar -xzf '$tmp_package' -C '$remote_base'"

ssh "$jetson_host" "mkdir -p '$remote_base/data/VietnameseASR/transcripts' '$remote_base/data/VietnameseASR/audio'"
scp "$manifest" "$jetson_host:$remote_base/data/VietnameseASR/transcripts/test.tsv"
rsync -av "$audio_root/audio/test" "$jetson_host:$remote_base/data/VietnameseASR/audio/"

ssh "$jetson_host" "cd '$remote_base/jetson_nano' && ${remote_install_env}bash install_jetson.sh"
ssh "$jetson_host" "cd '$remote_base/jetson_nano' && . .venv/bin/activate && bash run_performance_eval.sh --manifest '$remote_base/data/VietnameseASR/transcripts/test.tsv' --audio-root '$remote_base/data/VietnameseASR' --threads '$threads' --provider '$provider'"

mkdir -p "$script_dir/perf_results/jetson"
scp "$jetson_host:$remote_base/jetson_nano/perf_results/"* "$script_dir/perf_results/jetson/"
