#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$script_dir"

manifest=""
audio_root=""
threads="${THREADS:-2}"
limit="${LIMIT:-0}"
provider="${PROVIDER:-cpu}"
max_active_paths="${MAX_ACTIVE_PATHS:-20}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --manifest) manifest="$2"; shift 2 ;;
    --audio-root) audio_root="$2"; shift 2 ;;
    --threads) threads="$2"; shift 2 ;;
    --limit) limit="$2"; shift 2 ;;
    --provider) provider="$2"; shift 2 ;;
    --max-active-paths) max_active_paths="$2"; shift 2 ;;
    *)
      echo "Unknown option: $1" >&2
      exit 1
      ;;
  esac
done

if [[ -z "$manifest" || -z "$audio_root" ]]; then
  cat >&2 <<'EOF'
Usage:
  bash run_performance_eval.sh --manifest /path/to/test.tsv --audio-root /path/to/VietnameseASR

Optional:
  --threads N
  --limit N
  --provider cpu|cuda
  --max-active-paths N
EOF
  exit 1
fi

mkdir -p perf_results

python3 evaluate_performance.py \
  --manifest "$manifest" \
  --audio-root "$audio_root" \
  --threads "$threads" \
  --limit "$limit" \
  --provider "$provider" \
  --max-active-paths "$max_active_paths" \
  --output-json "perf_results/int8_${provider}_threads${threads}.json" \
  --output-md "perf_results/int8_${provider}_threads${threads}.md"

python3 evaluate_performance.py \
  --manifest "$manifest" \
  --audio-root "$audio_root" \
  --threads "$threads" \
  --limit "$limit" \
  --provider "$provider" \
  --max-active-paths "$max_active_paths" \
  --fp32 \
  --output-json "perf_results/fp32_${provider}_threads${threads}.json" \
  --output-md "perf_results/fp32_${provider}_threads${threads}.md"
