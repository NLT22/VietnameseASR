#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$script_dir"

stage=0
stop_stage=3
num_epochs=50
max_duration=60
exp_suffix="_robust"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --stage) stage="$2"; shift 2 ;;
    --stop_stage|--stop-stage) stop_stage="$2"; shift 2 ;;
    --num_epochs|--num-epochs) num_epochs="$2"; shift 2 ;;
    --max_duration|--max-duration) max_duration="$2"; shift 2 ;;
    --exp_suffix|--exp-suffix) exp_suffix="$2"; shift 2 ;;
    *)
      echo "Unknown option: $1" >&2
      exit 1
      ;;
  esac
done

if [ "$stage" -le 0 ] && [ "$stop_stage" -ge 0 ]; then
  echo "Stage 0: Rebuild clean splits from the original recordings"
  bash run.sh --stage -1 --stop_stage 7
  python3 local/audit_corpus.py
fi

if [ "$stage" -le 1 ] && [ "$stop_stage" -ge 1 ]; then
  echo "Stage 1: Prepare MUSAN manifests"
  python3 - <<'PY'
from lhotse.recipes import prepare_musan
prepare_musan(corpus_dir="musan", output_dir="data/manifests")
PY

  echo "Stage 1: Rebuild features with speed perturbation and keep long utterances"
  bash run.sh \
    --stage 8 \
    --stop_stage 12 \
    --perturb_speed 1 \
    --enable_musan 1 \
    --fbank_max_duration 40 \
    --overwrite_fbank 1
fi

if [ "$stage" -le 2 ] && [ "$stop_stage" -ge 2 ]; then
  echo "Stage 2: Train with dynamic MUSAN mixing and SpecAugment"
  bash run.sh \
    --stage 13 \
    --stop_stage 13 \
    --model_size small \
    --num_epochs "$num_epochs" \
    --max_duration "$max_duration" \
    --base_lr 0.005 \
    --enable_musan 1 \
    --enable_spec_aug 1 \
    --use_ctc 1 \
    --ctc_loss_scale 0.2 \
    --exp_suffix "$exp_suffix"
fi

if [ "$stage" -le 3 ] && [ "$stop_stage" -ge 3 ]; then
  echo "Stage 3: Decode using checkpoint averaging"
  bash run.sh \
    --stage 14 \
    --stop_stage 14 \
    --model_size small \
    --num_epochs "$num_epochs" \
    --max_duration "$max_duration" \
    --avg 10 \
    --decode_method beam_search \
    --exp_suffix "$exp_suffix"
fi
