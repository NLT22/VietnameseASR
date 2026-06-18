#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$script_dir"

stage=0
stop_stage=4
num_epochs=200
max_duration=50
output_root="/media/pc/aa20c6a6-01d2-4236-ad76-9a4e1fb2ab4a/VietnameseASR-closed-set"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --stage) stage="$2"; shift 2 ;;
    --stop_stage|--stop-stage) stop_stage="$2"; shift 2 ;;
    --num_epochs|--num-epochs) num_epochs="$2"; shift 2 ;;
    --max_duration|--max-duration) max_duration="$2"; shift 2 ;;
    --output_root|--output-root) output_root="$2"; shift 2 ;;
    *) echo "Unknown option: $1" >&2; exit 1 ;;
  esac
done

workspace="$output_root/workspace"
transcript_dir="$workspace/transcripts"
manifest_dir="$workspace/data/manifests"
fixed_manifest_dir="$manifest_dir/fixed"
lang_dir="$workspace/data/lang_bpe_500"
fbank_dir="$workspace/fbank"
exp_dir="$output_root/exp"

if [ "$stage" -le 0 ] && [ "$stop_stage" -ge 0 ]; then
  mkdir -p "$workspace"
  python3 local/prepare_closed_set.py --output-dir "$transcript_dir"
  python3 local/audit_corpus.py --transcript-dir "$transcript_dir"
fi

if [ "$stage" -le 1 ] && [ "$stop_stage" -ge 1 ]; then
  echo "Stage 1: Inspect each original recording once and reuse the closed-set manifest"
  python3 prepare_manifests.py \
    --transcript-dir "$transcript_dir" \
    --output-dir "$manifest_dir" \
    --dataset train
  for split in dev test; do
    cp "$manifest_dir/train_recordings.jsonl.gz" "$manifest_dir/${split}_recordings.jsonl.gz"
    cp "$manifest_dir/train_supervisions.jsonl.gz" "$manifest_dir/${split}_supervisions.jsonl.gz"
  done
  mkdir -p "$fixed_manifest_dir"
  for split in train dev test; do
    lhotse fix \
      "$manifest_dir/${split}_recordings.jsonl.gz" \
      "$manifest_dir/${split}_supervisions.jsonl.gz" \
      "$fixed_manifest_dir"
  done
  python3 local/export_text_corpus.py \
    --transcript-dir "$transcript_dir" \
    --output-text "$workspace/lang/transcript_words.txt"
  python3 local/train_bpe_model.py \
    --root "$workspace" \
    --input-txt lang/transcript_words.txt \
    --vocab-size 500
  python3 local/prepare_lang_bpe.py --lang-dir "$lang_dir"
fi

if [ "$stage" -le 2 ] && [ "$stop_stage" -ge 2 ]; then
  echo "Stage 2: Compute 16 kHz train features with speed perturbation (HDD-friendly workers)"
  python3 local/compute_fbank.py \
    --bpe-model "$lang_dir/bpe.model" \
    --manifest-dir "$fixed_manifest_dir" \
    --output-dir "$fbank_dir" \
    --dataset train \
    --max-duration 40 \
    --sampling-rate 16000 \
    --num-jobs 2 \
    --perturb-speed \
    --overwrite

  echo "Stage 2: Compute original validation features once and reuse them for test"
  python3 local/compute_fbank.py \
    --bpe-model "$lang_dir/bpe.model" \
    --manifest-dir "$fixed_manifest_dir" \
    --output-dir "$fbank_dir" \
    --dataset dev \
    --max-duration 40 \
    --sampling-rate 16000 \
    --num-jobs 2 \
    --overwrite
  cp "$fbank_dir/dev_cuts.jsonl.gz" "$fbank_dir/test_cuts.jsonl.gz"
  python3 local/validate_manifest.py --all --manifest-dir "$fbank_dir"
fi

if [ "$stage" -le 3 ] && [ "$stop_stage" -ge 3 ]; then
  mkdir -p "$exp_dir"
  python3 ASR/zipformer/train.py \
    --world-size 1 \
    --num-epochs "$num_epochs" \
    --start-epoch 1 \
    --use-fp16 0 \
    --manifest-dir "$fbank_dir" \
    --base-lr 0.005 \
    --exp-dir "$exp_dir" \
    --max-duration "$max_duration" \
    --bpe-model "$lang_dir/bpe.model" \
    --enable-musan 0 \
    --enable-spec-aug 1 \
    --bucketing-sampler 1 \
    --num-buckets 4 \
    --num-encoder-layers 2,2,2,2,2,2 \
    --feedforward-dim 512,768,768,768,768,768 \
    --num-heads 4,4,4,8,4,4 \
    --encoder-dim 192,256,256,256,256,256 \
    --encoder-unmasked-dim 192,192,192,192,192,192 \
    --decoder-dim 512 \
    --joiner-dim 512 \
    --use-ctc 1 \
    --ctc-loss-scale 0.2 \
    --save-every-n 1000000000 \
    --keep-last-k 1 \
    --keep-epoch-checkpoints false \
    --save-best-train-loss false
fi

if [ "$stage" -le 4 ] && [ "$stop_stage" -ge 4 ]; then
  if [ ! -f "$exp_dir/best-valid-loss.pt" ]; then
    echo "Missing best checkpoint: $exp_dir/best-valid-loss.pt" >&2
    exit 1
  fi
  ln -sfn best-valid-loss.pt "$exp_dir/epoch-0.pt"
  python3 ASR/zipformer/decode.py \
    --epoch 0 \
    --avg 1 \
    --use-averaged-model 0 \
    --exp-dir "$exp_dir" \
    --manifest-dir "$fbank_dir" \
    --bpe-model "$lang_dir/bpe.model" \
    --max-duration "$max_duration" \
    --decoding-method beam_search \
    --bucketing-sampler 0 \
    --num-encoder-layers 2,2,2,2,2,2 \
    --feedforward-dim 512,768,768,768,768,768 \
    --num-heads 4,4,4,8,4,4 \
    --encoder-dim 192,256,256,256,256,256 \
    --encoder-unmasked-dim 192,192,192,192,192,192 \
    --decoder-dim 512 \
    --joiner-dim 512

  recogs="$(find "$exp_dir/beam_search" -type f -name 'recogs-test-*' -printf '%T@ %p\n' | sort -nr | head -1 | cut -d' ' -f2-)"
  python3 local/closed_set_match.py \
    --recogs "$recogs" \
    --catalog "$transcript_dir/train.tsv"
  rm -f "$exp_dir/epoch-0.pt"
fi
