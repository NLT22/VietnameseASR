#!/usr/bin/env bash
set -euo pipefail

stage=0
stop_stage=100

corpus_root="$PWD"
exp_dir="$PWD/ASR/zipformer/exp"
bpe_dir="$PWD/data/lang_bpe_500"

num_epochs=1
world_size=1
max_duration=30

while [[ $# -gt 0 ]]; do
  case "$1" in
    --stage)
      stage="$2"
      shift 2
      ;;
    --stop_stage|--stop-stage|-stop_stage|-stop-stage)
      stop_stage="$2"
      shift 2
      ;;
    --num_epochs|--num-epochs)
      num_epochs="$2"
      shift 2
      ;;
    --world_size|--world-size)
      world_size="$2"
      shift 2
      ;;
    --max_duration|--max-duration)
      max_duration="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1" >&2
      echo "Usage: $0 [--stage N] [--stop_stage N] [--num_epochs N] [--world_size N] [--max_duration N]" >&2
      exit 1
      ;;
  esac
done

if [ "$stage" -le 0 ] && [ "$stop_stage" -ge 0 ]; then
  echo "Stage 0: Audit dataset"
  python3 audit_dataset.py
fi

if [ "$stage" -le 1 ] && [ "$stop_stage" -ge 1 ]; then
  echo "Stage 1: Prepare manifests"
  python3 prepare_manifests.py
fi

if [ "$stage" -le 2 ] && [ "$stop_stage" -ge 2 ]; then
  echo "Stage 2: Fix manifests"
  mkdir -p manifests_fixed

  lhotse fix manifests/train_recordings.jsonl.gz manifests/train_supervisions.jsonl.gz manifests_fixed
  lhotse fix manifests/dev_recordings.jsonl.gz manifests/dev_supervisions.jsonl.gz manifests_fixed
  lhotse fix manifests/test_recordings.jsonl.gz manifests/test_supervisions.jsonl.gz manifests_fixed
fi

if [ "$stage" -le 3 ] && [ "$stop_stage" -ge 3 ]; then
  echo "Stage 3: Export text corpus"
  python3 local/export_text_corpus.py
fi

if [ "$stage" -le 4 ] && [ "$stop_stage" -ge 4 ]; then
  echo "Stage 4: Train BPE"
  python3 local/train_bpe_model.py
fi

if [ "$stage" -le 5 ] && [ "$stop_stage" -ge 5 ]; then
  echo "Stage 5: Compute fbank"
  python3 local/compute_fbank.py
fi

if [ "$stage" -le 6 ] && [ "$stop_stage" -ge 6 ]; then
  echo "Stage 6: Tokenize smoke test"
  python3 local/tokenize_test.py
fi

if [ "$stage" -le 7 ] && [ "$stop_stage" -ge 7 ]; then
  echo "Stage 7: Train"

  python3 ASR/zipformer/train.py \
    --world-size "${world_size}" \
    --num-epochs "${num_epochs}" \
    --start-epoch 1 \
    --use-fp16 1 \
    --manifest-dir ./fbank \
    --base-lr 0.02 \
    --exp-dir "${exp_dir}" \
    --max-duration "${max_duration}" \
    --bpe-model "${bpe_dir}/bpe.model" \
    --enable-musan 0 \
    --bucketing-sampler 0 \
    --enable-spec-aug 0
fi

if [ "$stage" -le 8 ] && [ "$stop_stage" -ge 8 ]; then
  echo "Stage 8: Decode"

  epoch="${num_epochs}"
  avg=1

  python3 ASR/zipformer/decode.py \
    --epoch "${epoch}" \
    --avg "${avg}" \
    --use-averaged-model 0 \
    --exp-dir "${exp_dir}" \
    --manifest-dir ./fbank \
    --bpe-model "${bpe_dir}/bpe.model" \
    --max-duration "${max_duration}" \
    --decoding-method greedy_search \
    --bucketing-sampler 0
fi
