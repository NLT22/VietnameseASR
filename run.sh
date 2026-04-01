#!/usr/bin/env bash
set -euo pipefail

stage=0
stop_stage=100

corpus_root="$PWD"
exp_dir="$PWD/ASR/zipformer/exp_100bpe_0.02"
bpe_dir="$PWD/data/lang_bpe_100"

num_epochs=30
world_size=1
max_duration=100
base_lr=0.02
use_fp16=1

enable_musan=0
enable_spec_aug=0
bucketing_sampler=1
num_buckets=4
perturb_speed=0

decode_method="greedy_search"
use_averaged_model=0
avg=1

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
    --base_lr|--base-lr)
      base_lr="$2"
      shift 2
      ;;
    --use_fp16|--use-fp16)
      use_fp16="$2"
      shift 2
      ;;
    --enable_musan|--enable-musan)
      enable_musan="$2"
      shift 2
      ;;
    --enable_spec_aug|--enable-spec-aug)
      enable_spec_aug="$2"
      shift 2
      ;;
    --bucketing_sampler|--bucketing-sampler)
      bucketing_sampler="$2"
      shift 2
      ;;
    --num_buckets|--num-buckets)
      num_buckets="$2"
      shift 2
      ;;
    --perturb_speed|--perturb-speed)
      perturb_speed="$2"
      shift 2
      ;;
    --decode_method|--decode-method)
      decode_method="$2"
      shift 2
      ;;
    --use_averaged_model|--use-averaged-model)
      use_averaged_model="$2"
      shift 2
      ;;
    --avg)
      avg="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1" >&2
      echo "Usage: $0 [--stage N] [--stop_stage N] [--num_epochs N] [--world_size N] [--max_duration N] [--base_lr X] [--use_fp16 0|1] [--enable_musan 0|1] [--enable_spec_aug 0|1] [--bucketing_sampler 0|1] [--num_buckets N] [--perturb_speed 0|1] [--decode_method NAME] [--use_averaged_model 0|1] [--avg N]" >&2
      exit 1
      ;;
  esac
done

echo "corpus_root=$corpus_root"
echo "exp_dir=$exp_dir"
echo "bpe_dir=$bpe_dir"
echo "num_epochs=$num_epochs"
echo "world_size=$world_size"
echo "max_duration=$max_duration"
echo "base_lr=$base_lr"
echo "use_fp16=$use_fp16"
echo "enable_musan=$enable_musan"
echo "enable_spec_aug=$enable_spec_aug"
echo "bucketing_sampler=$bucketing_sampler"
echo "num_buckets=$num_buckets"
echo "perturb_speed=$perturb_speed"
echo "decode_method=$decode_method"
echo "use_averaged_model=$use_averaged_model"
echo "avg=$avg"

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
  echo "Stage 5: Prepare minimal lang dir"
  python3 local/prepare_lang_bpe.py --lang-dir "${bpe_dir}"
fi

if [ "$stage" -le 6 ] && [ "$stop_stage" -ge 6 ]; then
  echo "Stage 6: Compute fbank / cuts"
  cmd=(
    python3 local/compute_fbank.py
    --bpe-model "${bpe_dir}/bpe.model"
    --manifest-dir manifests_fixed
    --output-dir fbank
  )
  if [ "$perturb_speed" = "1" ]; then
    cmd+=(--perturb-speed)
  fi
  "${cmd[@]}"
fi

if [ "$stage" -le 7 ] && [ "$stop_stage" -ge 7 ]; then
  echo "Stage 7: Validate cut manifests"
  python3 local/validate_manifest.py --all --manifest-dir fbank
fi

if [ "$stage" -le 8 ] && [ "$stop_stage" -ge 8 ]; then
  echo "Stage 8: Display manifest statistics"
  python3 local/display_manifest_statistics.py --all --manifest-dir fbank
fi

if [ "$stage" -le 9 ] && [ "$stop_stage" -ge 9 ]; then
  echo "Stage 9: Tokenize smoke test"
  python3 local/tokenize_test.py
fi

if [ "$stage" -le 10 ] && [ "$stop_stage" -ge 10 ]; then
  echo "Stage 10: Train"
  python3 ASR/zipformer/train.py \
    --world-size "${world_size}" \
    --num-epochs "${num_epochs}" \
    --start-epoch 1 \
    --use-fp16 "${use_fp16}" \
    --manifest-dir ./fbank \
    --base-lr "${base_lr}" \
    --exp-dir "${exp_dir}" \
    --max-duration "${max_duration}" \
    --bpe-model "${bpe_dir}/bpe.model" \
    --enable-musan "${enable_musan}" \
    --enable-spec-aug "${enable_spec_aug}" \
    --bucketing-sampler "${bucketing_sampler}" \
    --num-buckets "${num_buckets}"
fi

if [ "$stage" -le 11 ] && [ "$stop_stage" -ge 11 ]; then
  echo "Stage 11: Decode"
  epoch="${num_epochs}"

  python3 ASR/zipformer/decode.py \
    --epoch "${epoch}" \
    --avg "${avg}" \
    --use-averaged-model "${use_averaged_model}" \
    --exp-dir "${exp_dir}" \
    --manifest-dir ./fbank \
    --bpe-model "${bpe_dir}/bpe.model" \
    --max-duration "${max_duration}" \
    --decoding-method "${decode_method}" \
    --bucketing-sampler 0
fi

if [ "$stage" -le 12 ] && [ "$stop_stage" -ge 12 ]; then
  echo "Stage 12: TensorBoard"
  echo "cd ~/icefall/egs/vi_asr_corpus"
  echo "tensorboard --logdir ASR/zipformer/exp/tensorboard --port 6006"
fi

if [ "$stage" -le 13 ] && [ "$stop_stage" -ge 13 ]; then
  echo "Stage 13: Show result files"
  echo "cd ~/icefall/egs/vi_asr_corpus"
  echo "ls ASR/zipformer"
  echo "ls ${exp_dir}"
  echo "find ${exp_dir} -maxdepth 2 -type f | sort"
fi
