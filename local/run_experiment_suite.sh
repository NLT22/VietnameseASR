#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
recipe_dir="$(cd -- "${script_dir}/.." && pwd)"
cd "${recipe_dir}"

model_sizes="${MODEL_SIZES:-base small}"
data_variant="${DATA_VARIANT:-raw}"
scratch_epochs="${SCRATCH_EPOCHS:-50}"
finetune_epochs="${FINETUNE_EPOCHS:-30}"
scratch_lr="${SCRATCH_LR:-0.01}"
finetune_lr="${FINETUNE_LR:-0.001}"
vocab_size="${VOCAB_SIZE:-100}"
max_duration="${MAX_DURATION:-50}"
world_size="${WORLD_SIZE:-1}"
use_fp16="${USE_FP16:-0}"
avg="${AVG:-1}"
use_averaged_model="${USE_AVERAGED_MODEL:-0}"
decode_method="${DECODE_METHOD:-all}"
run_scratch="${RUN_SCRATCH:-1}"
run_finetune="${RUN_FINETUNE:-1}"
run_prepare="${RUN_PREPARE:-0}"
prepare_stage="${PREPARE_STAGE:-0}"
prepare_stop_stage="${PREPARE_STOP_STAGE:-12}"
result_dir="${RESULT_DIR:-exp_suite_results}"
clean_checkpoints="${CLEAN_CHECKPOINTS:-1}"
clean_keep_epochs="${CLEAN_KEEP_EPOCHS:-30 50}"

base_ckpt="${BASE_PRETRAIN_CKPT:-pretrained/zipformer_base_ls960/exp/pretrained.pt}"
small_ckpt="${SMALL_PRETRAIN_CKPT:-pretrained/zipformer_small_ls960/exp/pretrained.pt}"

mkdir -p "${result_dir}/logs"

metadata="${result_dir}/experiments.tsv"
summary_md="${result_dir}/summary.md"
summary_csv="${result_dir}/summary.csv"

printf "name\tmodel\tdata_variant\tmode\tepochs\texp_dir\tcheckpoint\n" > "${metadata}"

log_msg() {
  printf "[%s] %s\n" "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

exp_dir_for() {
  local model="$1"
  local suffix="$2"
  local variant_suffix=""
  if [ "${data_variant}" = "nr" ]; then
    variant_suffix="_nr"
  fi

  if [ "${model}" = "base" ]; then
    printf "%s/ASR/zipformer/exp_bpe%s%s%s" "${recipe_dir}" "${vocab_size}" "${variant_suffix}" "${suffix}"
  else
    printf "%s/ASR/zipformer/exp_bpe%s_%s%s%s" "${recipe_dir}" "${vocab_size}" "${model}" "${variant_suffix}" "${suffix}"
  fi
}

pretrain_ckpt_for() {
  local model="$1"
  if [ "${model}" = "base" ]; then
    printf "%s" "${base_ckpt}"
  elif [ "${model}" = "small" ]; then
    printf "%s" "${small_ckpt}"
  else
    log_msg "ERROR: unsupported model '${model}' for pretrained checkpoint lookup"
    exit 1
  fi
}

require_real_checkpoint() {
  local ckpt="$1"
  if [ ! -s "${ckpt}" ]; then
    log_msg "ERROR: pretrained checkpoint not found or empty: ${ckpt}"
    exit 1
  fi

  local size
  size="$(stat -c '%s' "${ckpt}")"
  if [ "${size}" -lt 1048576 ]; then
    log_msg "ERROR: checkpoint is too small (${size} bytes), likely a Git LFS pointer: ${ckpt}"
    log_msg "Run: git -C <repo-dir> lfs pull"
    exit 1
  fi
}

run_and_log() {
  local log_file="$1"
  shift
  log_msg "Running: $*"
  "$@" 2>&1 | tee "${log_file}"
}

cleanup_exp_dir() {
  local exp_dir="$1"
  if [ "${clean_checkpoints}" != "1" ]; then
    return 0
  fi
  if [ ! -f "local/cleanup_checkpoints.sh" ]; then
    log_msg "WARNING: cleanup script missing: local/cleanup_checkpoints.sh"
    return 0
  fi
  log_msg "Cleaning checkpoints in ${exp_dir}; keeping epochs: ${clean_keep_epochs}"
  bash local/cleanup_checkpoints.sh \
    --keep-epochs "${clean_keep_epochs}" \
    "${exp_dir}"
}

if [ "${run_prepare}" = "1" ]; then
  log_file="${result_dir}/logs/prepare_${data_variant}.log"
  run_and_log "${log_file}" \
    bash run.sh \
      --data_variant "${data_variant}" \
      --vocab_size "${vocab_size}" \
      --stage "${prepare_stage}" \
      --stop_stage "${prepare_stop_stage}"
fi

for model in ${model_sizes}; do
  if [ "${model}" != "base" ] && [ "${model}" != "small" ]; then
    log_msg "ERROR: MODEL_SIZES only supports: base small. Got '${model}'"
    exit 1
  fi

  if [ "${run_scratch}" = "1" ]; then
    suffix="_scratch${scratch_epochs}"
    name="${model}_${data_variant}_scratch${scratch_epochs}"
    exp_dir="$(exp_dir_for "${model}" "${suffix}")"
    printf "%s\t%s\t%s\tscratch\t%s\t%s\t\n" \
      "${name}" "${model}" "${data_variant}" "${scratch_epochs}" "${exp_dir}" >> "${metadata}"

    log_file="${result_dir}/logs/${name}.log"
    run_and_log "${log_file}" \
      bash run.sh \
        --data_variant "${data_variant}" \
        --vocab_size "${vocab_size}" \
        --model_size "${model}" \
        --num_epochs "${scratch_epochs}" \
        --base_lr "${scratch_lr}" \
        --max_duration "${max_duration}" \
        --world_size "${world_size}" \
        --use_fp16 "${use_fp16}" \
        --avg "${avg}" \
        --use_averaged_model "${use_averaged_model}" \
        --decode_method "${decode_method}" \
        --exp_suffix "${suffix}" \
        --stage 13 \
        --stop_stage 14
    cleanup_exp_dir "${exp_dir}"
  fi

  if [ "${run_finetune}" = "1" ]; then
    ckpt="$(pretrain_ckpt_for "${model}")"
    require_real_checkpoint "${ckpt}"

    suffix="_finetune${finetune_epochs}"
    name="${model}_${data_variant}_finetune${finetune_epochs}"
    exp_dir="$(exp_dir_for "${model}" "${suffix}")"
    printf "%s\t%s\t%s\tpretrain_encoder\t%s\t%s\t%s\n" \
      "${name}" "${model}" "${data_variant}" "${finetune_epochs}" "${exp_dir}" "${ckpt}" >> "${metadata}"

    log_file="${result_dir}/logs/${name}.log"
    run_and_log "${log_file}" \
      bash run.sh \
        --data_variant "${data_variant}" \
        --vocab_size "${vocab_size}" \
        --model_size "${model}" \
        --num_epochs "${finetune_epochs}" \
        --base_lr "${finetune_lr}" \
        --max_duration "${max_duration}" \
        --world_size "${world_size}" \
        --use_fp16 "${use_fp16}" \
        --avg "${avg}" \
        --use_averaged_model "${use_averaged_model}" \
        --decode_method "${decode_method}" \
        --do_finetune 1 \
        --finetune_ckpt "${ckpt}" \
        --init_modules encoder \
        --exp_suffix "${suffix}" \
        --stage 13 \
        --stop_stage 14
    cleanup_exp_dir "${exp_dir}"
  fi
done

python3 local/summarize_experiments.py \
  --metadata "${metadata}" \
  --output "${summary_md}" \
  --csv-output "${summary_csv}"

log_msg "Done. Summary: ${summary_md}"
