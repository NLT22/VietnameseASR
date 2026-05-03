#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$script_dir"

stage=0
stop_stage=100

corpus_root="$script_dir"

vocab_size=100
bpe_dir="$PWD/data/lang_bpe_${vocab_size}"
manifest_dir="$PWD/data/manifests"
fixed_manifest_dir="$PWD/data/manifests/fixed"
transcript_dir="transcripts"
audio_root="audio"
fbank_dir="fbank"
musan_manifest_dir="$PWD/data/manifests"

num_epochs=30
world_size=1
max_duration=50
base_lr=0.01
use_fp16=0

enable_musan=0
enable_spec_aug=0
bucketing_sampler=1
num_buckets=4
perturb_speed=0
musan_dir="$PWD/musan"

offline_musan_aug=0
copies_per_utt=10
snr_min=10
snr_max=20

decode_method="all"
use_averaged_model=0
avg=1

# Streaming (causal) — train với --causal 1 rồi decode bằng streaming_decode.py
causal=0
chunk_size="16,32,64,-1"          # dùng khi train (list)
left_context_frames="64,128,256,-1" # dùng khi train (list)
decode_chunk_size=32               # dùng khi streaming decode (single value)
decode_left_context_frames=256     # dùng khi streaming decode (single value)
streaming_decode_method="greedy_search" # greedy_search | modified_beam_search

use_ctc=0
use_cr_ctc=0
ctc_loss_scale=0.2
cr_loss_scale=0.2
attention_decoder_loss_scale=0.0

do_finetune=0
finetune_ckpt=""
init_modules="encoder"
exp_suffix=""

data_variant="raw"
enable_nr=0

model_size="base"
num_encoder_layers="2,2,3,4,3,2"
feedforward_dim="512,768,1024,1536,1024,768"
num_heads="4,4,4,8,4,4"
encoder_dim="192,256,384,512,384,256"
encoder_unmasked_dim="192,192,256,256,256,192"
decoder_dim=512
joiner_dim=512

while [[ $# -gt 0 ]]; do
  case "$1" in
    --stage) stage="$2"; shift 2 ;;
    --stop_stage|--stop-stage|-stop_stage|-stop-stage) stop_stage="$2"; shift 2 ;;
    --vocab_size|--vocab-size) vocab_size="$2"; shift 2 ;;
    --num_epochs|--num-epochs) num_epochs="$2"; shift 2 ;;
    --world_size|--world-size) world_size="$2"; shift 2 ;;
    --max_duration|--max-duration) max_duration="$2"; shift 2 ;;
    --base_lr|--base-lr) base_lr="$2"; shift 2 ;;
    --use_fp16|--use-fp16) use_fp16="$2"; shift 2 ;;
    --enable_musan|--enable-musan) enable_musan="$2"; shift 2 ;;
    --enable_spec_aug|--enable-spec-aug) enable_spec_aug="$2"; shift 2 ;;
    --bucketing_sampler|--bucketing-sampler) bucketing_sampler="$2"; shift 2 ;;
    --num_buckets|--num-buckets) num_buckets="$2"; shift 2 ;;
    --perturb_speed|--perturb-speed) perturb_speed="$2"; shift 2 ;;
    --musan_dir|--musan-dir) musan_dir="$2"; shift 2 ;;
    --offline_musan_aug|--offline-musan-aug) offline_musan_aug="$2"; shift 2 ;;
    --copies_per_utt|--copies-per-utt) copies_per_utt="$2"; shift 2 ;;
    --snr_min|--snr-min) snr_min="$2"; shift 2 ;;
    --snr_max|--snr-max) snr_max="$2"; shift 2 ;;
    --decode_method|--decode-method|--decode_methods|--decode-methods) decode_method="$2"; shift 2 ;;
    --use_averaged_model|--use-averaged-model) use_averaged_model="$2"; shift 2 ;;
    --avg) avg="$2"; shift 2 ;;
    --causal) causal="$2"; shift 2 ;;
    --chunk_size|--chunk-size) chunk_size="$2"; shift 2 ;;
    --left_context_frames|--left-context-frames) left_context_frames="$2"; shift 2 ;;
    --decode_chunk_size|--decode-chunk-size) decode_chunk_size="$2"; shift 2 ;;
    --decode_left_context_frames|--decode-left-context-frames) decode_left_context_frames="$2"; shift 2 ;;
    --streaming_decode_method|--streaming-decode-method) streaming_decode_method="$2"; shift 2 ;;
    --use_ctc|--use-ctc) use_ctc="$2"; shift 2 ;;
    --use_cr_ctc|--use-cr-ctc) use_cr_ctc="$2"; shift 2 ;;
    --ctc_loss_scale|--ctc-loss-scale) ctc_loss_scale="$2"; shift 2 ;;
    --cr_loss_scale|--cr-loss-scale) cr_loss_scale="$2"; shift 2 ;;
    --attention_decoder_loss_scale|--attention-decoder-loss-scale) attention_decoder_loss_scale="$2"; shift 2 ;;
    --do_finetune|--do-finetune) do_finetune="$2"; shift 2 ;;
    --finetune_ckpt|--finetune-ckpt) finetune_ckpt="$2"; shift 2 ;;
    --init_modules|--init-modules) init_modules="$2"; shift 2 ;;
    --exp_suffix|--exp-suffix) exp_suffix="$2"; shift 2 ;;
    --data_variant|--data-variant) data_variant="$2"; shift 2 ;;
    --enable_nr|--enable-nr) enable_nr="$2"; shift 2 ;;
    --model_size|--model-size) model_size="$2"; shift 2 ;;
    --num_encoder_layers|--num-encoder-layers) num_encoder_layers="$2"; shift 2 ;;
    --feedforward_dim|--feedforward-dim) feedforward_dim="$2"; shift 2 ;;
    --num_heads|--num-heads) num_heads="$2"; shift 2 ;;
    --encoder_dim|--encoder-dim) encoder_dim="$2"; shift 2 ;;
    --encoder_unmasked_dim|--encoder-unmasked-dim) encoder_unmasked_dim="$2"; shift 2 ;;
    --decoder_dim|--decoder-dim) decoder_dim="$2"; shift 2 ;;
    --joiner_dim|--joiner-dim) joiner_dim="$2"; shift 2 ;;
    *)
      echo "Unknown option: $1" >&2
      echo "Usage: $0 [--stage N] [--stop_stage N] [--vocab_size N] [--num_epochs N] [--world_size N] [--max_duration N] [--base_lr X] [--use_fp16 0|1] [--enable_musan 0|1] [--enable_spec_aug 0|1] [--bucketing_sampler 0|1] [--num_buckets N] [--perturb_speed 0|1] [--musan_dir DIR] [--offline_musan_aug 0|1] [--copies_per_utt N] [--snr_min X] [--snr_max X] [--decode_method NAME|all] [--use_averaged_model 0|1] [--avg N] [--causal 0|1] [--chunk_size STR] [--left_context_frames STR] [--decode_chunk_size N] [--decode_left_context_frames N] [--streaming_decode_method NAME] [--use_ctc 0|1] [--use_cr_ctc 0|1] [--ctc_loss_scale X] [--cr_loss_scale X] [--attention_decoder_loss_scale X] [--do_finetune 0|1] [--finetune_ckpt PATH] [--init_modules encoder] [--exp_suffix TEXT] [--data_variant raw|nr] [--enable_nr 0|1] [--model_size base|small]" >&2
      exit 1
      ;;
  esac
done

case "$model_size" in
  base)
    ;;
  small)
    num_encoder_layers="2,2,2,2,2,2"
    feedforward_dim="512,768,768,768,768,768"
    num_heads="4,4,4,8,4,4"
    encoder_dim="192,256,256,256,256,256"
    encoder_unmasked_dim="192,192,192,192,192,192"
    decoder_dim=512
    joiner_dim=512
    ;;
  *)
    echo "ERROR: --model_size must be one of: base, small" >&2
    exit 1
    ;;
esac

case "$data_variant" in
  raw)
    transcript_dir="transcripts"
    audio_root="audio"
    manifest_dir="$PWD/data/manifests"
    fixed_manifest_dir="$PWD/data/manifests/fixed"
    fbank_dir="$PWD/fbank"
    variant_suffix=""
    ;;
  nr)
    transcript_dir="transcripts_nr"
    audio_root="audio_nr"
    manifest_dir="$PWD/data/manifests_nr"
    fixed_manifest_dir="$PWD/data/manifests_nr/fixed"
    fbank_dir="$PWD/fbank_nr"
    variant_suffix="_nr"
    ;;
  *)
    echo "ERROR: --data_variant must be one of: raw, nr" >&2
    exit 1
    ;;
esac

bpe_dir="$PWD/data/lang_bpe_${vocab_size}"
streaming_suffix=""
if [ "$causal" = "1" ]; then streaming_suffix="_streaming"; fi
if [ "$do_finetune" = "1" ] && [ -z "$exp_suffix" ]; then
  exp_suffix="_finetune"
fi
if [ "$model_size" = "base" ]; then
  exp_dir="$PWD/ASR/zipformer/exp_bpe${vocab_size}${streaming_suffix}${variant_suffix}${exp_suffix}"
else
  exp_dir="$PWD/ASR/zipformer/exp_bpe${vocab_size}_${model_size}${streaming_suffix}${variant_suffix}${exp_suffix}"
fi

echo "corpus_root=$corpus_root"
echo "vocab_size=$vocab_size"
echo "bpe_dir=$bpe_dir"
echo "exp_dir=$exp_dir"
echo "data_variant=$data_variant"
echo "enable_nr=$enable_nr"
echo "transcript_dir=$transcript_dir"
echo "audio_root=$audio_root"
echo "manifest_dir=$manifest_dir"
echo "fixed_manifest_dir=$fixed_manifest_dir"
echo "fbank_dir=$fbank_dir"
echo "musan_manifest_dir=$musan_manifest_dir"
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
echo "musan_dir=$musan_dir"
echo "offline_musan_aug=$offline_musan_aug"
echo "copies_per_utt=$copies_per_utt"
echo "snr_min=$snr_min"
echo "snr_max=$snr_max"
echo "decode_method=$decode_method"
echo "use_averaged_model=$use_averaged_model"
echo "avg=$avg"
echo "do_finetune=$do_finetune"
echo "finetune_ckpt=$finetune_ckpt"
echo "init_modules=$init_modules"
echo "exp_suffix=$exp_suffix"
echo "model_size=$model_size"
echo "num_encoder_layers=$num_encoder_layers"
echo "feedforward_dim=$feedforward_dim"
echo "num_heads=$num_heads"
echo "encoder_dim=$encoder_dim"
echo "encoder_unmasked_dim=$encoder_unmasked_dim"
echo "decoder_dim=$decoder_dim"
echo "joiner_dim=$joiner_dim"

if [ "$stage" -le 0 ] && [ "$stop_stage" -ge 0 ]; then
  echo "Stage 0: NoiseReduce audio variant (optional)"
  if [ "$enable_nr" = "1" ]; then
    python3 local/noise_reduce_audio.py \
      --input-transcript-dir transcripts \
      --output-transcript-dir transcripts_nr \
      --output-audio-root audio_nr \
      --overwrite
  else
    echo "Skip NoiseReduce because enable_nr=0"
  fi
fi

if [ "$stage" -le 1 ] && [ "$stop_stage" -ge 1 ]; then
  echo "Stage 1: Audit dataset"
  python3 audit_dataset.py --transcript-dir "${transcript_dir}"
fi

if [ "$stage" -le 2 ] && [ "$stop_stage" -ge 2 ]; then
  echo "Stage 2: Offline MUSAN augmentation for train split (optional)"
  if [ "$offline_musan_aug" = "1" ]; then
    if [ -z "$musan_dir" ]; then
      echo "ERROR: --offline_musan_aug 1 requires --musan_dir /path/to/musan" >&2
      exit 1
    fi
    python3 augment_train_with_musan.py \
      --corpus-root . \
      --musan-dir "$musan_dir" \
      --transcript-dir "$transcript_dir" \
      --audio-root "$audio_root" \
      --copies-per-utt "$copies_per_utt" \
      --snr-min "$snr_min" \
      --snr-max "$snr_max" \
      --overwrite
  else
    echo "Skip offline MUSAN augmentation because offline_musan_aug=0"
  fi
fi

if [ "$stage" -le 3 ] && [ "$stop_stage" -ge 3 ]; then
  echo "Stage 3: Prepare manifests"
  python3 prepare_manifests.py \
    --transcript-dir "${transcript_dir}" \
    --output-dir "${manifest_dir}"
fi

if [ "$stage" -le 4 ] && [ "$stop_stage" -ge 4 ]; then
  echo "Stage 4: Fix manifests"
  mkdir -p "${fixed_manifest_dir}"
  lhotse fix "${manifest_dir}/train_recordings.jsonl.gz" "${manifest_dir}/train_supervisions.jsonl.gz" "${fixed_manifest_dir}"
  lhotse fix "${manifest_dir}/dev_recordings.jsonl.gz" "${manifest_dir}/dev_supervisions.jsonl.gz" "${fixed_manifest_dir}"
  lhotse fix "${manifest_dir}/test_recordings.jsonl.gz" "${manifest_dir}/test_supervisions.jsonl.gz" "${fixed_manifest_dir}"
fi

if [ "$stage" -le 5 ] && [ "$stop_stage" -ge 5 ]; then
  echo "Stage 5: Export text corpus"
  python3 local/export_text_corpus.py --transcript-dir "${transcript_dir}"
fi

if [ "$stage" -le 6 ] && [ "$stop_stage" -ge 6 ]; then
  echo "Stage 6: Train BPE"
  python3 local/train_bpe_model.py --vocab-size "$vocab_size"
fi

if [ "$stage" -le 7 ] && [ "$stop_stage" -ge 7 ]; then
  echo "Stage 7: Prepare minimal BPE lang dir"
  python3 local/prepare_lang_bpe.py --lang-dir "${bpe_dir}"
fi

if [ "$stage" -le 8 ] && [ "$stop_stage" -ge 8 ]; then
  echo "Stage 8: Compute fbank / cuts"
  cmd=(python3 local/compute_fbank.py --bpe-model "${bpe_dir}/bpe.model" --manifest-dir "${fixed_manifest_dir}" --output-dir "${fbank_dir}")
  if [ "$perturb_speed" = "1" ]; then
    cmd+=(--perturb-speed)
  fi
  "${cmd[@]}"
fi

if [ "$stage" -le 9 ] && [ "$stop_stage" -ge 9 ]; then
  echo "Stage 9: Compute MUSAN fbank/cuts for online CutMix (optional)"
  if [ "$enable_musan" = "1" ]; then
    python3 local/compute_fbank_musan.py \
      --manifest-dir "${musan_manifest_dir}" \
      --output-dir "${fbank_dir}"
  else
    echo "Skip online MUSAN preparation because enable_musan=0"
  fi
fi

if [ "$stage" -le 10 ] && [ "$stop_stage" -ge 10 ]; then
  echo "Stage 10: Validate cut manifests"
  python3 local/validate_manifest.py --all --manifest-dir "${fbank_dir}"
fi

if [ "$stage" -le 11 ] && [ "$stop_stage" -ge 11 ]; then
  echo "Stage 11: Display manifest statistics"
  python3 local/display_manifest_statistics.py --all --manifest-dir "${fbank_dir}"
fi

if [ "$stage" -le 12 ] && [ "$stop_stage" -ge 12 ]; then
  echo "Stage 12: Tokenize smoke test"
  python3 local/tokenize_test.py \
  --vocab-size "${vocab_size}" \
  --text "hôm nay tôi học nhận dạng tiếng nói"
fi

if [ "$stage" -le 13 ] && [ "$stop_stage" -ge 13 ]; then
  echo "Stage 13: Train"
  train_script="ASR/zipformer/train.py"
  finetune_args=()
  if [ "$do_finetune" = "1" ]; then
    if [ -z "$finetune_ckpt" ]; then
      echo "ERROR: --do_finetune 1 requires --finetune_ckpt /path/to/checkpoint.pt" >&2
      exit 1
    fi
    train_script="ASR/zipformer/finetune.py"
    finetune_args=(
      --do-finetune 1
      --finetune-ckpt "${finetune_ckpt}"
      --init-modules "${init_modules}"
    )
  fi

  python3 "${train_script}" \
    --world-size "${world_size}" \
    --num-epochs "${num_epochs}" \
    --start-epoch 1 \
    --use-fp16 "${use_fp16}" \
    --manifest-dir "${fbank_dir}" \
    --base-lr "${base_lr}" \
    --exp-dir "${exp_dir}" \
    --max-duration "${max_duration}" \
    --bpe-model "${bpe_dir}/bpe.model" \
    --enable-musan "${enable_musan}" \
    --enable-spec-aug "${enable_spec_aug}" \
    --bucketing-sampler "${bucketing_sampler}" \
    --num-buckets "${num_buckets}" \
    --num-encoder-layers "${num_encoder_layers}" \
    --feedforward-dim "${feedforward_dim}" \
    --num-heads "${num_heads}" \
    --encoder-dim "${encoder_dim}" \
    --encoder-unmasked-dim "${encoder_unmasked_dim}" \
    --decoder-dim "${decoder_dim}" \
    --joiner-dim "${joiner_dim}" \
    --use-ctc "${use_ctc}" \
    --use-cr-ctc "${use_cr_ctc}" \
    --ctc-loss-scale "${ctc_loss_scale}" \
    --cr-loss-scale "${cr_loss_scale}" \
    --attention-decoder-loss-scale "${attention_decoder_loss_scale}" \
    --causal "${causal}" \
    --chunk-size "${chunk_size}" \
    --left-context-frames "${left_context_frames}" \
    "${finetune_args[@]}"
fi

if [ "$stage" -le 14 ] && [ "$stop_stage" -ge 14 ]; then
  echo "Stage 14: Decode"
  epoch="${num_epochs}"
  if [[ "$decode_method" == "all" || "$decode_method" == "auto" || "$decode_method" == "all3" ]]; then
    decode_methods=(greedy_search modified_beam_search beam_search)
  else
    # Accept either a single method or a comma/space-separated list.
    decode_methods=()
    IFS=', ' read -r -a requested_decode_methods <<< "$decode_method"
    for method in "${requested_decode_methods[@]}"; do
      if [ -n "$method" ]; then
        decode_methods+=("$method")
      fi
    done
  fi

  if [ "${#decode_methods[@]}" -eq 0 ]; then
    echo "ERROR: no decode methods selected" >&2
    exit 1
  fi

  decode_causal_args=()
  if [ "$causal" = "1" ]; then
    decode_causal_args=(
      --causal 1
      --chunk-size "${decode_chunk_size}"
      --left-context-frames "${decode_left_context_frames}"
    )
  fi

  for method in "${decode_methods[@]}"; do
    echo "Stage 14: Decode with ${method}"
    python3 ASR/zipformer/decode.py \
      --epoch "${epoch}" \
      --avg "${avg}" \
      --use-averaged-model "${use_averaged_model}" \
      --exp-dir "${exp_dir}" \
      --manifest-dir "${fbank_dir}" \
      --bpe-model "${bpe_dir}/bpe.model" \
      --max-duration "${max_duration}" \
      --decoding-method "${method}" \
      --bucketing-sampler 0 \
      --num-encoder-layers "${num_encoder_layers}" \
      --feedforward-dim "${feedforward_dim}" \
      --num-heads "${num_heads}" \
      --encoder-dim "${encoder_dim}" \
      --encoder-unmasked-dim "${encoder_unmasked_dim}" \
      --decoder-dim "${decoder_dim}" \
      --joiner-dim "${joiner_dim}" \
      "${decode_causal_args[@]}"
  done
fi

if [ "$stage" -le 15 ] && [ "$stop_stage" -ge 15 ]; then
  if [ "$causal" != "1" ]; then
    echo "Stage 15: Streaming decode bị bỏ qua (--causal không phải 1)"
  else
    epoch="${num_epochs}"
    echo "Stage 15: Streaming decode (chunk=${decode_chunk_size}, left_context=${decode_left_context_frames}, method=${streaming_decode_method})"
    python3 ASR/zipformer/streaming_decode.py \
      --epoch "${epoch}" \
      --avg "${avg}" \
      --use-averaged-model "${use_averaged_model}" \
      --exp-dir "${exp_dir}" \
      --manifest-dir "${fbank_dir}" \
      --bpe-model "${bpe_dir}/bpe.model" \
      --max-duration "${max_duration}" \
      --decoding-method "${streaming_decode_method}" \
      --causal 1 \
      --chunk-size "${decode_chunk_size}" \
      --left-context-frames "${decode_left_context_frames}" \
      --num-encoder-layers "${num_encoder_layers}" \
      --feedforward-dim "${feedforward_dim}" \
      --num-heads "${num_heads}" \
      --encoder-dim "${encoder_dim}" \
      --encoder-unmasked-dim "${encoder_unmasked_dim}" \
      --decoder-dim "${decoder_dim}" \
      --joiner-dim "${joiner_dim}"
  fi
fi
