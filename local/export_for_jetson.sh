#!/usr/bin/env bash
set -euo pipefail

stage=0
stop_stage=2
exp_dir=""
epoch=""
avg=1
streaming=0
out_dir=""
use_averaged_model=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --stage) stage="$2"; shift 2 ;;
    --stop_stage|--stop-stage) stop_stage="$2"; shift 2 ;;
    --exp_dir|--exp-dir) exp_dir="$2"; shift 2 ;;
    --epoch) epoch="$2"; shift 2 ;;
    --avg) avg="$2"; shift 2 ;;
    --streaming) streaming="$2"; shift 2 ;;
    --out_dir|--out-dir) out_dir="$2"; shift 2 ;;
    --use_averaged_model|--use-averaged-model) use_averaged_model="$2"; shift 2 ;;
    *) echo "Unknown option: $1" >&2; exit 1 ;;
  esac
done

if [[ -z "$exp_dir" || -z "$epoch" ]]; then
  echo "Usage: $0 --exp-dir EXP --epoch N [--avg M] [--streaming 0|1] [--out-dir DIR]" >&2
  exit 1
fi

recipe_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$recipe_dir"

if [[ -z "$out_dir" ]]; then
  name="$(basename "$exp_dir")"
  out_dir="$PWD/deploy/jetson_nano/model_${name}_epoch${epoch}_avg${avg}"
fi

mkdir -p "$out_dir"

common=(
  --epoch "$epoch"
  --avg "$avg"
  --use-averaged-model "$use_averaged_model"
  --exp-dir "$exp_dir"
  --tokens data/lang_bpe_100/tokens.txt
  --num-encoder-layers 2,2,2,2,2,2
  --feedforward-dim 512,768,768,768,768,768
  --num-heads 4,4,4,8,4,4
  --encoder-dim 192,256,256,256,256,256
  --encoder-unmasked-dim 192,192,192,192,192,192
  --decoder-dim 512
  --joiner-dim 512
)

if [[ "$stage" -le 0 && "$stop_stage" -ge 0 ]]; then
  if [[ "$streaming" == "1" ]]; then
    python3 ASR/zipformer/export-onnx-streaming.py \
      "${common[@]}" \
      --causal 1 \
      --chunk-size 32 \
      --left-context-frames 256
  else
    python3 ASR/zipformer/export-onnx.py "${common[@]}"
  fi
fi

if [[ "$stage" -le 1 && "$stop_stage" -ge 1 ]]; then
  suffix="epoch-${epoch}-avg-${avg}"
  if [[ "$streaming" == "1" ]]; then
    suffix="${suffix}-chunk-32-left-256"
  fi

  cp "$exp_dir/encoder-${suffix}.onnx" "$out_dir/encoder.onnx"
  cp "$exp_dir/decoder-${suffix}.onnx" "$out_dir/decoder.onnx"
  cp "$exp_dir/joiner-${suffix}.onnx" "$out_dir/joiner.onnx"
  if [[ -f "$exp_dir/encoder-${suffix}.int8.onnx" ]]; then
    cp "$exp_dir/encoder-${suffix}.int8.onnx" "$out_dir/encoder.int8.onnx"
    cp "$exp_dir/decoder-${suffix}.int8.onnx" "$out_dir/decoder.int8.onnx"
    cp "$exp_dir/joiner-${suffix}.int8.onnx" "$out_dir/joiner.int8.onnx"
  fi
  cp data/lang_bpe_100/tokens.txt data/lang_bpe_100/bpe.model data/lang_bpe_100/bpe.vocab "$out_dir"/
fi

if [[ "$stage" -le 2 && "$stop_stage" -ge 2 ]]; then
  cat > "$out_dir/MODEL_INFO.txt" <<EOF
Experiment: $exp_dir
Epoch: $epoch
Avg: $avg
Streaming: $streaming
Runtime: sherpa-onnx ONNX
TensorRT: validate on Jetson; non-streaming is known unsupported in current sherpa-onnx path
EOF
fi

echo "Wrote Jetson model package: $out_dir"
