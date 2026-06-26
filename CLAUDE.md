# CLAUDE.md — VietnameseASR

## Project Overview

Real Vietnamese ASR with icefall (streaming Zipformer2 + Pruned Transducer).
Same recipe code as `../vi_asr_corpus` (a toy 3-sentence dataset); the two must
stay code-identical — only the dataset differs. A pre-commit guard enforces this
(`local/check_recipe_sync.sh`).

- **Dataset**: VietnameseASR — diverse Vietnamese recordings, matched "học vẹt"
  train/dev/test splits, x10-repeated, utterances >20s filtered.
- **Model**: small streaming/causal Zipformer, BPE vocab 100.
- **Final model**: `ASR/zipformer/exp_bpe100_small_streaming_raw_x10_matched_u20_streaming_20260623_004646`,
  epoch 55, avg 10 (chunk 32, left-context 256).
- **Results**: local WER 9.79% (modified_beam_search); Jetson CUDA int8 full-set
  WER 9.17%, RTF 0.166.
- **Deploy**: ONNX package for Jetson Nano under `deploy/jetson_nano/`.

## Where things are documented

- `README.md` — recipe usage, `run_x10.sh`/`run.sh` stages and flags, export.
- `PROJECT_NOTES.md` — goals, historical results, Jetson/TensorRT notes.
- `deploy/jetson_nano/README.md` — Jetson deployment and evaluation.

## Conventions worth knowing

- Active pipeline script is `run_x10.sh` (matched x10 splits by default; pass
  `--matched_splits 0` for the plain layout). `run_robust.sh` and
  `run_matched_asr.sh` are thin wrappers over `run.sh`.
- Feature extraction uses **lhotse Fbank** (not `torchaudio.compliance.kaldi`)
  so training and inference features match.
- Select checkpoints by decode WER, not validation loss; keep `avg <= 10`.
- Streaming decode requires `--causal 1`.
- Jetson GPU path is the GPU-built C++ sherpa-onnx binary (provider=cuda,
  int8); the pip `sherpa_onnx` python wheel there is CPU-only.
