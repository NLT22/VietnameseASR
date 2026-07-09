# CLAUDE.md — VietnameseASR

## Project Overview

Real Vietnamese ASR with icefall (streaming Zipformer2 + Pruned Transducer).
Single recipe. The former sibling clone `vi_asr_corpus` has been merged in; its
data and models are archived under `datasets/vi_asr_corpus/`.

- **Dataset**: 800 real utterances (4 male speakers × 200, 4–5 s), plus 6,400
  Gwen-TTS clones across 5 more voices (3 female). Matched "học vẹt" splits
  (train == dev == test recordings).
- **Model**: small streaming/causal Zipformer, BPE vocab 100, chunk 32,
  left-context 256.
- **Current model**: `ASR/zipformer/exp_bpe100_small_streaming_divmix_x8`,
  epoch 60, avg 10.
- **Deploy**: `deploy/jetson_nano/model_divmix_x8_epoch60_avg10/` (int8 ONNX).

**All numbers live in `RESULTS.md`.** Do not duplicate them here — a stale copy
in this file outlived the model it described by two months.

## Read this before trusting any old result

Everything trained before 2026-07-09 used **corrupted transcripts**: an
off-by-one bug paired every Trung/Dung recording with the previous sentence's
text. Those experiments and their WERs were deleted, not archived. See
`RESULTS.md` → "The off-by-one bug".

The memorization benchmark (train == test) **cannot detect this**, and cannot
detect speaker generalization either. `eval_heldout_speaker.py` is the metric
that predicts whether the mic UI works for a real person.

## Conventions worth knowing

- `run.sh` is the single pipeline entry point (`bash run.sh --help`). Pass
  `--data_tag NAME` to run a pre-built transcript version
  (`transcripts_NAME/` → `fbank_NAME/`, `data/manifests_NAME/`, exp suffix
  `_NAME`). `run_x10.sh` is a thin backward-compatible wrapper.
  **`run.sh` has no export stage** — use `local/export_for_jetson.sh`.
- Data-prep scripts (`prepare_vi_asr_corpus.py`, `prepare_matched_splits.py`,
  `prepare_manifests.py`, `audit_dataset.py`, `declip.py`) live under `local/`,
  following the librispeech convention.
- **After any dataset ingest, verify `corr(audio_duration, text_word_count)`
  per speaker.** Correct alignment is clearly positive; a shifted set collapses
  toward 0. This is the check that would have caught the off-by-one bug.
- Feature extraction uses **lhotse Fbank** (not `torchaudio.compliance.kaldi`)
  so training and inference features match. Inference must scale waveforms to
  `[-1, 1]`.
- Select checkpoints by decode WER, not validation loss; keep `avg <= 10`.
- Streaming decode requires `--causal 1`.
- Decode with `deploy/jetson_nano/transcribe_beam_wav.py` (classic
  `beam_search`, numpy + onnxruntime). sherpa-onnx implements only
  `greedy_search` / `modified_beam_search`, which emit ≤1 symbol per encoder
  frame and score ~4× worse on this model.

## Gotchas that have bitten before

- `pkill -f PATTERN` / `pgrep -f PATTERN` **match their own command line**. A
  monitor that greps for its own target silently kills itself or never fires.
  Grep log files, or use a bracket class (`serve[r].py`).
- Jetson: `export OPENBLAS_CORETYPE=ARMV8` or every numpy import dies with
  "Illegal instruction". `.bashrc` sets it, but not for non-interactive SSH.
- `/tmp` is wiped on reboot. Keep training logs on the project disk.
