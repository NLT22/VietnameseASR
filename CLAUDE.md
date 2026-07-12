# CLAUDE.md — VietnameseASR

## Project Overview

Real Vietnamese ASR with icefall (streaming Zipformer2 + Pruned Transducer).
Single recipe. The former sibling clone `vi_asr_corpus` has been merged in; its
data and models are archived under `datasets/vi_asr_corpus/`.

- **Dataset**: 800 real utterances (4 male speakers × 200, 4–5 s), plus 6,400
  Gwen-TTS clones across 5 more voices (3 female). Matched "học vẹt" splits
  (train == dev == test recordings).
- **Model**: medium (M) streaming/causal Zipformer, BPE vocab 100, chunk 32,
  left-context 256. (small variant kept as backup.)
- **Current model**: `ASR/zipformer/exp_bpe100_medium_streaming_main_lr0045`,
  epoch 30, avg 10. Real 1.84% / held-out 1.80%; beats the small Hieu model
  (2.72% / 2.28%).
- **Deploy**: `deploy/jetson_nano/model_medium_epoch30_avg10/` (int8
  ONNX). Live on Jetson (`~/vasr/model` -> `model_medium`) and both live UIs
  (port 8000 public tunnel, 8100 speaker-id). Backup:
  `model_small_epoch50_avg10` (small).
- Only these two 5-speaker (`main`) experiments are kept; older ones were
  archived + deleted 2026-07-12 (see `docs/ARCHIVED_EXPERIMENTS.md`).
- `run.sh` embeds `--model_size` in the exp dir name (`exp_bpeNNN_medium_...` /
  `_small_...`), and `export_for_jetson.sh` auto-detects the encoder dims from it
  (`_medium_` / `_small_`). No symlink dance needed.

**All numbers live in `docs/RESULTS.md`.** `docs/TEACHING_NOTES.md` (EN) and
`docs/TEACHING_NOTES_VI.md` (VI) explain how the model, pipeline, config parameters
and deployment actually work, grounded in the code. Do not duplicate them here — a stale copy
in this file outlived the model it described by two months.

## Read this before trusting any old result

Everything trained before 2026-07-09 used **corrupted transcripts**: an
off-by-one bug paired every Trung/Dung recording with the previous sentence's
text. Those experiments and their WERs were deleted, not archived. See
`docs/RESULTS.md` → "The off-by-one bug".

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
  `prepare_manifests.py`, `audit_dataset.py`) live under `local/`,
  following the librispeech convention.
- **After any dataset ingest, verify `corr(audio_duration, text_word_count)`
  per speaker.** Correct alignment is clearly positive; a shifted set collapses
  toward 0. This is the check that would have caught the off-by-one bug.
- Feature extraction uses **lhotse Fbank** (not `torchaudio.compliance.kaldi`)
  so training and inference features match. Inference must scale waveforms to
  `[-1, 1]`.
- Select checkpoints by decode WER, not validation loss (dev == train here).
  `--avg` was swept: 1-15 are indistinguishable on unseen voices; heavier
  averaging only fits the memorization set harder. Deployed at `avg=10`.
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
