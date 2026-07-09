# Vietnamese ASR Project Notes

Goals, design decisions, and hard-won environment knowledge.
**All measured results live in `RESULTS.md`.** Nothing is duplicated here — a
stale copy of the numbers in this file survived two months past the model it
described.

## Target

- Real ASR model, not template matching.
- Small streaming Zipformer transducer, BPE vocab 100.
- Matched "học vẹt" setup: train/dev/test intentionally contain the same
  original recordings, for overfit evaluation.
- Deploy to Jetson Nano as int8 ONNX.
- Select checkpoints by decode WER, not validation loss alone.

## Two benchmarks, and why both exist

**Memorization WER** (`transcripts_matched_u20`, train == dev == test) measures
recall of specific recordings. It is the project's original goal: speak a
sentence that exists in the dataset, get it back.

It is **blind** to two things that matter:

1. *Label corruption* — a model happily memorizes wrong labels and still scores
   ~0.5%. This hid the off-by-one bug for months (see `RESULTS.md`).
2. *Speaker generalization* — whether the mic UI works for a person whose voice
   is not in training.

**`eval_heldout_speaker.py`** covers both: known sentences, unseen voices. It is
the number to quote when someone asks "does this work".

## Design decisions

- **Clones over offline MUSAN.** Cross-speaker TTS clones add speaker variety,
  which is the axis the model actually lacked. Offline MUSAN copies only add
  noise robustness and multiply disk 5–10×.
- **Synthetic ratio matters.** ~80% synthetic produced degenerate blank-heavy
  decodes (27–48% WER). 26–50% synthetic trains healthily.
- **All original speakers are male.** Female voices collapsed until female
  clones entered training. This was invisible on the memorization benchmark.
- **No speed perturbation.** It triples fbank cuts for variety that 9 distinct
  voices already supply.
- **Online MUSAN + SpecAugment**, not offline copies.

## Environment knowledge

### Jetson Nano (JetPack 4 / L4T R32.7.6, Python 3.6.9, CUDA 10)

- `onnxruntime==1.10.0`, `numpy==1.19.5` (apt's 1.13.3 lacks
  `numpy.core._multiarray_umath` → segfault), `kaldi-native-fbank` from source.
- **`export OPENBLAS_CORETYPE=ARMV8`** or any numpy import dies with "Illegal
  instruction". `~/.bashrc` sets it — but `.bashrc` early-returns for
  non-interactive shells, so SSH commands must export it explicitly.
- The pip `sherpa_onnx` wheel is **CPU-only** and *silently* falls back when
  given `provider="cuda"`. GPU needs a source build with
  `-DSHERPA_ONNX_ENABLE_GPU=ON`.
- GPU is **not** worth it for single-utterance UI: CPU ~7 s/clip vs GPU ~10 s
  warm, ~30 s cold (CUDA init per call). GPU only wins on sustained batch eval.
- TensorRT: non-streaming rejected outright; streaming fp32 never finishes its
  engine build in 300 s; streaming int8 aborts on a missing shape for a
  quantized MatMul. **No working TensorRT path.**
- fp16 ONNX will not load — mixed fp16/fp32 `Cast` nodes. int8 or fp32 only.
  (Training's `--use-fp16` autocast and the exporter's `--fp16` are unrelated.)

### Feature parity

Training uses lhotse `Fbank`. Inference must reproduce it: povey window,
`dither=0`, `snip_edges=False`, `preemph=0.97`, `low_freq=20`, `high_freq=-400`,
80 mels, 25/10 ms — and the waveform scaled to **`[-1, 1]`**. Kaldi's native
`[-32768, 32767]` shifts every mel bin by `ln(32768²) ≈ 20.8`.

### Gwen-TTS clone generation

`ggroup-ai-lab/gwen-tts` (Qwen3-TTS-0.6B finetuned on 1000 h Vietnamese).
`qwen-tts==0.1.1` pins `transformers==4.57.3`, so it lives in an isolated venv
(`gwen-tts/.venv-gwen`). References need silence trimming + peak normalization;
`ref_text == gen_text` causes repeated words.

## Tooling gotchas

- `pkill -f PATTERN` and `pgrep -f PATTERN` **match their own command line**.
  A background monitor grepping for its own target either kills itself or never
  fires. Grep log files, or use a bracket class (`serve[r].py`).
- `pkill` returning 1 (no match) aborts a compound command under `set -e`.
- `/tmp` is wiped on reboot; keep training logs on the project disk.
- Power loss truncates the in-flight checkpoint to 0 bytes. Delete it and resume
  with `--start_epoch N` (which loads `epoch-(N-1).pt`); everything else
  survives.
- `run.sh` has **no export stage** — stages end at 15 (streaming decode).
  Export is `local/export_for_jetson.sh`.
