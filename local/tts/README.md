# TTS clone generation

Scripts that build the synthetic half of the training set and the held-out
speaker eval set. Every number in `RESULTS.md` depends on these.

## External dependency

These need **Gwen-TTS** (`ggroup-ai-lab/gwen-tts`, Qwen3-TTS-0.6B finetuned on
1000 h Vietnamese), which is *not* vendored here — it carries model weights and
its own pinned dependency tree.

```bash
git clone https://github.com/ggroup-ai-lab/gwen-tts
cd gwen-tts
# qwen-tts==0.1.1 pins transformers==4.57.3, so keep it in its own venv:
python3 -m venv .venv-gwen
.venv-gwen/bin/pip install qwen-tts==0.1.1 soundfile numpy torch
```

Scripts locate it as a sibling of the icefall checkout. Override with
`GWEN_TTS_DIR=/path/to/gwen-tts`. They read `$GWEN_TTS_DIR/data/ref_info.json`
for the built-in reference voices.

Run them with the Gwen venv's python, not the icefall one.

## The scripts

| script | output | used for |
| --- | --- | --- |
| `build_crossspeaker.py` | `transcripts_crossspk/`, `audio_crossspk/` | 3,200 clones in the 5 project voices (all male) + the 800 real originals |
| `build_diverse_clones.py` | `transcripts_crossspk_diverse/`, `audio_crossspk_diverse/` | 3,200 clones in 3 female + 1 male voice |
| `make_heldout_speaker_eval.py` | `heldout_speaker_eval/` | 25 sentences × 2 unseen voices — the honest benchmark |

```bash
GWEN=/path/to/gwen-tts
$GWEN/.venv-gwen/bin/python local/tts/build_crossspeaker.py   --batch-size 8 --max-duration 20.0
$GWEN/.venv-gwen/bin/python local/tts/build_diverse_clones.py --batch-size 8 --max-duration 20.0
$GWEN/.venv-gwen/bin/python local/tts/make_heldout_speaker_eval.py
```

All three are resumable — they skip clips already on disk.

## Voice split — do not violate

`yen_nhi` (♀) and `khanh_toan` (♂) are **held out of training** and used only by
`make_heldout_speaker_eval.py`. Cloning a training sentence into either voice
turns the honest benchmark into another memorization test.

`build_diverse_clones.py:TRAIN_VOICES` excludes them deliberately.

## Reference-audio quality

`prepare_ref()` trims leading/trailing silence and peak-normalizes each
reference before cloning; outputs are loudness-normalized to RMS 0.09.

Two failure modes worth knowing:

- **Repeated words** in the output mean `ref_text == gen_text`. Use a reference
  clip whose text differs from what you are generating.
- **A quiet or clipped reference propagates.** Dung's original recordings peaked
  at 0.059 and cloned to a near-inaudible voice; Trung's had leading silence and
  clipping. Both were fixed by taking references from `datasets/vi_asr_corpus/`
  rather than `dataset/`.
