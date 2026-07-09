# VietnameseASR — Results

**Last updated:** 2026-07-10
**Framework:** icefall · streaming Zipformer2 + Pruned RNN-T Transducer
**BPE vocab:** 100 · **Chunk:** 32 · **Left context:** 256 · **Causal:** yes

> All results predating **2026-07-09** have been deleted, not archived. They were
> produced by models trained on corrupted transcripts (see
> [The off-by-one bug](#the-off-by-one-bug)) and are not comparable to anything
> here. Do not resurrect them from git history.

---

## Current model

`model_divmix_x8_epoch60_avg10` — exported to `deploy/jetson_nano/`.

Experiment: `ASR/zipformer/exp_bpe100_small_streaming_divmix_x8`, epoch 60, avg 10.

### Memorization set (train == dev == test, 800 real recordings)

This is a **"học vẹt" benchmark by design**: the model is scored on the exact
recordings it trained on. It measures recall of specific recordings.

| decoder | WER |
| --- | ---: |
| `greedy_search` | 4.36% |
| `modified_beam_search` (beam 4) | 3.31% |
| **`beam_search` (beam 4)** | **0.72%** |

### Held-out speakers — the honest benchmark

25 known sentences × 2 voices **never seen in training** (`yen_nhi` ♀,
`khanh_toan` ♂), synthesized with Gwen-TTS. 578 reference words.
Scored with `eval_heldout_speaker.py`, classic `beam_search`, beam 4.

| model | khanh_toan ♂ | yen_nhi ♀ | overall |
| --- | ---: | ---: | ---: |
| `model_streaming_u20_epoch55_avg10` (real only, **pre-fix**) | 82.70% | 99.31% | 91.00% |
| `model_realdom_mix_epoch60_avg10` (5 male clones, **pre-fix**) | 49.48% | 92.04% | 70.76% |
| **`model_divmix_x8_epoch60_avg10`** | **1.38%** | **16.96%** | **9.17%** |

Adding female voices to training moved `yen_nhi` from 92.04% → 16.96%. All five
original project speakers are male, which is why female voices collapsed.

> **Confound, stated plainly.** The `divmix_x8` run changed *two* variables at
> once: the label fix and the female clones. The `yen_nhi`-specific collapse
> points at the clones (the corrupted labels were in Trung and Dung, both male,
> and would not selectively rescue a female voice); the `modified_beam_search`
> jump 9.79% → 3.31% points at the label fix. Both are inference, not
> measurement. A clean split needs corrected-labels + male-only-clones as a
> control. **That experiment has not been run.**

### On-device (Jetson Nano, int8, CPU, classic beam_search)

| clip | RTF |
| --- | ---: |
| 2.9 s | 0.30 |
| 3.8 s | 0.26 |
| 6.8 s | 0.27 |

Spot-check on real recordings, new vs old model:

| utterance | pre-fix deployed model | `divmix_x8` |
| --- | --- | --- |
| Dung — *"đèn giao thông chuyển sang màu đỏ…"* | *"để dây tôi triển sau bảo đêm…"* ✗ | ✓ |
| Trung — *"cầu vồng bắc ngang bầu trời…"* | *"tôi lỗi tript"* ✗ | ✓ (1 word) |
| Khoi — *"bạn có thể giúp tôi kiểm tra…"* | ✓ | ✓ |

The old model fails on **exactly** Dung and Trung — the two speakers whose labels
were shifted — and handles Khoi correctly. Khoi is the control.

---

## The off-by-one bug

Windows Voice Recorder names the **first** take `Recording.wav` and later takes
`Recording (2).wav`, `(3)`, … Natural sort places the unnumbered file **last**,
so `list_audio_files()` paired script line 1 with `Recording (2)`, line 2 with
`Recording (3)`, …, and line 200 with `Recording.wav`.

**Every Trung and Dung utterance was trained against the previous sentence's
text** — 330 of 730 real recordings at the time. Quan (`Recording (1..200)`) and
Khoi (`001..200.wav`) have no unnumbered file and were never affected.

### How it was found

Longer sentences take longer to say, so `corr(audio_duration, text_word_count)`
discriminates the correct pairing. Quan, having no bare file, is identical under
both orderings — the built-in control that validates the test.

| speaker | natural sort | bare-first | verdict |
| --- | ---: | ---: | --- |
| Quan (control) | +0.472 | +0.472 | unaffected |
| Trung | +0.858 | **+0.954** | shifted |
| Dung | −0.013 | **+0.216** | shifted |

Fixed by `recorder_key()` in `local/prepare_vi_asr_corpus.py`: sort on
`(int of "(N)" or 1, natural_key)`.

### Why it survived so long

Memorization WER is **blind to label corruption**. The model learns each
recording's wrong label consistently and still scores ~0.5%. The bug only
surfaces when a human speaks a sentence and gets a *different* sentence back —
which is exactly the symptom that was reported and initially misdiagnosed as a
speaker-generalization problem.

**Rule:** after any dataset ingest, check `corr(duration, word_count)` per
speaker. Correct alignment is clearly positive (~+0.5…+0.95); a shifted set
collapses toward 0.

---

## Corrections to earlier claims

Recorded because they were stated confidently and were wrong.

**"sherpa-onnx has a hard ~9.2% structural floor."** Overstated. Widening
`max_active_paths` does saturate (9.36% → 9.19% for beam 4 → 32), but most of the
9.2% was the label bug, not the decoder. After the fix the same
`modified_beam_search` scores **3.31%**. The *gap* between decoders is real
(3.31% vs 0.72% on identical weights) because sherpa's algorithms emit ≤1 symbol
per encoder frame while this model is trained to emit several — but its size was
inflated by bad labels. **Measure both decoders on every new model.**

**"beam_search is too slow to deploy."** Wrong. That was measured on icefall's
PyTorch path (2.6 s/utt). The ONNX implementation runs at 0.35 s/utt on PC and
RTF ~0.27 on the Nano — *faster* than the sherpa-onnx path it replaced.

**"The 0.52% → 2.37% gap is int8 quantization."** Wrong. fp32 gives identical
error counts. int8 costs nothing.

**"The cross-speaker clones didn't help — they slightly hurt."** Wrong; that was
measured on the memorization benchmark, which cannot see speaker generalization.
On held-out speakers the clones cut WER 91.00% → 70.76%.

---

## Known limitations

**Long utterances are out of distribution.** The re-recorded sentences are 4–5 s,
so `divmix_x8` (12,800 cuts) contains only 72 cuts ≥ 15 s and exactly one ≥ 19 s.
A 19 s clip decodes to an **empty string**. Fine for spoken sentences; unusable
for long-form audio. The superseded models handled 20 s clips because their
training data was ~20 s median.

**`yen_nhi` at 16.96% carries almost all remaining held-out error** (49 of 53
words). One female voice in training took her from 92% to 17%; more female
reference voices are the obvious next lever.

**Non-streaming is not deployable with the good decoder.** The non-streaming
encoder takes 2 inputs (`x`, `x_lens`) and carries no chunk metadata, while
`jetson_beam_decode.py` reads `T`/`decode_chunk_len` and builds 74 cached state
tensors. Porting is small but unwritten. Without it, non-streaming falls back to
sherpa-onnx and its weaker decoders. TensorRT rejects non-streaming outright
(`Tensorrt support for Online models only`).

**fp16 ONNX will not load** — mixed fp16/fp32 `Cast` nodes inside the Zipformer
graph. int8 or fp32 only.

---

## Dataset

| | count | note |
| --- | ---: | --- |
| Real recordings | 800 | 4 speakers × 200, all male, 4–5 s |
| Cross-speaker clones | 3,200 | Gwen-TTS: Dung, Khoi, Quan, Trung, Hieu (male) |
| Diverse clones | 3,200 | Gwen-TTS: my_van, ai_vy, dieu_linh (female) + tran_lam (male) |
| **`divmix_x8` train** | **12,800** | real ×8 (50%) + 6,400 clones, **9 voices** |
| dev / test | 800 each | the clean real originals (memorization by design) |

Held out from training entirely: `yen_nhi` (♀), `khanh_toan` (♂).

Augmentation: online MUSAN + SpecAugment. No speed perturbation (it triples cuts
for variety the 9 voices already supply).

Trung's re-recorded audio was **hard-clipped in all 200 files** (3.4% of samples
at full scale). Repaired with `local/declip.py` — cubic-Hermite reconstruction of
each clipped run from the clean samples bracketing it. All 77,827 runs repaired,
none refused. Gain reduction cannot fix clipping: the samples above ±1.0 were
never stored. The repair is only valid because every run was shorter than a
quarter pitch period; the script refuses longer runs rather than fabricate.

---

## Reproducing

```bash
# rebuild corrected splits from dataset/
bash run.sh --stage -1 --stop_stage -1

# train + decode
bash run.sh --data_tag divmix_x8 --exp_suffix "" \
  --vocab_size 100 --model_size small --causal 1 \
  --use_fp16 1 --max_duration 700 --num_workers 4 \
  --num_epochs 60 --use_averaged_model 1 --avg 10 \
  --enable_musan 1 --enable_spec_aug 1 --perturb_speed 0 \
  --stage 3 --stop_stage 14

# export ONNX (run.sh has no export stage)
bash local/export_for_jetson.sh \
  --exp-dir ASR/zipformer/exp_bpe100_small_streaming_divmix_x8 \
  --epoch 60 --avg 10 --streaming 1 --use-averaged-model 1

# the benchmark that matters
python3 eval_heldout_speaker.py \
  --model-dir deploy/jetson_nano/model_divmix_x8_epoch60_avg10
```

Clone generation lives in the sibling `gwen-tts/` workspace
(`build_crossspeaker.py`, `build_diverse_clones.py`, `make_heldout_speaker_eval.py`).
