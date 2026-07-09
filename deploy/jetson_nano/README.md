# Jetson Nano Deployment

Self-contained offline deployment package for the VietnameseASR streaming
Zipformer transducer. This document explains **how the model gets from a
PyTorch checkpoint onto the Nano**, **what runs it**, and **every limitation we
actually hit** — framework, library, and environment.

---

## 1. The deployment path

```
  PyTorch checkpoint            (ASR/zipformer/exp_*/epoch-N.pt)
        │
        │  export-onnx-streaming.py  (icefall)   +  avg-N checkpoint averaging
        ▼
  3 ONNX graphs                 encoder.onnx / decoder.onnx / joiner.onnx   (fp32)
        │
        │  onnxruntime.quantization  (dynamic int8)
        ▼
  encoder.int8.onnx  decoder.int8.onnx  joiner.int8.onnx
        │
        │  scp the whole model_*/ dir  (+ tokens.txt, bpe.model)
        ▼
  Jetson Nano — onnxruntime CPU  +  our own beam_search decoder
```

A transducer is **three separate graphs**, not one:

| graph | input | output | role |
| --- | --- | --- | --- |
| **encoder** | `x[1,77,80]` fbank + 74 cached state tensors | `enc[1,64,512]` + 74 new states | listens to audio |
| **decoder** | `y[N,2]` (last 2 token ids) | `[N,512]` | language-model-ish context |
| **joiner** | `enc[N,512] + dec[N,512]` | `logit[N,100]` | scores next token |

`blank_id=0`, `context_size=2`, vocab 100. The encoder is **streaming/causal**:
it consumes a 77-frame segment, advances by `decode_chunk_len=64`, and carries
74 state tensors between chunks. Those numbers are read from the ONNX metadata
at load time (`jetson_beam_decode.py`), never hardcoded.

Model config: small causal Zipformer2, chunk 32, left-context 256, BPE vocab 100.

---

## 2. Two runtimes — and why we stopped using the obvious one

### sherpa-onnx (`transcribe_streaming_wav.py`, `transcribe_wav.py`)

The standard k2 deployment runtime. Easy, C++/pip, handles streaming state.

**Hard limitation: sherpa-onnx implements only `greedy_search` and
`modified_beam_search`.** Both emit **at most one symbol per encoder frame**.
Our model is trained to emit *multiple* symbols per frame, so decoding is
starved of output slots and produces mass deletions.

This is not a tuning problem — widening the beam saturates:

| `max_active_paths` | 4 | 8 | 16 | 32 |
| --- | ---: | ---: | ---: | ---: |
| WER (old model) | 9.36% | 9.28% | 9.22% | 9.19% |

Widening the beam cannot buy back a symbol the algorithm is unable to emit.

> **Correction (2026-07-09).** We previously called ~9.2% a hard structural
> floor. That was wrong. Much of it was the off-by-one label bug, not the
> decoder: after the fix, the same `modified_beam_search` scores **3.31%**
> (`divmix_x8`). The *gap* between the two decoders is real and structural
> (3.31% vs 0.72% on the same model), but its size was inflated by bad labels.
> Measure both decoders on any new model rather than assuming the old ratio.

### Our decoder (`jetson_beam_decode.py`, `transcribe_beam_wav.py`)

**numpy + onnxruntime only.** No sherpa-onnx, no torch, no k2. Implements
classic `beam_search` (Graves 2012, Alg. 1), which *can* emit multiple symbols
per frame, with decoder/joiner caching so it stays fast.

| decoder | WER (618-utt memorization set) | errors |
| --- | ---: | --- |
| sherpa-onnx `modified_beam_search` | 9.19–9.79% | **862 deletions** |
| our `beam_search` | **2.37%** | 47 deletions |

Same model. Same weights. The ~7-point gap is almost entirely deletions the
sherpa algorithm cannot emit.

`transcribe_beam_wav.py` is a drop-in replacement for
`transcribe_streaming_wav.py` — same CLI, and it accepts-and-ignores
`--threads/--provider/--max-active-paths/--fp32` so callers don't break.

> **Nothing forced us onto sherpa-onnx.** It was convenience. The ONNX graphs
> are plain onnxruntime graphs and anything can drive them.

---

## 3. Precision: int8 works, fp32 works, fp16 does not

| precision | status | notes |
| --- | --- | --- |
| **int8** | ✅ default | 35% faster, 3× smaller, **bit-identical error counts to fp32** |
| fp32 | ✅ works | reference; no accuracy gain over int8 here |
| **fp16** | ❌ **won't load** | see below |

int8 vs fp32 was measured, not assumed: identical error counts on the full set.
So int8 costs nothing and is the default.

**Why fp16 fails.** Training's `--use-fp16` (mixed-precision autocast) and the
exporter's `--fp16` (ONNX weight conversion) are *different things*. Converting
this Zipformer graph to fp16 leaves mixed fp16/fp32 `Cast` nodes, and ONNX
Runtime rejects it at session init:

```
Type Error: Type (tensor(float16)) of output arg
(/feed_forward1/out_proj/Cast_output_0) does not match expected type (tensor(float))
```

fp16 on the Nano would need a separate CUDA/TensorRT conversion + validation
path. **Supported here: int8 or fp32.**

---

## 4. Environment: JetPack 4 is the constraint

Jetson Nano (reflashed): **JetPack 4 / L4T R32.7.6, Ubuntu 18.04, Python 3.6.9,
CUDA 10**. This is an old, frozen stack and it dictates every pin.

| thing | requirement | what breaks otherwise |
| --- | --- | --- |
| `onnxruntime` | **1.10.0** | newer wheels need a glibc/CUDA the Nano doesn't have |
| `numpy` | **1.19.5** | apt's 1.13.3 lacks `numpy.core._multiarray_umath` → **segfault** |
| `kaldi-native-fbank` | source build | no aarch64 wheel for py3.6 |
| **`OPENBLAS_CORETYPE=ARMV8`** | **must be exported** | numpy's OpenBLAS misdetects the CPU → **"Illegal instruction"** |

`OPENBLAS_CORETYPE=ARMV8` is persisted in `~/.bashrc` on the Nano. Forget it and
*any* numpy import dies with an illegal instruction and no useful message.

`install_jetson.sh` prefers `python3.8` from apt when available — on Python 3.6,
`sherpa-onnx` has no aarch64 wheel and compiles from source, which is very slow.
Our beam decoder needs only numpy + onnxruntime, so this matters less now.

### GPU on the Nano

- The **pip `sherpa_onnx` wheel is CPU-only.** It ships no
  `libonnxruntime_providers_cuda.so` and **silently falls back to CPU** when you
  pass `provider="cuda"` — it does not warn. GPU requires a source build with
  `-DSHERPA_ONNX_ENABLE_GPU=ON`.
- **GPU is not worth it for the live UI.** Measured: CPU ≈ 7 s/clip; GPU ≈ 10 s
  warm and ~30 s cold (CUDA init per call). The GPU only wins on sustained batch
  eval, never on single utterances.

### TensorRT status

Built sherpa-onnx from source against ONNX Runtime GPU 1.11.0. `csrc/session.cc`
needed a Nano patch to drop provider options ORT 1.11 doesn't know
(`trt_timing_cache_enable`, `trt_timing_cache_path`, `trt_detailed_build_log`,
`trt_dump_subgraphs`). Outcome:

- non-streaming → rejected: `Tensorrt support for Online models only`
- streaming fp32 → engine build/decode did not finish within 300 s
- streaming int8 → aborts: `TensorRT input: /out/MatMul_output_0_output_quantized has no shape specified`

**There is no working TensorRT package.** int8 ONNX on CPU is the shipping path.

---

## 5. Feature extraction must match training exactly

Training uses **lhotse `Fbank`**, so inference must reproduce it bit-for-bit or
WER collapses. `jetson_asr.py:compute_fbank()` mirrors lhotse's defaults via
`kaldi_native_fbank`: povey window, `dither=0`, `snip_edges=False`,
`preemph=0.97`, `low_freq=20`, `high_freq=-400`, 80 mel bins, 25 ms / 10 ms.

**The waveform must be scaled to `[-1, 1]`** (`pcm.astype(np.float32) / 32768.0`).
Kaldi's native `[-32768, 32767]` convention shifts every filterbank bin by
`ln(32768²) ≈ 20.8` — the model then sees features it has never seen.

---

## 6. Running it

Copy this folder to the Nano, then:

```bash
cd jetson_nano
bash install_jetson.sh
export OPENBLAS_CORETYPE=ARMV8        # or rely on ~/.bashrc
```

Transcribe a wav (any rate/channels — it resamples):

```bash
python3 transcribe_beam_wav.py --model-dir model_realdom_mix_epoch60_avg10 audio.wav
```

The transcript is printed as the **last stdout line** (so callers can `tail -1`).

Legacy sherpa-onnx path, for comparison only:

```bash
python3 transcribe_streaming_wav.py --model-dir model_streaming_u20_epoch55_avg10 audio.wav
```

### Live mic UI

`remote_mic_ui/` serves a browser mic recorder with a **model selector**.
`--asr-mode` picks the backend script:

| `asrMode` | script | decoder |
| --- | --- | --- |
| `beam` | `transcribe_beam_wav.py` | our classic beam_search |
| `streaming` | `transcribe_streaming_wav.py` | sherpa-onnx |
| `nonstream` | `transcribe_wav.py` | sherpa-onnx offline |

Default is `mix-beam`. Expose it with a **cloudflared quick tunnel** (static
binary, no account) — `localhost.run` drops the server-side session while ssh
stays alive, producing a confusing `no tunnel here :(`.

---

## 7. Results

Full 618-utterance memorization set, streaming u20 model:

| runtime | provider | precision | WER | RTF |
| --- | --- | --- | ---: | ---: |
| **our beam_search** | CPU | int8 | **2.37%** | ~0.26 |
| icefall (PyTorch) beam_search | GPU | fp32 | 0.52% | — |
| sherpa-onnx modified_beam | CUDA (C++ binary) | int8 | 9.17% | 0.166 |
| sherpa-onnx modified_beam | CPU (pip wheel) | int8 | ~9.17% | ~0.20 |

On the Nano, the beam decoder ran end-to-end at **~0.97% WER, 4.98 s/utt
(RTF 0.26)** on the bring-up subset — faster *and* far more accurate than the
old sherpa-onnx deployment (~7 s/clip, ~9.2%).

### ⚠️ These WER numbers are a memorization metric

Train, dev, and test are the **same recordings** ("học vẹt" by design). WER here
measures recall of specific recordings — it is **blind to speaker
generalization**, and it is blind to label corruption.

Two things it hid:

1. An **off-by-one bug** paired every Trung/Dung recording with the previous
   sentence's text (`Recording.wav` natural-sorts *after* `Recording (200).wav`).
   The model happily memorized the wrong labels and still scored ~0.5%. Fixed in
   `local/prepare_vi_asr_corpus.py:recorder_key()`.
2. On **held-out speakers** (unseen voices, `eval_heldout_speaker.py`), the same
   models score 70–91% WER. All project speakers are male, so female voices
   collapse.

**`model_divmix_x8_epoch60_avg10` is the first package trained on corrected
labels** (and the first with female voices in training). Every *other* package
here predates the fix — treat their numbers as unreliable.

Held-out speakers, classic beam_search, same 50-clip eval set:

| package | khanh_toan ♂ | yen_nhi ♀ | overall |
| --- | ---: | ---: | ---: |
| `model_streaming_u20_epoch55_avg10` (real only) | 82.70% | 99.31% | 91.00% |
| `model_realdom_mix_epoch60_avg10` (5 male clones) | 49.48% | 92.04% | 70.76% |
| **`model_divmix_x8_epoch60_avg10`** | **1.38%** | **16.96%** | **9.17%** |

---

## 8. Package contents

| file | purpose |
| --- | --- |
| `jetson_beam_decode.py` | numpy+onnxruntime transducer + classic beam_search |
| `jetson_asr.py` | wav → fbank → text, lhotse-matching features |
| `transcribe_beam_wav.py` | CLI, drop-in for the sherpa runner |
| `transcribe_streaming_wav.py`, `transcribe_wav.py` | legacy sherpa-onnx runners |
| `onnx_beam_search.py` | PC-side prototype (imports icefall's `OnnxModel`) |
| `install_jetson.sh` | Nano-side installer |
| `remote_mic_ui/` | browser mic demo + model selector |
| `evaluate_*.py`, `run_performance_eval.sh` | WER/RTF benchmarks |
| `model_*/` | exported ONNX packages (`MODEL_INFO.txt` in each) |
