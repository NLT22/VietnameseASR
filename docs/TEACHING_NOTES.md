# VietnameseASR — everything you need to understand this project

Written for **you, months from now**. It assumes you remember the project existed
and nothing else. Every number and shape here was read out of the code or
measured on this machine — nothing is from memory. Where the project is broken,
it says so. Where I was wrong earlier, it says that too.

Read `RESULTS.md` for what the models score. This file is *why things are the
way they are*.

---

## 0. The one-paragraph summary

We record 5 people saying 200 short Vietnamese sentences each (**1,000 real
recordings**). The current `main` training set repeats the real recordings and
adds Gwen-TTS voice clones, giving `transcripts_main/train.tsv` about **16,000
training rows** while dev/test stay the clean matched real recordings. We train a
**streaming Zipformer transducer** to transcribe the fixed sentence set. Train,
dev and test intentionally contain the same sentence set ("học vẹt" — rote
learning), so the headline WER measures *memorization of this domain*, not open
Vietnamese ASR skill. The deployed model is a medium Zipformer exported to int8
ONNX and runs on Jetson Nano plus the live browser UI. The single most important
lesson in the project is that **the headline metric was blind to a bug that
corrupted 45% of the training labels for two months**.

> **Update (2026-07-12):** the deployed model is now **medium (M)**
> (`deploy/jetson_nano/model_medium_epoch30_avg10`), 5 speakers incl.
> Hieu: real WER **1.84%**, held-out **1.80%** — it beats the small model on every
> speaker and still runs real-time on the Nano (RTF ≈ 0.31). The `divmix_x8`
> (4-speaker) model used as the running example below was archived + deleted; its
> shapes and pipeline still illustrate how everything works. See `RESULTS.md` and
> `ARCHIVED_EXPERIMENTS.md`.

---

## 1. The model: transducer = three networks

A transducer (RNN-T) is not one model. It is three, trained jointly and exported
as three separate ONNX graphs. Verified shapes from
`deploy/jetson_nano/model_medium_epoch30_avg10/`:

| graph | inputs | outputs | job |
| --- | --- | --- | --- |
| **encoder** | `x[N,77,80]` + **74 state tensors** | `encoder_out[N,16,512]` + 74 new states | listen to audio |
| **decoder** | `y[N,2]` (last 2 token ids) | `decoder_out[N,512]` | remember what was just said |
| **joiner** | `encoder_out[N,512]`, `decoder_out[N,512]` | `logit[N,100]` | score the next token |

### The encoder

Eats **80-dim log-mel filterbank** frames (not raw audio). Its config, from the
ONNX metadata:

```
num_encoder_layers   2,2,2,2,2,2      # 6 stacks x 2 layers = 12 layers
encoder_dims         192,256,256,256,256,256
num_heads            4,4,4,8,4,4
left_context_len     256,128,64,32,64,128
cnn_module_kernels   31,31,15,15,15,31
decode_chunk_len     64
T                    77
```

It is **causal**: it never looks at future audio beyond its chunk. That is what
makes streaming possible, and it costs accuracy versus a bidirectional encoder.

**It subsamples ~4×.** Feed it `(1,77,80)` and you get `(1,16,512)` back. So:

- one feature frame = 10 ms (frame shift)
- one **encoder frame = 40 ms** of audio
- a 77-frame window = `25 + 76*10` = **785 ms** of audio
- the window advances by `decode_chunk_len=64` frames = **640 ms**, producing 16
  encoder frames

The 13-frame overlap (`77 - 64`) is the model's right-context: it peeks 130 ms
ahead within its own chunk.

### The 74 state tensors

This is the thing that confuses everyone. `encoder.onnx` has **75 inputs**: the
audio, plus 74 tensors that carry the encoder's memory across chunks. Their
shapes are computed in `deploy/jetson_nano/jetson_beam_decode.py:reset()`:

```python
for i in range(len(num_encoder_layers)):        # 6 stacks
    key_dim = query_head_dims[i] * num_heads[i]
    nonlin  = 3 * encoder_dims[i] // 4
    val_dim = value_head_dims[i] * num_heads[i]
    pad     = cnn_module_kernels[i] // 2
    for _ in range(num_encoder_layers[i]):      # 2 layers each
        cached_key          (left_context_len[i], N, key_dim)
        cached_nonlin_attn  (1, N, left_context_len[i], nonlin)
        cached_val1         (left_context_len[i], N, val_dim)
        cached_val2         (left_context_len[i], N, val_dim)
        cached_conv1        (N, encoder_dims[i], pad)
        cached_conv2        (N, encoder_dims[i], pad)
embed_states  (N, 128, 3, 19)
processed_lens(N,)
```

`6 stacks × 2 layers × 6 tensors = 72`, plus `embed_states` and `processed_lens`
= **74**. Four of the six are attention KV caches (that is what
`left_context_len` means — how many past frames each stack can attend to); two
are convolution ring buffers, because a causal conv of kernel 31 needs the
previous 15 samples.

Start of an utterance = all 74 are zeros. `reset()` does that. **Forget to reset
between utterances and the model hears the previous sentence.**

### The decoder (it is not what you think)

Not "the thing that decodes." It is a tiny **language model over tokens**. Input
`y[N,2]` — the last two token ids. Output a 512-dim vector.

`context_size=2` means it is a **bigram**: it only knows the previous two tokens.
This is why it is stateless and cacheable — the same two-token context always
gives the same output, which is why `beam_search()` keeps a `dec_cache` dict.

### The joiner

Adds acoustic evidence and language context, projects to `logit[N,100]` — one
score per token in the vocabulary. `blank_id=0`.

### Why the transducer emits multiple symbols per frame

At each encoder frame `t`, the joiner scores 100 tokens. If it picks **blank**,
the model advances to frame `t+1`. If it picks a **real token**, it emits it,
updates the decoder context, and **stays on the same frame `t`** to possibly emit
another.

That loop is the whole point of a transducer, and it is the single most important
fact in this project, because:

> `sherpa-onnx` implements only `greedy_search` and `modified_beam_search`, and
> **both emit at most one symbol per frame.** Our model is trained to emit
> several. On the same weights, sherpa scores 3.31% and our classic
> `beam_search` scores **0.72%** — the gap is almost entirely *deletions* the
> sherpa algorithms structurally cannot produce.

Look at `beam_search()` in `jetson_beam_decode.py`: the `while True:` inside the
`for t in range(...)` loop is the multi-symbol emission. `greedy()` has no such
inner loop, and that is why it is worse.

### The loss (why "pruned")

From `ASR/zipformer/train.py`:

```
simple_loss, pruned_loss, ... = model(..., prune_range=5, am_scale=0.0, lm_scale=0.25)
loss = simple_loss_scale * simple_loss + pruned_loss   # simple_loss_scale=0.5
```

The exact RNN-T loss sums over every path through a `(T frames × U tokens)`
lattice — for us that is ~100 × ~60 cells, each needing a full joiner forward.
That is enormous. So:

1. **`simple_loss`**: replace the joiner with `encoder_out + decoder_out` (a
   cheap sum, no MLP). Compute the exact lattice on that. It is a bad model but a
   good *guide*.
2. Use its gradients to find, for each frame, the ~5 token positions that
   actually matter (`prune_range=5`).
3. **`pruned_loss`**: run the real joiner on only that narrow band.

`lm_scale=0.25` and `am_scale=0.0` weight how much the simple model's LM/AM
components contribute to picking the band. This is why it is called **Pruned
RNN-T**. It is an optimization of the loss, not of the model.

---

## 2. Tokens: why a 100-token vocabulary for 1,913 unique words

`data/lang_bpe_100/` holds a **SentencePiece BPE** model with **exactly 100**
tokens. `tokens.txt` maps token → id and is what `logit[N,100]` indexes:

```
<blk>      0     # the transducer's "emit nothing, advance time"
<sos/eos>  1
<unk>      2
▁t         3     # "▁" (U+2581) means "start of word"
ng         4
...
ỵ          99
```

Our corpus has **1,913 unique words / 20,790 word tokens** (the corpus is
already NFC-normalized; verified). You cannot have a word-level vocabulary —
most words would appear a handful of times. BPE splits words into frequent
subword pieces, so `▁t` + `ôi` composes "tôi" and unseen words still decompose
into known pieces.

> **Trap.** `cut -f4 | tr ' ' '\n' | sort -u | wc -l` reports **2,279** for this
> file. That is wrong — `sort -u` under glibc collation leaves 366 duplicate
> lines on Vietnamese diacritics. Count unique tokens in Python, not in `sort`.
> I published 2,279 in the first draft of this document for exactly that reason.

Why 100 and not 500? Vietnamese is **monosyllabic and analytic**: words are short
(1-2 syllables), the phoneme inventory is small, and orthography is nearly
phonetic. A hundred pieces covers the syllable structure. A bigger vocabulary
would mean more output classes, each with fewer training examples — and we only
have ~1.1 hours of real audio.

Detokenization is just `"".join(pieces).replace("▁", " ")` — see `detok()`.

**Consequence to remember:** the joiner's 100 logits are *subword* scores. A
single wrong token can mangle a word without changing sentence length, which is
why WER can look stable while the output turns to nonsense.

---

## 3. The data pipeline, stage by stage

`run.sh` is the single entry point (`bash run.sh --help`) for data prep, train,
decode, and deployment export. Stage 16 calls `local/export_for_jetson.sh`, which
wraps the icefall ONNX exporter and writes the Jetson/live-UI model package.

```
dataset/<Speaker>/            raw wavs + script.txt   (one line per recording)
   │  stage -1  local/prepare_matched_splits.py
   ▼
transcripts_matched_u20/{train,dev,test}.tsv          utt_id, speaker, audio_path, text
   │  stage -2  local/tts/*.py + local/hieu_pipeline.py -> transcripts_main/
   │  stage 3   local/prepare_manifests.py            -> lhotse Recording/Supervision
   │  stage 4   fix manifests
   │  stage 5-7 export text corpus -> train BPE -> lang_bpe_100/
   │  stage 8   local/compute_fbank.py                -> 80-dim lhotse Fbank
   ▼
fbank_<tag>/{train,dev,test}_cuts.jsonl.gz            + feature .lca files
   │  stage 9   MUSAN fbank (for online noise mixing)
   │  stage 13  ASR/zipformer/train.py
   ▼
ASR/zipformer/exp_.../epoch-N.pt
   │  stage 16 local/export_for_jetson.sh
   │           (export-onnx-streaming.py + quantize_dynamic)
   ▼
deploy/jetson_nano/model_.../{encoder,decoder,joiner}.int8.onnx + tokens.txt
```

For the current deployed line:

```
transcripts_matched_u20/     1,000 clean real utterances, same sentence set
transcripts_main/            train = real x8 + TTS clones; dev/test = clean real
data/manifests_main/         lhotse manifests for transcripts_main
fbank_main/                  80-bin Fbank cuts/features for main
data/lang_bpe_100/           SentencePiece BPE model and tokens.txt
ASR/zipformer/exp_bpe100_medium_streaming_main_lr0045/
deploy/jetson_nano/model_medium_epoch30_avg10/
```

### Theory-to-code map

This is the practical bridge from the theory in this document to the files that
implement it now:

| concept | current file(s) | job |
| --- | --- | --- |
| Build matched real splits | `local/prepare_matched_splits.py`, `local/prepare_vi_asr_corpus.py` | pair `dataset/<speaker>/` audio with script lines, produce train/dev/test TSV |
| Generate and mix TTS clones | `local/tts/build_crossspeaker.py`, `local/tts/build_diverse_clones.py`, `local/hieu_pipeline.py` | create clone TSV/audio and assemble `transcripts_main` |
| Audit labels/durations | `local/audit_dataset.py`, `local/validate_manifest.py`, `local/display_manifest_statistics.py` | catch broken paths, bad duration/text, suspicious data |
| Lhotse manifests | `local/prepare_manifests.py` | TSV -> recordings/supervisions JSONL.GZ |
| Text corpus and BPE | `local/export_text_corpus.py`, `local/train_bpe_model.py`, `local/prepare_lang_bpe.py` | train SentencePiece BPE and create `tokens.txt` / `words.txt` |
| Fbank features | `local/compute_fbank.py`, `local/compute_fbank_musan.py` | write lhotse cuts and optional MUSAN cuts |
| Data module | `ASR/zipformer/asr_datamodule.py` | load cuts, batch by duration, apply augmentation |
| Model definition | `ASR/zipformer/model.py`, `encoder_interface.py`, `zipformer.py`, `subsampling.py`, `decoder.py`, `joiner.py` | Zipformer-Transducer network |
| Training | `ASR/zipformer/train.py`, `optim.py`, `scaling.py` | Pruned RNN-T loss, ScaledAdam/Eden, checkpoint writing |
| Averaging | `ASR/zipformer/generate_averaged_model.py` plus decode/export flags | average checkpoints or use `model_avg` |
| PyTorch decode | `ASR/zipformer/decode.py`, `streaming_decode.py`, `beam_search.py`, `streaming_beam_search.py` | WER on cuts/checkpoints |
| ONNX export | `local/export_for_jetson.sh`, `ASR/zipformer/export-onnx-streaming.py`, `export-onnx.py` | export encoder/decoder/joiner graphs and int8 quantize |
| ONNX runtime decode | `deploy/jetson_nano/jetson_beam_decode.py`, `transcribe_beam_wav.py`, `jetson_asr.py`, `onnx_beam_search.py` | production-style inference without torch/k2 |
| Performance eval | `deploy/jetson_nano/evaluate_performance.py`, `evaluate_streaming_performance.py`, `run_performance_eval.sh` | PC/Jetson speed and WER checks |
| Live UI | `live_ui/server.py`, `live_ui/stream_decoder.py`, `live_ui/vad.py`, `live_ui/speaker_id.py` | browser mic, streaming decode, VAD, speaker ID |

The old CTC/JIT recipe files were intentionally removed from this project. The
active path is Transducer -> ONNX encoder/decoder/joiner -> custom beam search.

### How audio pairs with text (the bug that defines this project)

`prepare_matched_splits.py` calls `scan_auto_dataset()`, which does:

```python
audio_files = list_audio_files(speaker_dir)   # sorted
texts       = read_prompt_lines(script_file)  # one line per recording
assert len(audio_files) == len(texts)
zip(audio_files, texts)                       # POSITIONAL pairing
```

There is **no id in the transcript**. Pairing is purely positional, so the sort
order of the audio files *is* the labelling.

Windows Voice Recorder names the first take `Recording.wav` and later takes
`Recording (2).wav`, `(3)`… Natural sort puts the **unnumbered file last**. So
line 1 paired with take 2, line 2 with take 3, …, line 200 with take 1.

Every Trung and Dung utterance was labelled with the previous sentence's text —
**330 of 730 recordings**. Quan (`Recording (1..200)`) and Khoi (`001..200.wav`)
have no unnumbered file and were never affected.

Fixed by `recorder_key()` in `local/prepare_vi_asr_corpus.py`:

```python
def recorder_key(name):
    m = re.search(r"\((\d+)\)", name)
    return (int(m.group(1)) if m else 1, natural_key(name))   # missing "(N)" == take 1
```

**The check that catches it.** Longer sentences take longer to say, so
`corr(audio_duration, text_word_count)` should be clearly positive per speaker. A
shifted set collapses toward zero. Quan, having no bare file, is identical under
both orderings — a free control.

| speaker | natural sort | bare-first | |
| --- | ---: | ---: | --- |
| Quan (control) | +0.472 | +0.472 | unaffected |
| Trung | +0.858 | **+0.954** | shifted |
| Dung | −0.013 | **+0.216** | shifted |

**Run this after any ingest. It is three lines and it would have saved months.**

### Features: lhotse Fbank, and the scaling trap

Training uses **lhotse's `Fbank`**, so inference must reproduce it exactly.
`live_ui/stream_decoder.py:fbank_opts()` and `deploy/jetson_nano/jetson_asr.py`
both mirror it:

```
samp_freq 16000   dither 0.0        snip_edges False
frame_length 25ms frame_shift 10ms  preemph 0.97
window povey      remove_dc_offset True
num_mel_bins 80   low_freq 20       high_freq -400   (i.e. 8000-400 = 7600 Hz)
```

**The waveform must be float in `[-1, 1]`.** Kaldi natively uses
`[-32768, 32767]`. Feed it integers and every mel bin shifts by
`ln(32768²) ≈ 20.8`, and the model sees features it has never seen. This is a
silent, total failure — no exception, just garbage. `pcm.astype(np.float32) /
32768.0` everywhere.

### Augmentation

- **Online MUSAN** (`--enable-musan`): noise mixed in on the fly during training.
  Needs `fbank_*/musan_cuts.jsonl.gz` from stage 9.
- **SpecAugment** (`--enable-spec-aug`): masks time/frequency bands of the fbank.
- **Speed perturb**: *off*. It triples the number of cuts, for variety that our
  9 distinct voices already supply.
- **Offline MUSAN copies**: abandoned. It multiplies disk 5-10× and only teaches
  noise robustness, whereas the axis we actually lacked was *speaker variety*.

### The clones

`local/tts/` uses **Gwen-TTS** (Qwen3-TTS-0.6B finetuned on 1000 h Vietnamese) to
speak each sentence in other voices. Cross-speaker clones give 5 male voices;
diverse clones add 3 female + 1 male.

Two rules, both learned by breaking them:

1. **`yen_nhi` and `khanh_toan` never enter training.** They are the held-out
   eval voices. Clone a training sentence into them and the honest benchmark
   silently becomes another memorization test.
2. **Reference audio quality propagates.** Dung's originals peaked at 0.059 and
   cloned to a near-inaudible voice. References come from
   `datasets/vi_asr_corpus/`, are silence-trimmed and peak-normalized, and
   `ref_text` must differ from `gen_text` or the output repeats words.

Ratio matters: ~80% synthetic gave degenerate blank-heavy decodes (27-48% WER).
26-50% synthetic trains healthily.

---

## 3b. Config parameters — the complete list

Defaults below are **`run.sh`'s**, read from the code. The current recipe default
is `base_lr=0.045`; older notes that mention a forced `0.01` describe the bug
that made early models train too slowly.

### run.sh — pipeline

| flag | default | meaning |
| --- | --- | --- |
| `--stage` / `--stop_stage` | `-1` / `100` | which stages to run |
| `--data_tag` | `""` | run a pre-built transcript set: `transcripts_NAME/` → `fbank_NAME/`, `data/manifests_NAME/`, exp suffix `_NAME`. Forces `matched_splits=0` |
| `--matched_splits` | `1` | use the "học vẹt" split (train == dev == test) |
| `--exp_suffix` | `_x10_matched` | experiment dir suffix |
| `--exp_dir_policy` | `auto` | `auto` new dir if non-empty, `reuse` resume, `fail` stop |
| `--build_clones` | `0` | stage -2: build Gwen-TTS clones and assemble the tagged train set |
| `--do_export` | `1` | stage 16: export int8 ONNX package |

### run.sh — data / features

| flag | default | meaning |
| --- | --- | --- |
| `--vocab_size` | `100` | BPE tokens → joiner output dim |
| `--split_max_duration` | `20` | drop recordings > 20 s when building splits |
| `--feature_max_duration` | `20` | cut filter at fbank time |
| `--musan_dir` | `./musan` | MUSAN root |

### run.sh — augmentation

| flag | default | we used | meaning |
| --- | --- | --- | --- |
| `--enable_musan` | `0` | **1** | online MUSAN noise mixing |
| `--enable_spec_aug` | `0` | **1** | SpecAugment |
| `--perturb_speed` | `0` | `0` | speed perturb (triples cuts) |
| `--offline_musan_aug` | `1` | **0** | write noisy copies to disk |
| `--copies_per_utt` / `--snr_min` / `--snr_max` | `10` / `10` / `20` | – | offline aug only |
| `--real_mult` | `8` | **8** | repeat real recordings when assembling clone-mix train sets |

### run.sh — model size

`--model_size small` forces (preset `base` = leave `train.py` defaults):

```
num_encoder_layers    2,2,2,2,2,2
feedforward_dim       512,768,768,768,768,768
num_heads             4,4,4,8,4,4
encoder_dim           192,256,256,256,256,256
encoder_unmasked_dim  192,192,192,192,192,192
decoder_dim           512
joiner_dim            512
```

Individually overridable: `--num_encoder_layers`, `--encoder_dim`,
`--feedforward_dim`, `--num_heads`, `--encoder_unmasked_dim`, `--decoder_dim`,
`--joiner_dim`.

### run.sh — training

| flag | default | we used | meaning |
| --- | --- | --- | --- |
| `--num_epochs` | `30` | **60 for sweeps; deployed epoch 30** | epochs |
| `--start_epoch` | `1` | – | resume; loads `epoch-(N-1).pt` |
| `--max_duration` | `500` | **700** | **seconds of audio per batch**, not utterances |
| `--num_workers` | `2` | **4** | dataloader workers |
| `--base_lr` | `0.045` | **0.045** | LR; the old forced `0.01` was much worse |
| `--use_fp16` | – | **1** | mixed precision |
| `--bucketing_sampler` / `--num_buckets` | `1` / `4` | same | batch similar-length cuts |
| `--causal` | `0` | **1** | **required for streaming** |
| `--chunk_size` | `16,32,64,-1` | same | a *list*; training randomises over it |
| `--left_context_frames` | `64,128,256,-1` | same | a *list* |

### Loss (passed through to train.py)

| flag | default | meaning |
| --- | --- | --- |
| `--prune-range` | `5` | width of the token band kept after pruning |
| `--lm-scale` / `--am-scale` | `0.25` / `0.0` | LM/AM weight when choosing the band |
| `--simple-loss-scale` | `0.5` | weight of `simple_loss` |
| `--ctc-loss-scale` / `--cr-loss-scale` | `0.2` / `0.2` | only with `--use_ctc` / `--use_cr_ctc` |

### train.py only (not exposed by run.sh — edit the code)

| flag | default | meaning |
| --- | --- | --- |
| `--average-period` | **200** | batches between `model_avg` updates |
| `--lr-batches` / `--lr-epochs` | `7500` / `3.5` | LR decay constants |
| `--warm-step` | `2000` | warmup; drives `simple_loss_scale` decay |
| `--seed` | `42` | |

> ⚠️ **`average-period=200` > 159 batches/epoch** (12,800 cuts at
> `max_duration 700`). `model_avg` — what `--use_averaged_model` reads — updates
> *less than once per epoch*, so adjacent checkpoints often carry the identical
> averaged snapshot: `avg=1` and `avg=2` export **byte-identical** weights.
> Lower `--average-period` to ~50 for real averaging.

### Decode / export

| flag | default | we used | meaning |
| --- | --- | --- | --- |
| `--decode_methods` | `all` | `all` | greedy / modified_beam / beam_search |
| `--use_averaged_model` | `0` | **1** | use `model_avg` |
| `--avg` | `1` | **10** | average the last N checkpoints |
| `--decode_chunk_size` | `32` | `32` | a *single value* (unlike `--chunk_size`) |
| `--decode_left_context_frames` | `256` | `256` | single value |

### The command for the current `main` line

```bash
GWEN_TTS_DIR=/path/to/gwen-tts bash run.sh \
  --data_tag main --build_clones 1 \
  --vocab_size 100 --model_size medium --causal 1 \
  --base_lr 0.045 --use_fp16 1 --max_duration 700 --num_workers 4 \
  --num_epochs 60 --use_averaged_model 1 --avg 10 \
  --enable_musan 1 --enable_spec_aug 1 --perturb_speed 0 \
  --offline_musan_aug 0 --exp_suffix "_lr0045" \
  --stage -2 --stop_stage 16
```

If `transcripts_main/` already exists and you do not want to regenerate clones,
start at stage 3 with the same flags.

### live_ui / VAD

| flag | default | meaning |
| --- | --- | --- |
| `--beam` | `4` | beam width |
| `--no-vad` | off | one ever-growing hypothesis |
| `--vad-threshold` | `0.5` | speech entry threshold |
| `--min-silence-ms` | `700` | pause that ends a sentence |

In-code constants: `FRAME=512`, `CONTEXT=64` (graph takes 576), `lo = threshold −
0.15` (hysteresis), `min_speech_ms=250`, `speech_pad_ms=200`, `PREROLL_FRAMES=8`
(256 ms), `TAIL_FRAMES=6` (192 ms).

## 4. Why our benchmarks lie (read this twice)

There are two benchmarks and they measure different things. Confusing them is how
a two-month bug survived.

### The memorization set

`transcripts_matched_u20/{train,dev,test}.tsv` are **byte-identical** (same
md5). The 800 real recordings appear in training ×8. So "test WER" is:

> given a recording the model was trained on, does it reproduce the label it was
> trained on?

This is **blind to two things**:

1. **Label corruption.** A model happily memorizes wrong labels *consistently*
   and still scores ~0.5%. The off-by-one bug made no dent in this metric.
2. **Speaker generalization.** It cannot tell you whether the mic UI works for a
   human.

It is not useless — it was the original project goal — but it must never be
quoted as "the model's WER" without that sentence attached.

### The held-out speaker set

`eval_heldout_speaker.py`: 25 known sentences × 2 voices never seen in training
(`yen_nhi` ♀, `khanh_toan` ♂). This is the number that predicts whether a
stranger can use the system.

### They disagree, and that is informative

The older pre-Hieu `avg` sweep (checkpoint averaging over the last N epochs, all
`beam_search` beam 4) is still useful because it shows the trade-off:

| avg | real 800 recordings (pre-Hieu) | held-out speakers |
| --- | ---: | ---: |
| 1 | 2.30% | 8.48% |
| 10 *(deployed)* | 2.18% | 9.17% |
| 30 | **1.54%** | 10.73% |

More averaging is **significantly** better on the recordings we trained on
(−0.64 pts, 95% CI [−0.79, −0.50], 20,790 words) and **probably** worse on unseen
voices (+1.54 pts, 95% CI [−0.52, +3.62] — spans zero, P=0.92, only 578 words).

That is memorize-vs-generalize, made visible only because both metrics exist.
We keep `avg=10`: not significantly worse than anything on either metric.
`avg=30` is over-fitting to recordings we already own.

### Statistical power — the part I got wrong

The held-out set has **578 reference words**. A difference of 4 words is 0.69 WER
points. I quoted "9.17% vs 8.48%" as if it meant something; bootstrapping over
the 50 utterances gives a 95% CI of [−0.86, +2.26]. **It cannot resolve
differences under ~2 WER points.** Any comparison I made on that set below that
threshold was noise dressed as a finding.

Rule: before believing a WER difference, bootstrap-resample the utterances. It is
ten lines and it will delete half your conclusions.

### The benchmark nobody built

Your actual use case is **an unseen recording of a seen speaker** — you saying a
sentence that is in the dataset, but a fresh take. The memorization set is the
same recordings; the held-out set is different voices. *Neither is your case.*
Twenty fresh recordings from the five project speakers would settle every `avg`/decoder
question honestly. It does not exist yet.

---

## 5. Streaming, for real

### Why the model can stream at all

The encoder is causal and carries its 74 states forward. It was **always** able
to run chunk-by-chunk; the batch runner just handed it the whole utterance.
`OnnxTransducer.encode()` literally loops:

```python
while start + segment <= len(x):        # segment = 77
    run encoder on x[start : start+77] with current states
    states = new_states
    start += offset                     # offset = 64
```

### Why incremental decoding is *exact*, not an approximation

`beam_search()` is **frame-synchronous** — it loops over encoder frames carrying
the hypothesis set `B` and `dec_cache`. `live_ui/stream_decoder.py` splits that
loop into `IncrementalBeam.step(enc_t)`. Same algorithm, same beam, same output.

Not asserted — enforced:

```bash
python3 live_ui/stream_decoder.py --self-check some_16k.wav
# asserts streaming output == batch beam_search output
```

Passes on all 12 example clips (2.9-9.8 s).

**Partials update every 0.64 s** (`decode_chunk_len=64` × 10 ms). That is baked
into the export, not tunable in the UI. Decode RTF is 0.027-0.23, so the chunk
boundary is the *only* latency.

### Silero VAD, and its three traps

`live_ui/vad.py` runs Silero as ONNX through onnxruntime (no torch). Speech
starts a decoder session; 700 ms of silence finalizes the sentence and starts a
fresh one — necessary because the model was trained on single 4-5 s sentences and
one ever-growing hypothesis drifts out of distribution.

1. **The graph wants 576 samples**, not 512: 64 samples of context + 512 new.
   Feed it a bare 512 and it does **not error** — it returns ≈0 for every frame,
   i.e. "never speech". My first version reported 0% speech on a speech file.
   `vad.py --self-check` now asserts it actually *finds* the speech.
2. **Pre-roll.** Onset is detected several frames late; replay the last 256 ms
   into the decoder or the first phoneme is clipped.
3. **Never feed the trailing silence.** The endpoint fires only after 700 ms of
   quiet. Feeding that to the decoder makes it hallucinate — real observed
   output: `"...dừng lại và t"`, `"...mùa hạn chỉ"`. Audio is withheld once the
   silence run begins, keeping a 192 ms tail for final consonants.

---

## 6. Deployment

### Export

```bash
bash local/export_for_jetson.sh --exp-dir ASR/zipformer/exp_... \
     --epoch 60 --avg 10 --streaming 1 --use-averaged-model 1
```

Runs `export-onnx-streaming.py`, then `quantize_dynamic(..., weight_type=QuantType.QInt8)`,
then copies `tokens.txt` + `bpe.model`. Output is a self-contained model dir.

### Precision

| | status |
| --- | --- |
| **int8** | default. 35% faster, 3× smaller, **identical error counts to fp32** (measured) |
| fp32 | works, no accuracy gain here |
| **fp16** | **will not load** |

fp16 fails at session init with mixed fp16/fp32 `Cast` nodes inside the Zipformer
graph:

```
Type Error: Type (tensor(float16)) of output arg
(/feed_forward1/out_proj/Cast_output_0) does not match expected type (tensor(float))
```

Training's `--use-fp16` (mixed-precision autocast) and the exporter's `--fp16`
(weight conversion) are unrelated things. Do not conflate them.

### Runtime: do not use sherpa-onnx

sherpa-onnx is the standard k2 runtime and it is **the wrong choice for this
model**, because it lacks classic `beam_search` (see §1). We ship
`jetson_beam_decode.py`: **numpy + onnxruntime only**, no torch, no k2, no
sherpa. It is both more accurate *and* faster than what it replaced.

> **Correction to an earlier claim.** I once wrote that sherpa's ~9.2% was a hard
> structural floor. Wrong. Most of it was the label bug — after the fix the same
> `modified_beam_search` scores 3.31%. The *gap between decoders* is structural
> (3.31% vs 0.72% on identical weights); its *size* was inflated by bad data.
> Measure both decoders on every new model.

### Jetson Nano environment (JetPack 4 / L4T R32.7.6 / Python 3.6.9 / CUDA 10)

| thing | requirement | failure if wrong |
| --- | --- | --- |
| `onnxruntime` | **1.10.0** | newer wheels need unavailable glibc/CUDA |
| `numpy` | **1.19.5** | apt's 1.13.3 lacks `numpy.core._multiarray_umath` → **segfault** |
| `kaldi-native-fbank` | source build | no aarch64 wheel for py3.6 |
| **`OPENBLAS_CORETYPE=ARMV8`** | must be exported | numpy misdetects the CPU → **"Illegal instruction"**, no message |

`~/.bashrc` sets `OPENBLAS_CORETYPE`, but **`.bashrc` early-returns for
non-interactive shells**, so any `ssh host 'command'` must export it explicitly.

- The pip `sherpa_onnx` wheel is **CPU-only** and *silently* falls back when given
  `provider="cuda"`. GPU needs a source build with `-DSHERPA_ONNX_ENABLE_GPU=ON`.
- **GPU is not worth it** for single-utterance UI: CPU ≈7 s/clip vs GPU ≈10 s
  warm, ≈30 s cold (CUDA init per call). GPU wins only on sustained batch eval.
- **TensorRT does not work.** Non-streaming is rejected outright
  (`Tensorrt support for Online models only`); streaming fp32 never finishes its
  engine build in 300 s; streaming int8 aborts on a missing shape for a quantized
  MatMul.

### Non-streaming: why it is not deployed

The non-streaming encoder takes **2 inputs** (`x[N,T,80]`, `x_lens[N]`) and
carries no chunk metadata — our `jetson_beam_decode.py` reads `T` /
`decode_chunk_len` and builds 74 state tensors, so it would crash at load. The
port is small (one forward pass instead of the chunk loop) but unwritten. Without
it, non-streaming falls back to sherpa-onnx and its weaker decoders. It also
cannot produce partial results by definition, and its memory grows with utterance
length.

---

## 7. Limitations, honestly

- **Long utterances are out of distribution.** The re-recorded sentences are
  4-7 s, so very long clips are barely represented. A 19 s clip was observed to
  decode to an **empty string** in the archived line. Fine for short commands;
  unusable for long-form.
- **`yen_nhi` (♀) carries almost all remaining held-out error** — 49 of 53 words.
  One female voice in training took her 92% → 17%. More female references is the
  obvious next lever.
- **Speaker-specific errors still matter.** In the final medium model, Trung
  remains harder than Quan/Dung/Hieu. Treat per-speaker WER as a first-class
  metric, not just the average.
- **`average_period=200` > 159 batches/epoch.** `model_avg` — what
  `--use-averaged-model` reads — updates **less than once per epoch**. Adjacent
  epoch checkpoints often carry *identical* averaged weights (`avg=1` and `avg=2`
  export byte-identical fp32). Checkpoint averaging is doing less than the flag
  implies. Drop `--average-period` below ~50 if you want real averaging.
- **The held-out eval is underpowered** (578 words, ~2 WER points resolution).
- **Everything trained before 2026-07-09 used corrupted labels** and was deleted.

---

## 8. Debugging playbook

**The model outputs garbage / empty strings.**
1. Feature parity: is the waveform in `[-1,1]`? Is it 16 kHz mono? Are the fbank
   opts identical to `fbank_opts()`?
2. Did you `reset()` the 74 encoder states between utterances?
3. Is the clip longer than ~15 s? Out of distribution → blank-heavy → empty.

**WER is great but real speech fails.**
You have a memorization metric and a label bug. Run the alignment check:
`corr(audio_duration, text_word_count)` per speaker. Then decode a *training*
wav (should be near-perfect) versus a fresh recording of the same sentence.

**Blank-heavy / mass deletions.**
You are probably on `greedy_search` or `modified_beam_search`, which emit ≤1
symbol per frame. Use classic `beam_search`. Check the deletion count in the
error stats, not just the WER.

**A WER difference looks meaningful.**
Bootstrap-resample the utterances and get a CI. On the held-out set, anything
under ~2 points is noise.

**`pgrep -f` / `pkill -f` match their own command line.**
This has bitten me **three times**, including inside the guard I wrote to prevent
it. A background monitor grepping for its own target either kills itself or loops
forever. Grep log files, or use a bracket class (`serve[r].py`). Also: `pkill`
returning 1 (no match) aborts a compound command under `set -e`.

**Other traps.**
- `/tmp` is wiped on reboot. Keep training logs on the project disk.
- Power loss truncates the in-flight checkpoint to **0 bytes**. Delete it and
  resume with `--start_epoch N` (which loads `epoch-(N-1).pt`). Nothing else is
  lost.
- Export is `run.sh` stage 16, implemented by `local/export_for_jetson.sh`.
- Selecting a checkpoint by validation loss is wrong here — dev == train.

---

## 9. If you pick this up again

In rough order of expected value:

1. **Record 20 fresh utterances per speaker** (sentences already in the dataset).
   That is the real deployment condition and no benchmark covers it. It settles
   `avg`, decoder, and speaker-specific questions at once.
2. **Add more female reference voices** to the clone set. `yen_nhi` is 92% of the
   remaining held-out error.
3. **Enlarge the held-out eval** past 578 words so it can resolve <2 WER points.
4. Check whether Trung's persistent gap is the de-clipping, by scoring his
   original (clipped) audio against the repaired audio on the same model.
5. Lower `--average-period` to ~50 and see whether checkpoint averaging starts
   behaving as documented.
