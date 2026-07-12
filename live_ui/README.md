# Live streaming UI (local)

Real-time ASR in the browser: **words appear while you speak**, not after you
stop. Silero VAD ends each sentence on a pause, so you can dictate continuously.
Runs entirely on this machine.

```bash
GWEN=/path/to/gwen-tts     # any python with onnxruntime + kaldi_native_fbank
$GWEN/.venv-gwen/bin/python live_ui/server.py
# open http://localhost:8100
```

Defaults to `deploy/jetson_nano/model_medium_epoch30_avg10` with classic
`beam_search` (beam 4). Override with `--model-dir` / `--beam` / `--port`,
`--min-silence-ms` (pause that ends a sentence), `--vad-threshold`, `--no-vad`.

## From your phone

A browser only grants microphone access over **HTTPS** (or `localhost`), so LAN
IP alone will not work. Use a tunnel:

```bash
cloudflared tunnel --url http://localhost:8100
# -> https://<random>.trycloudflare.com   open that on the phone
```

The page picks `wss://` automatically when served over HTTPS. Verified: audio
streams and transcribes correctly through the tunnel.

## How this differs from batch transcription (`transcribe_beam_wav.py`)

|  | batch (`transcribe_beam_wav.py`) | `live_ui` (this) |
| --- | --- | --- |
| when you see text | after the whole clip | while you speak |
| audio path | wav file → decoder | mic → WebSocket → this machine |
| decode | one batch pass | incremental, per encoder chunk |
| latency | full clip × RTF, after the fact | ~0.64 s behind the microphone |

They share the model and the decoder.

## Why it is exact, not an approximation

The model is a **causal streaming Zipformer**: the encoder consumes a 77-frame
window, advances 64 frames, and carries 74 cached state tensors forward. It was
always able to run chunk-by-chunk — the old runner simply fed it the whole
utterance at once.

`jetson_beam_decode.beam_search` is already **frame-synchronous**: it loops over
encoder frames carrying the hypothesis set `B` and the decoder cache.
`stream_decoder.IncrementalBeam` splits that loop into a `step(enc_t)` call. Same
algorithm, same beam, same output.

That last claim is enforced, not asserted:

```bash
python3 live_ui/stream_decoder.py --self-check some_16k.wav
# asserts streaming output == batch beam_search output
```

Verified identical on all 12 example clips (2.9 s – 9.8 s).

## Latency

Partials update every **64 encoder frames = 0.64 s** — that is
`decode_chunk_len` baked into the exported model, not a tunable here. Audio is
sent in 128 ms blocks, so a chunk fires as soon as its frames arrive.

Measured RTF **0.027** on this machine (≈37× real time), so the decoder is never
the bottleneck; the 0.64 s chunk boundary is.

## Feature parity

`stream_decoder.fbank_opts()` mirrors lhotse's `Fbank` defaults exactly, and the
browser is asked for a **16 kHz** `AudioContext` so no resampling happens. PCM is
scaled to `[-1, 1]`. Any drift here silently destroys WER — the UI warns if the
browser refuses 16 kHz.

## VAD / endpointing

`vad.py` runs **Silero VAD** as ONNX through onnxruntime (no torch).
`models/silero_vad.onnx` is vendored from `silero-vad` 6.2.1 (MIT).

Speech starts a decoder session; `min_silence_ms` (700 ms default) of silence
finalizes the sentence and starts a fresh one. This matters: the ASR model was
trained on single 4–5 s sentences, so one ever-growing hypothesis drifts out of
distribution.

Three details that are easy to get wrong, all learned the hard way:

- **The ONNX graph wants 576 samples** (64-sample context + 512 new), not 512.
  Feed it a bare 512 and it does *not* error — it returns ~0 for every frame,
  i.e. "never speech". `vad.py --self-check` asserts it actually finds speech.
- **Pre-roll.** Onset is detected a few frames late, so the last 256 ms of
  pre-speech audio is replayed into the decoder or the first phoneme is clipped.
- **Do not feed the trailing silence.** The endpoint only fires after 700 ms of
  quiet; feeding that to the decoder makes it hallucinate
  (`"...dừng lại và t"`). Audio is withheld once the silence run starts, keeping
  a 192 ms tail for final consonants.

```bash
python3 live_ui/vad.py --self-check some_16k.wav
```

## Known limits

- The model was trained on 4–5 s sentences (only 72 of 12,800 cuts ≥ 15 s).
  Speak in sentences; a long monologue drifts out of distribution.
- One session per WebSocket; each connection loads its own ONNX session (~26 MB).
  Fine for local use, not for many concurrent users.
- Hypotheses are not re-scored across an endpoint: once a sentence finalizes it
  is frozen, even if the next words would have disambiguated it.

## Speaker identification

Each finalized segment is labelled with **who** said it — "Dung: …", "Quan: …" —
via `speaker_id.py`. Because we have enrollment audio for every project speaker,
this is closed-set **identification**, not blind diarization (see
`docs/DIARIZATION.md`).

The 27 MB embedding weight is not in git (download once):

```bash
curl -L -o live_ui/models/speaker_embedding_campplus.onnx \
  https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced.onnx
```

```bash
python3 live_ui/speaker_id.py --separation-check   # prove features separate our voices
python3 live_ui/speaker_id.py --enroll             # build enrolled_speakers.json from dataset/
python3 live_ui/speaker_id.py --identify some.wav
```

Model: 3D-Speaker CAM++ zh_en advanced ONNX (27 MB, 192-dim), no torch. Segment
audio → 80-dim fbank (per-utterance global-mean normalized) → embedding → nearest
enrolled speaker by cosine, rejecting to "unknown" below `--speaker-threshold`
(default 0.5).

Measured on our 4 speakers: same-speaker cosine **0.80** vs different **0.22**
(distributions do not overlap), **100% ID on 120 held-out clips**, unenrolled
`yen_nhi` rejected at 0.25. The Mandarin/English-trained embeddings transfer to
Vietnamese cleanly — verified by `--separation-check`, which asserts the gap.

Disable with `--no-speaker-id`. Turns itself off automatically if
`models/enrolled_speakers.json` is missing.
