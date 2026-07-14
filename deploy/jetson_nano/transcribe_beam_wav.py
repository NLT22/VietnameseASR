#!/usr/bin/env python3
"""
Drop-in replacement for transcribe_streaming_wav.py that uses the classic
RNN-T beam_search decoder (~2.4% WER) instead of sherpa-onnx's
modified_beam_search (~9.2% floor).

Same CLI contract as the sherpa scripts, and prints ONLY the transcript on the
final stdout line -- callers should read stdout's last line.

Accepts any wav (resamples to 16 kHz mono internally), unlike the strict
jetson_asr.py, because the browser records at arbitrary rates.
"""
import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from jetson_beam_decode import (  # noqa: E402
    OnnxTransducer, beam_search, greedy, load_tokens, detok,
)
from jetson_asr import compute_fbank  # noqa: E402


def load_wav_any(path):
    """Load any wav -> mono float32 [-1,1] @ 16 kHz (lhotse's scale)."""
    import soundfile as sf

    y, sr = sf.read(path, dtype="float32", always_2d=False)
    if getattr(y, "ndim", 1) == 2:
        y = y.mean(axis=1)
    if sr != 16000:
        import librosa

        y = librosa.resample(y, orig_sr=sr, target_sr=16000)
    return np.asarray(y, dtype=np.float32)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", required=True)
    p.add_argument("--beam", type=int, default=4)
    p.add_argument("--method", choices=["beam_search", "greedy"], default="beam_search")
    # accepted for CLI compatibility with the sherpa-onnx scripts; unused here
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--provider", default="cpu")
    p.add_argument("--max-active-paths", type=int, default=4)
    p.add_argument("--fp32", action="store_true")
    p.add_argument("--rtf", action="store_true", help="print timing / real-time factor to stderr")
    p.add_argument("wav")
    a = p.parse_args()

    model_dir = a.model_dir
    if not os.path.isabs(model_dir):
        model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), model_dir)

    m = OnnxTransducer(model_dir)
    id2tok = load_tokens(os.path.join(model_dir, "tokens.txt"))

    t0 = time.time()
    samples = load_wav_any(a.wav)
    audio_dur = len(samples) / 16000.0
    feats = compute_fbank(samples)
    enc = m.encode(feats)
    ids = beam_search(m, enc, a.beam) if a.method == "beam_search" else greedy(m, enc)
    elapsed = time.time() - t0

    # server.py takes the LAST stdout line as the transcript
    print(detok(ids, id2tok))

    if a.rtf:
        rtf = elapsed / audio_dur if audio_dur > 0 else float("inf")
        print(
            f"[timing] audio={audio_dur:.2f}s process={elapsed:.2f}s RTF={rtf:.3f}",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
