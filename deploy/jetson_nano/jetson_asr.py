#!/usr/bin/env python3
"""
End-to-end Vietnamese ASR on Jetson Nano: wav -> text.

  wav (16k mono) -> kaldi-native-fbank (80-dim) -> ONNX streaming encoder
                 -> classic RNN-T beam_search -> text

Deps: onnxruntime, numpy, kaldi-native-fbank.  (No torch / sherpa-onnx.)
Requires: export OPENBLAS_CORETYPE=ARMV8   (Nano OpenBLAS CPU detection bug)

Fbank settings mirror lhotse's FbankConfig defaults used at training time
(povey window, dither 0, snip_edges False, preemph 0.97, low 20, high -400,
80 bins) and the waveform is kept in [-1, 1] -- verified to match the training
features (mean abs diff 0.016, i.e. lilcom compression noise).

Usage:
  export OPENBLAS_CORETYPE=ARMV8
  python3 jetson_asr.py --model-dir model audio.wav [more.wav ...]
"""
import argparse
import time
import wave

import numpy as np

from jetson_beam_decode import (
    OnnxTransducer, beam_search, greedy, load_tokens, detok, edit_distance,
)


def read_wav16k_mono(path):
    w = wave.open(path, "rb")
    if w.getframerate() != 16000 or w.getnchannels() != 1 or w.getsampwidth() != 2:
        raise ValueError(
            f"{path}: need 16 kHz mono 16-bit PCM, got "
            f"{w.getframerate()} Hz / {w.getnchannels()} ch / {w.getsampwidth()*8} bit.\n"
            f"  convert with: ffmpeg -i in.wav -ar 16000 -ac 1 out.wav"
        )
    pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    return pcm.astype(np.float32) / 32768.0  # [-1,1] -- matches lhotse


def compute_fbank(samples):
    import kaldi_native_fbank as knf

    o = knf.FbankOptions()
    o.frame_opts.samp_freq = 16000
    o.frame_opts.dither = 0.0
    o.frame_opts.snip_edges = False
    o.frame_opts.frame_length_ms = 25.0
    o.frame_opts.frame_shift_ms = 10.0
    o.frame_opts.remove_dc_offset = True
    o.frame_opts.preemph_coeff = 0.97
    o.frame_opts.window_type = "povey"
    o.mel_opts.num_bins = 80
    o.mel_opts.low_freq = 20.0
    o.mel_opts.high_freq = -400.0
    o.energy_floor = 1e-10
    o.use_energy = False

    f = knf.OnlineFbank(o)
    f.accept_waveform(16000, samples.tolist())
    f.input_finished()
    return np.stack([f.get_frame(i) for i in range(f.num_frames_ready)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="model")
    ap.add_argument("--beam", type=int, default=4)
    ap.add_argument("--method", choices=["beam_search", "greedy"], default="beam_search")
    ap.add_argument("--ref", help="optional reference text file to score WER")
    ap.add_argument("wavs", nargs="+")
    a = ap.parse_args()

    m = OnnxTransducer(a.model_dir)
    id2tok = load_tokens(f"{a.model_dir}/tokens.txt")

    tot_e = tot_r = 0
    for path in a.wavs:
        t0 = time.time()
        samples = read_wav16k_mono(path)
        feats = compute_fbank(samples)
        t1 = time.time()
        enc = m.encode(feats)
        t2 = time.time()
        ids = beam_search(m, enc, a.beam) if a.method == "beam_search" else greedy(m, enc)
        t3 = time.time()

        text = detok(ids, id2tok)
        dur = len(samples) / 16000.0
        total = t3 - t0
        print(f"\n{path}  ({dur:.1f}s audio)")
        print(f"  {text}")
        print(f"  fbank {t1-t0:.2f}s  encode {t2-t1:.2f}s  decode {t3-t2:.2f}s"
              f"  | total {total:.2f}s  RTF {total/dur:.3f}")

        if a.ref:
            ref = open(a.ref, encoding="utf-8").read().split()
            e = edit_distance(ref, text.split())
            tot_e += e
            tot_r += len(ref)

    if a.ref and tot_r:
        print(f"\nWER = {100.0*tot_e/tot_r:.2f}%  ({tot_e}/{tot_r})")


if __name__ == "__main__":
    main()
