#!/usr/bin/env python3
"""
Score models on the HELD-OUT SPEAKER set (known sentences, unseen voices).

The project's usual WER is measured on the exact recordings used for training,
so it is blind to speaker generalization -- the thing that decides whether the
mic UI works for a real person. This scores the same sentences spoken by voices
that never appear in training.

Usage:
  python3 eval_heldout_speaker.py --model-dir deploy/jetson_nano/model_X [--limit N]
"""
import argparse
import csv
import os
import sys
import time

VI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(VI, "deploy", "jetson_nano"))

from jetson_beam_decode import (  # noqa: E402
    OnnxTransducer, beam_search, greedy, load_tokens, detok, edit_distance,
)
from transcribe_beam_wav import load_wav_any  # noqa: E402
from jetson_asr import compute_fbank  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--data", default=os.path.join(VI, "heldout_speaker_eval"))
    ap.add_argument("--beam", type=int, default=4)
    ap.add_argument("--method", choices=["beam_search", "greedy"], default="beam_search")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    md = a.model_dir if os.path.isabs(a.model_dir) else os.path.join(VI, a.model_dir)
    m = OnnxTransducer(md)
    id2tok = load_tokens(os.path.join(md, "tokens.txt"))

    rows = list(csv.DictReader(open(os.path.join(a.data, "manifest.tsv")), delimiter="\t"))
    if a.limit:
        rows = rows[: a.limit]

    per_voice = {}
    tot_e = tot_r = 0
    t0 = time.time()
    for r in rows:
        wav = os.path.join(a.data, r["utt_id"] + ".wav")
        if not os.path.isfile(wav):
            continue
        feats = compute_fbank(load_wav_any(wav))
        enc = m.encode(feats)
        ids = beam_search(m, enc, a.beam) if a.method == "beam_search" else greedy(m, enc)
        hyp = detok(ids, id2tok).split()
        ref = r["text"].split()
        e = edit_distance(ref, hyp)
        tot_e += e
        tot_r += len(ref)
        v = per_voice.setdefault(r["voice"], [0, 0])
        v[0] += e
        v[1] += len(ref)

    dt = time.time() - t0
    print(f"model: {os.path.basename(md)}   method={a.method} beam={a.beam}")
    for v, (e, n) in sorted(per_voice.items()):
        print(f"  {v:<12} WER {100.0*e/max(n,1):6.2f}%   ({e}/{n})")
    print(f"  {'OVERALL':<12} WER {100.0*tot_e/max(tot_r,1):6.2f}%   ({tot_e}/{tot_r})"
          f"   {dt/max(len(rows),1):.2f}s/utt")


if __name__ == "__main__":
    main()
