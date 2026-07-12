#!/usr/bin/env python3
"""
Closed-set speaker identification via CAM++ embeddings (ONNX, no torch).

We have enrollment audio for every project speaker, so this is identification,
not blind diarization: embed a segment, take the nearest enrolled speaker by
cosine, reject if the best cosine is below a threshold. See docs/DIARIZATION.md.

Model: 3D-Speaker CAM++ zh_en advanced, embedding dim 192, 80-dim fbank input,
feature_normalize_type=global-mean (per-utterance mean subtraction over time --
this cancels any constant waveform-scaling offset, so [-1,1] vs int16 does not
matter here, unlike the ASR fbank).

  python3 live_ui/speaker_id.py --separation-check   # proves features separate speakers
  python3 live_ui/speaker_id.py --enroll             # build enrolled vectors
  python3 live_ui/speaker_id.py --identify a.wav     # label a wav
"""
import glob
import json
import os

import numpy as np
import onnxruntime as ort

HERE = os.path.dirname(os.path.abspath(__file__))
VI = os.path.dirname(HERE)
MODEL = os.path.join(HERE, "models", "speaker_embedding_campplus.onnx")
ENROLL_JSON = os.path.join(HERE, "models", "enrolled_speakers.json")
SAMPLE_RATE = 16000


def fbank(samples):
    """80-dim kaldi fbank, then per-utterance global-mean normalization."""
    import kaldi_native_fbank as knf

    o = knf.FbankOptions()
    o.frame_opts.samp_freq = SAMPLE_RATE
    o.frame_opts.dither = 0.0
    o.frame_opts.snip_edges = False
    o.frame_opts.window_type = "povey"
    o.mel_opts.num_bins = 80
    f = knf.OnlineFbank(o)
    f.accept_waveform(SAMPLE_RATE, samples.astype(np.float32).tolist())
    f.input_finished()
    feats = np.stack([f.get_frame(i) for i in range(f.num_frames_ready)])
    feats = feats - feats.mean(axis=0, keepdims=True)   # global-mean (per metadata)
    return feats.astype(np.float32)


class SpeakerEmbedder:
    def __init__(self, model_path=MODEL):
        so = ort.SessionOptions()
        so.intra_op_num_threads = 1
        so.log_severity_level = 3
        self.s = ort.InferenceSession(model_path, so, providers=["CPUExecutionProvider"])
        self.name = self.s.get_inputs()[0].name

    def embed(self, samples):
        """samples: float32 [-1,1] 16 kHz mono -> L2-normalized 192-dim vector."""
        feats = fbank(samples)[None, ...]
        emb = self.s.run(None, {self.name: feats})[0][0]
        n = np.linalg.norm(emb)
        return emb / n if n > 0 else emb


def cosine(a, b):
    return float(np.dot(a, b))          # inputs already L2-normalized


# --------------------------------------------------------------- enrollment
def _read_wav_any(path):
    from transcribe_beam_wav import load_wav_any  # resamples to 16k mono [-1,1]
    return load_wav_any(path)


def enroll(speakers_dir=None, per_speaker=25, out=ENROLL_JSON):
    """One averaged, L2-normalized vector per speaker from dataset/<Spk>/."""
    import sys
    sys.path.insert(0, os.path.join(VI, "deploy", "jetson_nano"))
    speakers_dir = speakers_dir or os.path.join(VI, "dataset")
    emb = SpeakerEmbedder()
    enrolled = {}
    for spk in sorted(os.listdir(speakers_dir)):
        d = os.path.join(speakers_dir, spk)
        if not os.path.isdir(d):
            continue
        wavs = sorted(glob.glob(os.path.join(d, "*.wav")) +
                      glob.glob(os.path.join(d, "*.flac")))[:per_speaker]
        if not wavs:
            continue
        vecs = [emb.embed(_read_wav_any(w)) for w in wavs]
        v = np.mean(vecs, axis=0)
        v = v / np.linalg.norm(v)
        enrolled[spk] = v.tolist()
        print(f"  enrolled {spk:8} from {len(wavs)} clips")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(enrolled, open(out, "w"))
    print(f"wrote {len(enrolled)} speakers -> {out}")
    return enrolled


def load_enrolled(path=ENROLL_JSON):
    d = json.load(open(path))
    return {k: np.array(v, dtype=np.float32) for k, v in d.items()}


def identify(vec, enrolled, threshold=0.0):
    """Return (speaker, score). speaker is 'unknown' if best score < threshold."""
    best, bs = "unknown", -1.0
    for spk, ev in enrolled.items():
        c = cosine(vec, ev)
        if c > bs:
            best, bs = spk, c
    return (best if bs >= threshold else "unknown"), bs


# --------------------------------------------------------------- checks
def _separation_check():
    """The features are correct iff same-speaker cosine >> different-speaker.

    This is the feature-parity proof: a wrong fbank still yields vectors, but they
    do not separate speakers. Mirrors the corr(duration, word_count) check.
    """
    import sys
    sys.path.insert(0, os.path.join(VI, "deploy", "jetson_nano"))
    emb = SpeakerEmbedder()
    ds = os.path.join(VI, "dataset")
    spks = [s for s in sorted(os.listdir(ds)) if os.path.isdir(os.path.join(ds, s))]
    # 6 clips per speaker
    vecs = {}
    for s in spks:
        ws = sorted(glob.glob(os.path.join(ds, s, "*.wav")) +
                    glob.glob(os.path.join(ds, s, "*.flac")))[:6]
        vecs[s] = [emb.embed(_read_wav_any(w)) for w in ws]

    same, diff = [], []
    for s in spks:
        vs = vecs[s]
        for i in range(len(vs)):
            for j in range(i + 1, len(vs)):
                same.append(cosine(vs[i], vs[j]))
    for a in range(len(spks)):
        for b in range(a + 1, len(spks)):
            for va in vecs[spks[a]]:
                for vb in vecs[spks[b]]:
                    diff.append(cosine(va, vb))
    same, diff = np.array(same), np.array(diff)
    print(f"  speakers: {spks}")
    print(f"  same-speaker  cosine: mean={same.mean():.3f}  min={same.min():.3f}  (n={len(same)})")
    print(f"  diff-speaker  cosine: mean={diff.mean():.3f}  max={diff.max():.3f}  (n={len(diff)})")
    gap = same.mean() - diff.mean()
    # a usable margin: same-mean should sit clearly above diff-mean
    sep_ok = gap > 0.15 and same.min() > diff.mean()
    print(f"  separation gap = {gap:+.3f}   suggested threshold ~ {(same.mean()+diff.mean())/2:.3f}")
    print(f"  {'SEPARATES speakers -> features OK' if sep_ok else 'WEAK separation -> features likely wrong'}")
    assert gap > 0.15, "embeddings do not separate our speakers; check feature extraction"
    print("  separation-check OK")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--separation-check", action="store_true")
    p.add_argument("--enroll", action="store_true")
    p.add_argument("--identify", metavar="WAV")
    p.add_argument("--threshold", type=float, default=0.0)
    a = p.parse_args()
    if a.separation_check:
        _separation_check()
    elif a.enroll:
        enroll()
    elif a.identify:
        import sys
        sys.path.insert(0, os.path.join(VI, "deploy", "jetson_nano"))
        e = SpeakerEmbedder()
        who, sc = identify(e.embed(_read_wav_any(a.identify)),
                           load_enrolled(), a.threshold)
        print(f"{who}\t{sc:.3f}")
    else:
        p.print_help()
