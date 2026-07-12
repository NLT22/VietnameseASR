#!/usr/bin/env python3
"""
Silero VAD (ONNX, no torch) + an endpointer that segments continuous speech.

The ASR model was trained on single sentences. Without endpointing the live UI
accumulates one ever-growing hypothesis and drifts out of distribution. Silero
tells us where a sentence ends so we can finalize it and start a fresh session.

Model: silero_vad_16k_op15.onnx, MIT (snakers4/silero-vad), vendored in models/.

  python3 live_ui/vad.py --self-check speech.wav
"""
import os

import numpy as np
import onnxruntime as ort

SAMPLE_RATE = 16000
FRAME = 512          # silero v5 @16k consumes exactly 512 new samples per call
CONTEXT = 64         # ...prepended with the previous 64 samples -> 576 in total
FRAME_MS = FRAME / SAMPLE_RATE * 1000.0   # 32 ms

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_VAD = os.path.join(HERE, "models", "silero_vad.onnx")


class SileroVAD:
    """Stateful per-frame speech probability."""

    def __init__(self, model_path=DEFAULT_VAD):
        so = ort.SessionOptions()
        so.intra_op_num_threads = 1
        so.log_severity_level = 3          # silence initializer warnings
        self.s = ort.InferenceSession(model_path, so,
                                      providers=["CPUExecutionProvider"])
        self.sr = np.array(SAMPLE_RATE, dtype=np.int64)
        self.reset()

    def reset(self):
        self.state = np.zeros((2, 1, 128), dtype=np.float32)
        self.ctx = np.zeros(CONTEXT, dtype=np.float32)

    def __call__(self, frame):
        """frame: float32 [-1,1], exactly FRAME samples. Returns P(speech).

        The graph wants CONTEXT+FRAME samples. Feeding it a bare 512 does not
        error -- it silently returns ~0 for every frame, i.e. "never speech".
        """
        if frame.shape[0] != FRAME:
            raise ValueError(f"expected {FRAME} samples, got {frame.shape[0]}")
        frame = frame.astype(np.float32)
        x = np.concatenate([self.ctx, frame])[None, :]
        out, self.state = self.s.run(
            None, {"input": x, "state": self.state, "sr": self.sr})
        self.ctx = frame[-CONTEXT:]
        return float(out[0][0])


class Endpointer:
    """Turn per-frame probabilities into speech-start / speech-end events.

    Hysteresis: enter speech above `threshold`, leave below `threshold - 0.15`.
    A single threshold chatters on breaths and fricatives.
    """

    def __init__(self, threshold=0.5, min_speech_ms=250, min_silence_ms=700,
                 speech_pad_ms=200):
        self.hi = threshold
        self.lo = max(0.05, threshold - 0.15)
        self.min_speech = min_speech_ms / FRAME_MS
        self.min_silence = min_silence_ms / FRAME_MS
        self.pad = speech_pad_ms / FRAME_MS
        self.reset()

    def reset(self):
        self.speaking = False
        self.run_speech = 0.0
        self.run_silence = 0.0

    def update(self, prob):
        """Returns 'start', 'end', or None."""
        if not self.speaking:
            if prob >= self.hi:
                self.run_speech += 1
                if self.run_speech >= self.min_speech:
                    self.speaking = True
                    self.run_silence = 0.0
                    return "start"
            else:
                self.run_speech = 0.0
        else:
            if prob < self.lo:
                self.run_silence += 1
                if self.run_silence >= self.min_silence + self.pad:
                    self.speaking = False
                    self.run_speech = 0.0
                    return "end"
            else:
                self.run_silence = 0.0
        return None


def _self_check(wav):
    import wave

    w = wave.open(wav)
    assert w.getframerate() == SAMPLE_RATE, "need 16 kHz mono"
    pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    pcm = pcm.astype(np.float32) / 32768.0

    v = SileroVAD()
    probs = [v(pcm[i:i + FRAME]) for i in range(0, len(pcm) - FRAME + 1, FRAME)]
    speech = sum(p >= 0.5 for p in probs)
    print(f"  {wav}")
    print(f"    frames={len(probs)}  speech={speech} ({100*speech/len(probs):.0f}%)"
          f"  mean_p={np.mean(probs):.3f}")
    # Dropping the 64-sample context makes every prob ~0 without raising, so a
    # bare "did it run" check would pass. Demand it actually finds the speech.
    assert speech > 0.3 * len(probs), "speech wav should be mostly speech"
    assert np.mean(probs) > 0.3, "suspiciously low probs -- lost the VAD context?"

    # digital silence must not trip the VAD
    v.reset()
    sil = np.zeros(FRAME, dtype=np.float32)
    sp = [v(sil) for _ in range(30)]
    print(f"    silence mean_p={np.mean(sp):.4f}")
    assert max(sp) < 0.5, "silence must not be detected as speech"

    # endpointer must fire exactly one start on speech-then-silence
    v.reset()
    ep = Endpointer()
    events = []
    for i in range(0, len(pcm) - FRAME + 1, FRAME):
        e = ep.update(v(pcm[i:i + FRAME]))
        if e:
            events.append(e)
    for _ in range(60):                       # ~2 s of trailing silence
        e = ep.update(v(sil))
        if e:
            events.append(e)
    print(f"    events={events}")
    assert events[:1] == ["start"], f"expected a start, got {events}"
    assert "end" in events, "trailing silence must produce an end"
    print("  self-check OK")


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--self-check", metavar="WAV", required=True)
    a = p.parse_args()
    _self_check(a.self_check)
