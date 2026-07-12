#!/usr/bin/env python3
"""
Incremental streaming decoder: emits a partial transcript every encoder chunk.

`jetson_beam_decode.beam_search` is already frame-synchronous -- it loops over
encoder frames carrying (B, dec_cache). Split that loop into a stepper and the
same algorithm runs live, with no accuracy compromise. Verified by the
self-check: streaming output == batch output on the same wav.

  python3 live_ui/stream_decoder.py --self-check some.wav
"""
import os
import sys

import numpy as np

_JETSON = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "..", "deploy", "jetson_nano")
sys.path.insert(0, os.path.abspath(_JETSON))

from jetson_beam_decode import (  # noqa: E402
    BLANK, NEG_INF, OnnxTransducer, detok, load_tokens, log_softmax, logaddexp,
)

SAMPLE_RATE = 16000


def fbank_opts():
    """Exactly lhotse's Fbank defaults -- must match training or WER collapses."""
    import kaldi_native_fbank as knf

    o = knf.FbankOptions()
    o.frame_opts.samp_freq = SAMPLE_RATE
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
    return o


class IncrementalBeam:
    """One frame of classic RNN-T beam search at a time. Multi-symbol per frame."""

    def __init__(self, model, beam=4):
        self.m = model
        self.beam = beam
        self.ctx = model.context_size
        self.B = {tuple([BLANK] * self.ctx): 0.0}
        self.dec_cache = {}

    def step(self, enc_t):
        m, ctx, beam = self.m, self.ctx, self.beam
        A, self.B = self.B, {}
        B = self.B
        joint_cache = {}
        enc_t = enc_t[None, :]
        while True:
            y = max(A, key=A.get)
            lp = A.pop(y)
            dk = y[-ctx:]
            if dk not in self.dec_cache:
                self.dec_cache[dk] = m.decoder(dk)
            if dk not in joint_cache:
                joint_cache[dk] = log_softmax(m.joiner(enc_t, self.dec_cache[dk])[0])
            logp = joint_cache[dk]

            B[y] = logaddexp(B.get(y, NEG_INF), lp + float(logp[BLANK]))

            k = min(beam + 1, logp.shape[0])
            for i in np.argpartition(-logp, k - 1)[:k]:
                i = int(i)
                if i == BLANK:
                    continue
                ny = y + (i,)
                A[ny] = logaddexp(A.get(ny, NEG_INF), lp + float(logp[i]))
            if not A:
                break
            a_best = max(A.values())
            if sum(1 for v in B.values() if v > a_best) >= beam:
                break
        self.B = dict(sorted(B.items(), key=lambda kv: -kv[1])[:beam])

    def best(self):
        return list(max(self.B, key=self.B.get)[self.ctx:])


class StreamingSession:
    """Feed audio as it arrives; read the partial transcript after each chunk."""

    def __init__(self, model_dir, beam=4):
        import kaldi_native_fbank as knf

        self.m = OnnxTransducer(model_dir)
        self.m.reset()
        self.id2tok = load_tokens(os.path.join(model_dir, "tokens.txt"))
        self.fb = knf.OnlineFbank(fbank_opts())
        self.beam = IncrementalBeam(self.m, beam)
        self.feats = []          # frames seen so far
        self.start = 0           # index of the next encoder window
        self.chunks = 0
        self.finished = False

    # -- encoder ---------------------------------------------------------
    def _run_window(self, window):
        feed = {self.m.enc_in[0]: window[None, ...]}
        for name, st in zip(self.m.enc_in[1:], self.m.states):
            feed[name] = st
        res = self.m.enc.run(self.m.enc_out, feed)
        self.m.states = res[1:]
        self.chunks += 1
        for t in range(res[0].shape[1]):
            self.beam.step(res[0][0, t])

    def _drain(self):
        seg, hop = self.m.segment, self.m.offset
        while self.start + seg <= len(self.feats):
            self._run_window(np.stack(self.feats[self.start:self.start + seg]))
            self.start += hop

    def accept_waveform(self, samples):
        """samples: float32 in [-1, 1] at 16 kHz. Returns partial text."""
        if self.finished:
            return self.text()
        self.fb.accept_waveform(SAMPLE_RATE, samples.tolist())
        while self.fb.num_frames_ready > len(self.feats):
            self.feats.append(self.fb.get_frame(len(self.feats)))
        self._drain()
        return self.text()

    def finish(self):
        """Flush the tail. Mirrors the padding in OnnxTransducer.encode()."""
        if self.finished:
            return self.text()
        self.fb.input_finished()
        while self.fb.num_frames_ready > len(self.feats):
            self.feats.append(self.fb.get_frame(len(self.feats)))
        self._drain()

        seg, hop = self.m.segment, self.m.offset
        rem = len(self.feats) - self.start
        # Window k covers [hop*k, hop*k + seg); frames past start+(seg-hop) are
        # unseen. Pad and run one last window iff any remain (or none ever ran).
        if self.chunks == 0 or rem > seg - hop:
            tail = self.feats[self.start:]
            if tail:
                w = np.stack(tail)
                if w.shape[0] < seg:
                    w = np.concatenate(
                        [w, np.zeros((seg - w.shape[0], w.shape[1]), np.float32)], 0)
                self._run_window(w)
        self.finished = True
        return self.text()

    def text(self):
        return detok(self.beam.best(), self.id2tok)


# ------------------------------------------------------------------ check
def _self_check(wav):
    """Streaming must equal batch decode on the same audio."""
    from jetson_beam_decode import beam_search
    from transcribe_beam_wav import load_wav_any

    md = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                      "deploy", "jetson_nano", "model_medium_epoch30_avg10")
    md = os.path.abspath(md)
    pcm = load_wav_any(wav)

    m = OnnxTransducer(md)
    id2tok = load_tokens(os.path.join(md, "tokens.txt"))
    from jetson_asr import compute_fbank
    batch = detok(beam_search(m, m.encode(compute_fbank(pcm)), 4), id2tok)

    s = StreamingSession(md, beam=4)
    step = int(0.25 * SAMPLE_RATE)
    for i in range(0, len(pcm), step):
        s.accept_waveform(pcm[i:i + step])
    stream = s.finish()

    print(f"  batch    : {batch}")
    print(f"  streaming: {stream}")
    assert stream == batch, "streaming diverged from batch decode"
    print(f"  self-check OK ({s.chunks} chunks)")


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--self-check", metavar="WAV")
    a = p.parse_args()
    if a.self_check:
        _self_check(a.self_check)
    else:
        p.print_help()
