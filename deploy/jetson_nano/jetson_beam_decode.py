#!/usr/bin/env python3
"""
Standalone classic RNN-T beam_search decoder for Jetson Nano.

Dependencies: onnxruntime + numpy ONLY.
No torch, no k2, no lhotse, no sentencepiece (detokenizes from tokens.txt).

Why: sherpa-onnx implements only greedy/modified_beam_search, both capped at
one symbol per encoder frame -> ~9.2% WER on this model. Classic beam_search
may emit several symbols per frame and reaches ~2.4% WER at similar cost.

Input: precomputed 80-dim fbank features (.npy, shape (T,80)) + reference text,
so this isolates the decoder from feature extraction.

Usage:
  python3 jetson_beam_decode.py --model-dir model --data feats --beam 4
"""
import argparse
import glob
import math
import os
import time

import numpy as np
import onnxruntime as ort

BLANK = 0
NEG_INF = float("-inf")


def log_softmax(x):
    m = x.max()
    e = np.exp(x - m)
    return (x - m) - math.log(e.sum())


def logaddexp(a, b):
    if a == NEG_INF:
        return b
    if b == NEG_INF:
        return a
    m = a if a > b else b
    return m + math.log(math.exp(a - m) + math.exp(b - m))


class OnnxTransducer:
    def __init__(self, model_dir):
        so = ort.SessionOptions()
        so.inter_op_num_threads = 1
        so.intra_op_num_threads = 2  # Nano has 4 cores; leave headroom
        p = ["CPUExecutionProvider"]
        self.enc = ort.InferenceSession(f"{model_dir}/encoder.int8.onnx", so, providers=p)
        self.dec = ort.InferenceSession(f"{model_dir}/decoder.int8.onnx", so, providers=p)
        self.joi = ort.InferenceSession(f"{model_dir}/joiner.int8.onnx", so, providers=p)

        self.enc_in = [i.name for i in self.enc.get_inputs()]
        self.enc_out = [o.name for o in self.enc.get_outputs()]
        meta = self.enc.get_modelmeta().custom_metadata_map
        il = lambda s: [int(v) for v in meta[s].split(",")]
        self.segment = int(meta["T"])
        self.offset = int(meta["decode_chunk_len"])
        self.context_size = 2
        self._cfg = (
            il("num_encoder_layers"), il("encoder_dims"), il("cnn_module_kernels"),
            il("left_context_len"), il("query_head_dims"), il("value_head_dims"),
            il("num_heads"),
        )
        self.reset()

    def reset(self, batch=1):
        """Zero the 74 encoder state tensors (same order as ONNX inputs[1:])."""
        nl, ed, ck, lc, qh, vh, nh = self._cfg
        s = []
        for i in range(len(nl)):
            key_dim = qh[i] * nh[i]
            emb = ed[i]
            nonlin = 3 * emb // 4
            val_dim = vh[i] * nh[i]
            pad = ck[i] // 2
            for _ in range(nl[i]):
                s.append(np.zeros((lc[i], batch, key_dim), dtype=np.float32))
                s.append(np.zeros((1, batch, lc[i], nonlin), dtype=np.float32))
                s.append(np.zeros((lc[i], batch, val_dim), dtype=np.float32))
                s.append(np.zeros((lc[i], batch, val_dim), dtype=np.float32))
                s.append(np.zeros((batch, emb, pad), dtype=np.float32))
                s.append(np.zeros((batch, emb, pad), dtype=np.float32))
        s.append(np.zeros((batch, 128, 3, 19), dtype=np.float32))
        s.append(np.zeros((batch,), dtype=np.int64))
        self.states = s

    def encode(self, feats):
        """Stream the causal encoder over the whole utterance; return (T',512)."""
        self.reset()
        x = feats.astype(np.float32)
        n = x.shape[0]
        if n < self.segment:
            pad = self.segment - n
        else:
            rem = (n - self.segment) % self.offset
            pad = 0 if rem == 0 else self.offset - rem
        if pad:
            x = np.concatenate([x, np.zeros((pad, x.shape[1]), np.float32)], 0)

        outs = []
        start = 0
        while start + self.segment <= x.shape[0]:
            chunk = x[start:start + self.segment][None, ...]
            feed = {self.enc_in[0]: chunk}
            for name, st in zip(self.enc_in[1:], self.states):
                feed[name] = st
            res = self.enc.run(self.enc_out, feed)
            outs.append(res[0])
            self.states = res[1:]
            start += self.offset
        return np.concatenate(outs, axis=1)[0]  # (T',512)

    def decoder(self, ctx_tokens):
        y = np.array([list(ctx_tokens)], dtype=np.int64)
        return self.dec.run(None, {self.dec.get_inputs()[0].name: y})[0]

    def joiner(self, enc_t, dec_out):
        n = [i.name for i in self.joi.get_inputs()]
        return self.joi.run(None, {n[0]: enc_t, n[1]: dec_out})[0]


def beam_search(m, enc, beam=4, max_sym=20000):
    """Classic RNN-T beam search: may emit MULTIPLE symbols per frame."""
    ctx = m.context_size
    B = {tuple([BLANK] * ctx): 0.0}
    dec_cache = {}
    sym = 0
    for t in range(enc.shape[0]):
        if sym >= max_sym:
            break
        A, B = B, {}
        joint_cache = {}
        enc_t = enc[t:t + 1]
        while True:
            y = max(A, key=A.get)
            lp = A.pop(y)
            dk = y[-ctx:]
            if dk not in dec_cache:
                dec_cache[dk] = m.decoder(dk)
            if dk not in joint_cache:
                joint_cache[dk] = log_softmax(m.joiner(enc_t, dec_cache[dk])[0])
            logp = joint_cache[dk]

            B[y] = logaddexp(B.get(y, NEG_INF), lp + float(logp[BLANK]))

            k = min(beam + 1, logp.shape[0])
            for i in np.argpartition(-logp, k - 1)[:k]:
                i = int(i)
                if i == BLANK:
                    continue
                ny = y + (i,)
                A[ny] = logaddexp(A.get(ny, NEG_INF), lp + float(logp[i]))
                sym += 1
            if not A:
                break
            a_best = max(A.values())
            if sum(1 for v in B.values() if v > a_best) >= beam:
                break
        B = dict(sorted(B.items(), key=lambda kv: -kv[1])[:beam])
    best = max(B, key=B.get)
    return list(best[ctx:])


def greedy(m, enc):
    ctx = m.context_size
    hyp = [BLANK] * ctx
    d = m.decoder(hyp)
    for t in range(enc.shape[0]):
        y = int(m.joiner(enc[t:t + 1], d)[0].argmax())
        if y != BLANK:
            hyp.append(y)
            d = m.decoder(hyp[-ctx:])
    return hyp[ctx:]


def load_tokens(path):
    id2tok = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if len(parts) == 2:
                id2tok[int(parts[1])] = parts[0]
    return id2tok


def detok(ids, id2tok):
    s = "".join(id2tok.get(i, "") for i in ids)
    return s.replace("▁", " ").strip()


def edit_distance(r, h):
    d = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(h) + 1):
            cur = d[j]
            d[j] = min(d[j] + 1, d[j - 1] + 1, prev + (r[i - 1] != h[j - 1]))
            prev = cur
    return d[len(h)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="model")
    ap.add_argument("--data", default="feats")
    ap.add_argument("--beam", type=int, default=4)
    ap.add_argument("--method", choices=["beam_search", "greedy"], default="beam_search")
    a = ap.parse_args()

    m = OnnxTransducer(a.model_dir)
    id2tok = load_tokens(f"{a.model_dir}/tokens.txt")
    print("segment=%d offset=%d ctx=%d" % (m.segment, m.offset, m.context_size))

    npys = sorted(glob.glob(os.path.join(a.data, "*.npy")))
    tot_e = tot_r = 0
    t_enc = t_dec = 0.0
    for p in npys:
        feats = np.load(p)
        ref = open(p.replace(".npy", ".txt"), encoding="utf-8").read().split()
        t0 = time.time(); enc = m.encode(feats); t1 = time.time()
        ids = beam_search(m, enc, a.beam) if a.method == "beam_search" else greedy(m, enc)
        t2 = time.time()
        t_enc += t1 - t0; t_dec += t2 - t1
        hyp = detok(ids, id2tok).split()
        tot_e += edit_distance(ref, hyp); tot_r += len(ref)
        print("  %-22s %5.1fs audio  enc %4.1fs  dec %4.1fs" %
              (os.path.basename(p), feats.shape[0] / 100.0, t1 - t0, t2 - t1))

    n = len(npys)
    print("\nmethod=%s beam=%d utts=%d" % (a.method, a.beam, n))
    print("WER=%.2f%%  (%d/%d)" % (100.0 * tot_e / max(tot_r, 1), tot_e, tot_r))
    print("encode %.2fs/utt   decode %.2fs/utt   total %.2fs/utt" %
          (t_enc / n, t_dec / n, (t_enc + t_dec) / n))


if __name__ == "__main__":
    main()
