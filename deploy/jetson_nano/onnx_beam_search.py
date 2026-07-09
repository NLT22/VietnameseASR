#!/usr/bin/env python3
"""
Classic RNN-T beam_search on top of the exported ONNX encoder/decoder/joiner.

Why this exists: sherpa-onnx only implements greedy_search and
modified_beam_search, both of which emit at most ONE symbol per encoder frame.
On this memorization-style model that costs ~9.2% WER (mostly deletions),
while icefall's classic beam_search -- which may emit MULTIPLE symbols per
frame -- reaches 0.52%. This script reproduces that decode path using only
onnxruntime, so it can eventually run on the Jetson.

Features are read from the precomputed lhotse fbank cuts (identical to what
icefall's decode.py consumed), so this isolates the decoder rather than
re-implementing feature extraction.

Usage:
  python3 onnx_beam_search.py --num-utts 20 --beam 4
  python3 onnx_beam_search.py --num-utts 20 --beam 4 --method greedy   # baseline
"""
import argparse
import importlib.util
import math
import sys
import time
import types

import numpy as np
import torch

VI = "/media/pc/c88ba509-53f0-4c97-9e44-e33483754b08/icefall/egs/VietnameseASR"
DEFAULT_MODEL = f"{VI}/deploy/jetson_nano/model_streaming_u20_epoch55_avg10"
DEFAULT_CUTS = f"{VI}/fbank_x10_matched/test_cuts.jsonl.gz"

NEG_INF = float("-inf")


def _load_onnx_model_class():
    """Import icefall's OnnxModel (handles the 74 encoder state tensors).

    The module imports kaldifeat at top level for its own feature extractor,
    which we don't use -- stub it so the import succeeds.
    """
    if "kaldifeat" not in sys.modules:
        fake = types.ModuleType("kaldifeat")
        fake.FbankOptions = object
        fake.OnlineFbank = object
        fake.OnlineFeature = object
        sys.modules["kaldifeat"] = fake
    path = f"{VI}/ASR/zipformer/onnx_pretrained-streaming.py"
    spec = importlib.util.spec_from_file_location("_onnx_streaming", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.OnnxModel


def logaddexp(a: float, b: float) -> float:
    if a == NEG_INF:
        return b
    if b == NEG_INF:
        return a
    m = max(a, b)
    return m + math.log(math.exp(a - m) + math.exp(b - m))


def encode_utterance(model, feats: np.ndarray) -> torch.Tensor:
    """Stream the causal encoder over the whole utterance, collect all frames.

    Record-then-decode: we run the streaming encoder chunk-by-chunk (so the
    causal/left-context behaviour matches training) but accumulate every output
    frame, then decode offline over the full sequence.
    """
    model.init_encoder_states(1)  # reset per utterance
    segment, offset = model.segment, model.offset

    x = torch.from_numpy(feats).float()  # (T, 80)
    # pad tail so the final window is complete
    if x.shape[0] < segment:
        pad = segment - x.shape[0]
    else:
        rem = (x.shape[0] - segment) % offset
        pad = 0 if rem == 0 else (offset - rem)
    if pad:
        x = torch.cat([x, torch.zeros(pad, x.shape[1])], dim=0)

    outs = []
    start = 0
    while start + segment <= x.shape[0]:
        chunk = x[start : start + segment].unsqueeze(0)  # (1, segment, 80)
        outs.append(model.run_encoder(chunk))  # (1, t', 512)
        start += offset
    return torch.cat(outs, dim=1)[0]  # (T', 512)


def greedy_decode(model, enc: torch.Tensor, context_size: int):
    """max-sym-per-frame=1, same limit sherpa-onnx has. Baseline for comparison."""
    blank = 0
    hyp = [blank] * context_size
    dec = model.run_decoder(torch.tensor([hyp], dtype=torch.int64))
    for t in range(enc.shape[0]):
        logit = model.run_joiner(enc[t : t + 1], dec).squeeze(0)
        y = int(logit.argmax().item())
        if y != blank:
            hyp.append(y)
            dec = model.run_decoder(
                torch.tensor([hyp[-context_size:]], dtype=torch.int64)
            )
    return hyp[context_size:]


def beam_search(model, enc: torch.Tensor, context_size: int, beam: int = 4,
                max_sym_per_utt: int = 20000):
    """Classic RNN-T beam search (Graves Alg.1), mirroring icefall's beam_search.

    Crucially this may emit MULTIPLE symbols at the same encoder frame -- that
    is the difference from greedy / modified_beam_search.
    """
    blank = 0
    T = enc.shape[0]
    init = tuple([blank] * context_size)
    B = {init: 0.0}
    sym_per_utt = 0

    dec_cache = {}   # last-context tokens -> decoder_out
    for t in range(T):
        if sym_per_utt >= max_sym_per_utt:
            break
        A, B = B, {}
        joint_cache = {}
        enc_t = enc[t : t + 1]  # (1, 512)

        while True:
            y_star = max(A, key=A.get)
            lp = A.pop(y_star)

            dkey = y_star[-context_size:]
            if dkey not in dec_cache:
                dec_cache[dkey] = model.run_decoder(
                    torch.tensor([list(dkey)], dtype=torch.int64)
                )
            dec_out = dec_cache[dkey]

            if dkey not in joint_cache:
                logit = model.run_joiner(enc_t, dec_out).squeeze(0)
                joint_cache[dkey] = torch.log_softmax(logit.float(), dim=-1)
            logp = joint_cache[dkey]

            # blank -> stays at this frame's output set B
            B[y_star] = logaddexp(B.get(y_star, NEG_INF), lp + float(logp[blank]))

            # non-blank -> re-enters A, can expand again at the SAME frame
            topv, topi = torch.topk(logp, min(beam + 1, logp.numel()))
            for v, i in zip(topv.tolist(), topi.tolist()):
                if i == blank:
                    continue
                ny = y_star + (i,)
                A[ny] = logaddexp(A.get(ny, NEG_INF), lp + v)
                sym_per_utt += 1

            if not A:
                break
            a_best = max(A.values())
            if sum(1 for v in B.values() if v > a_best) >= beam:
                break

        B = dict(sorted(B.items(), key=lambda kv: -kv[1])[:beam])

    best = max(B, key=B.get)
    return list(best[context_size:])


def wer(ref_words, hyp_words):
    """Levenshtein edit distance on words -> (errors, ref_len)."""
    m, n = len(ref_words), len(hyp_words)
    d = list(range(n + 1))
    for i in range(1, m + 1):
        prev, d[0] = d[0], i
        for j in range(1, n + 1):
            cur = d[j]
            d[j] = min(d[j] + 1, d[j - 1] + 1,
                       prev + (ref_words[i - 1] != hyp_words[j - 1]))
            prev = cur
    return d[n], m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default=DEFAULT_MODEL)
    ap.add_argument("--cuts", default=DEFAULT_CUTS)
    ap.add_argument("--num-utts", type=int, default=20)
    ap.add_argument("--beam", type=int, default=4)
    ap.add_argument("--method", choices=["beam_search", "greedy"], default="beam_search")
    ap.add_argument("--int8", action="store_true", default=True)
    ap.add_argument("--fp32", dest="int8", action="store_false")
    args = ap.parse_args()

    import sentencepiece as spm
    from lhotse import load_manifest_lazy

    suffix = "int8.onnx" if args.int8 else "onnx"
    OnnxModel = _load_onnx_model_class()
    model = OnnxModel(
        f"{args.model_dir}/encoder.{suffix}",
        f"{args.model_dir}/decoder.{suffix}",
        f"{args.model_dir}/joiner.{suffix}",
    )
    sp = spm.SentencePieceProcessor()
    sp.load(f"{args.model_dir}/bpe.model")
    ctx = model.context_size

    cuts = load_manifest_lazy(args.cuts)
    tot_err = tot_ref = 0
    t0 = time.time()
    n = 0
    for cut in cuts:
        if n >= args.num_utts:
            break
        feats = cut.load_features()
        enc = encode_utterance(model, feats)
        if args.method == "greedy":
            ids = greedy_decode(model, enc, ctx)
        else:
            ids = beam_search(model, enc, ctx, beam=args.beam)
        hyp = sp.decode(ids).split()
        ref = cut.supervisions[0].text.split()
        e, r = wer(ref, hyp)
        tot_err += e
        tot_ref += r
        n += 1
        if n <= 2:
            print(f"[{cut.id}]\n  ref: {' '.join(ref[:12])} ...\n  hyp: {' '.join(hyp[:12])} ...")
    dt = time.time() - t0
    print(f"\nmethod={args.method} beam={args.beam} precision={'int8' if args.int8 else 'fp32'}")
    print(f"utts={n}  WER={100.0*tot_err/max(tot_ref,1):.2f}%  ({tot_err}/{tot_ref})")
    print(f"time={dt:.1f}s  ({dt/max(n,1):.2f}s/utt)")


if __name__ == "__main__":
    main()
