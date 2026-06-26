#!/usr/bin/env python3
import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

# ponytail: the pip sherpa_onnx wheel on the Jetson is CPU-only (--provider cuda
# silently falls back to CPU). For GPU we reuse the GPU-built C++ sherpa-onnx
# binary, same invocation as eval_sherpa_binary.py. Defaults below are the Jetson
# build paths; override with --bin/--lib elsewhere.
# CEILING: the binary pays ~25s CUDA engine init PER call. Worth it only for
# batch/amortized use; for the per-request live UI, CPU (~7s) beats CUDA (~30s)
# on the Nano, so keep the UI on --provider cpu. Upgrade path: a persistent GPU
# recognizer process (or a GPU-enabled python sherpa_onnx wheel) if the UI must
# run on GPU.
BIN_DEFAULT = "/home/thayhoang/sherpa-onnx-gpu-build/sherpa-onnx/build-gpu/bin/sherpa-onnx"
LIB_DEFAULT = (
    "/home/thayhoang/sherpa-onnx-gpu-build/sherpa-onnx/build-gpu/lib:"
    "/home/thayhoang/sherpa-onnx-gpu-build/sherpa-onnx/build-gpu/_deps/onnxruntime-src/lib"
)


def get_args():
    parser = argparse.ArgumentParser(
        description="Run the VietnameseASR streaming Zipformer ONNX model."
    )
    parser.add_argument("wav", type=Path)
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "model_streaming",
    )
    parser.add_argument("--fp32", action="store_true")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument(
        "--decoding-method",
        choices=["greedy_search", "modified_beam_search"],
        default="modified_beam_search",
    )
    parser.add_argument("--max-active-paths", type=int, default=4)
    parser.add_argument(
        "--provider",
        choices=["cpu", "cuda", "trt", "tensorrt"],
        default="cpu",
    )
    parser.add_argument("--bin", default=BIN_DEFAULT, help="GPU sherpa-onnx C++ binary")
    parser.add_argument("--lib", default=LIB_DEFAULT, help="LD_LIBRARY_PATH for --bin")
    return parser.parse_args()


def resample_linear(samples, src_rate, dst_rate=16000):
    if src_rate == dst_rate:
        return samples
    if samples.size == 0:
        return samples
    duration = samples.shape[0] / float(src_rate)
    dst_len = max(1, int(round(duration * dst_rate)))
    src_x = np.linspace(0.0, duration, num=samples.shape[0], endpoint=False)
    dst_x = np.linspace(0.0, duration, num=dst_len, endpoint=False)
    return np.interp(dst_x, src_x, samples).astype(np.float32)


def read_wave(path: Path):
    samples, sample_rate = sf.read(path, dtype="float32", always_2d=False)
    if samples.ndim == 2:
        samples = samples.mean(axis=1)
    samples = np.ascontiguousarray(samples, dtype=np.float32)
    samples = resample_linear(samples, sample_rate)
    return np.ascontiguousarray(samples, dtype=np.float32), 16000


def transcribe_with_gpu_binary(args, samples, sample_rate):
    """GPU path: run the C++ sherpa-onnx binary (real CUDA/TRT support)."""
    suffix = ".onnx" if args.fp32 else ".int8.onnx"
    md = args.model_dir
    provider = "trt" if args.provider == "tensorrt" else args.provider
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
    sf.write(tmp, samples, sample_rate, subtype="PCM_16")
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = args.lib + (
        ":" + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else ""
    )
    cmd = [
        args.bin,
        f"--tokens={md / 'tokens.txt'}",
        f"--encoder={md / ('encoder' + suffix)}",
        f"--decoder={md / ('decoder' + suffix)}",
        f"--joiner={md / ('joiner' + suffix)}",
        "--modeling-unit=bpe",
        f"--bpe-vocab={md / 'bpe.vocab'}",
        f"--provider={provider}",
        f"--num-threads={args.threads}",
        f"--decoding-method={args.decoding_method}",
        f"--max-active-paths={args.max_active_paths}",
        "--model-type=zipformer2",
        tmp,
    ]
    try:
        p = subprocess.run(
            cmd, env=env, universal_newlines=True,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        )
    finally:
        os.unlink(tmp)
    text = ""
    for line in p.stdout.splitlines():
        line = line.strip()
        if line.startswith("{") and '"text"' in line:
            try:
                text = json.loads(line).get("text", "")
            except Exception:
                pass
    if p.returncode != 0 and not text:
        sys.stderr.write(p.stdout)
        raise SystemExit(f"sherpa-onnx GPU binary failed (rc={p.returncode})")
    return text


def make_recognizer(args):
    import sherpa_onnx  # ponytail: CPU path only; GPU path uses the C++ binary

    suffix = "" if args.fp32 else ".int8"
    provider = "trt" if args.provider == "tensorrt" else args.provider
    model_dir = args.model_dir
    return sherpa_onnx.OnlineRecognizer.from_transducer(
        encoder=str(model_dir / f"encoder{suffix}.onnx"),
        decoder=str(model_dir / f"decoder{suffix}.onnx"),
        joiner=str(model_dir / f"joiner{suffix}.onnx"),
        tokens=str(model_dir / "tokens.txt"),
        num_threads=args.threads,
        sample_rate=16000,
        feature_dim=80,
        dither=0.0,
        decoding_method=args.decoding_method,
        max_active_paths=args.max_active_paths,
        modeling_unit="bpe",
        bpe_vocab=str(model_dir / "bpe.vocab"),
        provider=provider,
        model_type="zipformer2",
    )


def main():
    args = get_args()
    samples, sample_rate = read_wave(args.wav)
    if args.provider in ("cuda", "trt", "tensorrt"):
        print(transcribe_with_gpu_binary(args, samples, sample_rate).strip())
        return
    recognizer = make_recognizer(args)
    stream = recognizer.create_stream()
    stream.accept_waveform(sample_rate, samples)
    stream.input_finished()
    while recognizer.is_ready(stream):
        recognizer.decode_stream(stream)
    print(recognizer.get_result(stream).strip())


if __name__ == "__main__":
    main()
