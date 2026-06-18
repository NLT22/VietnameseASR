#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np
import sherpa_onnx
import soundfile as sf


def get_args():
    parser = argparse.ArgumentParser(
        description="Run the VietnameseASR Zipformer ONNX model with sherpa-onnx."
    )
    parser.add_argument("wav", type=Path, help="Path to a 16 kHz mono wav file.")
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "model",
        help="Directory containing encoder/decoder/joiner ONNX files and tokens.",
    )
    parser.add_argument(
        "--fp32",
        action="store_true",
        help="Use fp32 ONNX files instead of the int8 files.",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=2,
        help="ONNX Runtime CPU thread count. Jetson Nano usually works best with 2.",
    )
    parser.add_argument(
        "--decoding-method",
        choices=["greedy_search", "modified_beam_search"],
        default="modified_beam_search",
    )
    parser.add_argument(
        "--max-active-paths",
        type=int,
        default=20,
        help="Used by modified_beam_search.",
    )
    parser.add_argument(
        "--provider",
        choices=["cpu", "cuda"],
        default="cpu",
        help="ONNX Runtime provider used by sherpa-onnx.",
    )
    return parser.parse_args()


def read_wave(path: Path):
    samples, sample_rate = sf.read(path, dtype="float32", always_2d=False)
    if samples.ndim == 2:
        samples = samples.mean(axis=1)
    samples = np.ascontiguousarray(samples, dtype=np.float32)
    if sample_rate != 16000:
        raise ValueError(
            f"{path} is {sample_rate} Hz. Convert it to 16 kHz first, e.g. "
            f"ffmpeg -y -i {path} -ar 16000 -ac 1 converted.wav"
        )
    return samples, sample_rate


def make_recognizer(args):
    suffix = "" if args.fp32 else ".int8"
    model_dir = args.model_dir
    return sherpa_onnx.OfflineRecognizer.from_transducer(
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
        provider=args.provider,
        model_type="transducer",
    )


def main():
    args = get_args()
    samples, sample_rate = read_wave(args.wav)
    recognizer = make_recognizer(args)
    stream = recognizer.create_stream()
    stream.accept_waveform(sample_rate, samples)
    recognizer.decode_stream(stream)
    print(stream.result.text.strip())


if __name__ == "__main__":
    main()
