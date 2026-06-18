#!/usr/bin/env python3
import argparse
import csv
import json
import platform
import resource
import statistics
import subprocess
import time
from pathlib import Path

import numpy as np
import sherpa_onnx
import soundfile as sf


def get_args():
    parser = argparse.ArgumentParser(
        description="Evaluate VietnameseASR streaming ONNX runtime speed and WER."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--audio-root", type=Path, required=True)
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "model_streaming",
    )
    parser.add_argument("--fp32", action="store_true", help="Use fp32 ONNX files.")
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
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--trt-fp16", type=int, default=1)
    parser.add_argument("--trt-cache", type=Path, default=None)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--output-md", type=Path, default=None)
    return parser.parse_args()


def normalize_text(text: str) -> str:
    return " ".join(text.strip().lower().split())


def edit_distance(ref, hyp):
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, start=1):
        cur = [i] + [0] * len(hyp)
        for j, h in enumerate(hyp, start=1):
            cur[j] = min(
                prev[j] + 1,
                cur[j - 1] + 1,
                prev[j - 1] + (0 if r == h else 1),
            )
        prev = cur
    return prev[-1]


def read_manifest(path: Path, audio_root: Path, limit: int):
    rows = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            audio_path = Path(row["audio_path"])
            if not audio_path.is_absolute():
                audio_path = audio_root / audio_path
            rows.append(
                {
                    "utt_id": row["utt_id"],
                    "audio_path": audio_path,
                    "text": normalize_text(row["text"]),
                }
            )
            if limit and len(rows) >= limit:
                break
    return rows


def read_wave(path: Path):
    samples, sample_rate = sf.read(path, dtype="float32", always_2d=False)
    if samples.ndim == 2:
        samples = samples.mean(axis=1)
    samples = np.ascontiguousarray(samples, dtype=np.float32)
    if sample_rate != 16000:
        raise ValueError(f"{path} is {sample_rate} Hz; expected 16000 Hz")
    return samples, sample_rate


def make_recognizer(args):
    suffix = "" if args.fp32 else ".int8"
    model_dir = args.model_dir
    provider = "trt" if args.provider == "tensorrt" else args.provider
    trt_cache = args.trt_cache or (model_dir / "trt_cache")
    trt_cache.mkdir(parents=True, exist_ok=True)
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
        trt_fp16_enable=bool(args.trt_fp16),
        trt_engine_cache_enable=True,
        trt_timing_cache_enable=True,
        trt_engine_cache_path=str(trt_cache),
        trt_timing_cache_path=str(trt_cache),
    )


def decode_one(recognizer, samples, sample_rate):
    stream = recognizer.create_stream()
    stream.accept_waveform(sample_rate, samples)
    stream.input_finished()
    start = time.perf_counter()
    while recognizer.is_ready(stream):
        recognizer.decode_stream(stream)
    elapsed = time.perf_counter() - start
    return normalize_text(recognizer.get_result(stream)), elapsed


def percentile(values, pct):
    if not values:
        return 0.0
    values = sorted(values)
    k = (len(values) - 1) * pct / 100.0
    f = int(k)
    c = min(f + 1, len(values) - 1)
    if f == c:
        return values[f]
    return values[f] + (values[c] - values[f]) * (k - f)


def get_cpu_name():
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        fallback = ""
        for line in cpuinfo.read_text(errors="ignore").splitlines():
            key = line.split(":", 1)[0].strip().lower()
            if key in {"model name", "hardware"}:
                return line.split(":", 1)[-1].strip()
            if key == "processor" and not fallback:
                fallback = line.split(":", 1)[-1].strip()
        if fallback:
            return fallback
    return platform.processor() or "unknown"


def get_tegrastats_sample():
    if not Path("/usr/bin/tegrastats").exists() and not Path("/bin/tegrastats").exists():
        return ""
    try:
        proc = subprocess.Popen(
            ["tegrastats", "--interval", "1000"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        try:
            line = proc.stdout.readline() if proc.stdout else ""
            return line.strip()
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=2)
    except Exception:
        return ""


def summarize(args, rows, recognizer):
    total_ref_words = 0
    total_errors = 0
    total_audio_sec = 0.0
    decode_times = []
    examples = []
    waves = [read_wave(row["audio_path"]) for row in rows]

    for samples, sample_rate in waves[: max(args.warmup, 0)]:
        decode_one(recognizer, samples, sample_rate)

    start_all = time.perf_counter()
    for row, (samples, sample_rate) in zip(rows, waves):
        hyp, elapsed = decode_one(recognizer, samples, sample_rate)
        ref_words = row["text"].split()
        hyp_words = hyp.split()
        errors = edit_distance(ref_words, hyp_words)
        total_ref_words += len(ref_words)
        total_errors += errors
        duration = float(samples.shape[0]) / sample_rate
        total_audio_sec += duration
        decode_times.append(elapsed)
        if len(examples) < 5:
            examples.append(
                {
                    "utt_id": row["utt_id"],
                    "ref": row["text"],
                    "hyp": hyp,
                    "wer": errors / max(1, len(ref_words)),
                    "audio_sec": duration,
                    "decode_sec": elapsed,
                    "rtf": elapsed / duration if duration > 0 else 0.0,
                }
            )
    wall_sec = time.perf_counter() - start_all
    max_rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    model_type = "fp32" if args.fp32 else "int8"
    return {
        "system": {
            "hostname": platform.node(),
            "machine": platform.machine(),
            "platform": platform.platform(),
            "cpu": get_cpu_name(),
            "tegrastats": get_tegrastats_sample(),
        },
        "config": {
            "model_type": model_type,
            "model_dir": str(args.model_dir),
            "threads": args.threads,
            "decoding_method": args.decoding_method,
            "max_active_paths": args.max_active_paths,
            "provider": args.provider,
            "manifest": str(args.manifest),
            "utterances": len(rows),
            "warmup": args.warmup,
            "streaming": True,
        },
        "metrics": {
            "wer_percent": 100.0 * total_errors / max(1, total_ref_words),
            "word_errors": total_errors,
            "ref_words": total_ref_words,
            "audio_sec": total_audio_sec,
            "wall_sec": wall_sec,
            "decode_sec_sum": sum(decode_times),
            "rtf_wall": wall_sec / total_audio_sec if total_audio_sec > 0 else 0.0,
            "rtf_decode_sum": sum(decode_times) / total_audio_sec
            if total_audio_sec > 0
            else 0.0,
            "audio_x_realtime_wall": total_audio_sec / wall_sec if wall_sec > 0 else 0.0,
            "latency_avg_ms": 1000.0 * statistics.mean(decode_times)
            if decode_times
            else 0.0,
            "latency_p50_ms": 1000.0 * percentile(decode_times, 50),
            "latency_p90_ms": 1000.0 * percentile(decode_times, 90),
            "latency_p95_ms": 1000.0 * percentile(decode_times, 95),
            "latency_max_ms": 1000.0 * max(decode_times) if decode_times else 0.0,
            "max_rss_mb": max_rss_kb / 1024.0,
        },
        "examples": examples,
    }


def write_markdown(path: Path, result):
    metrics = result["metrics"]
    config = result["config"]
    system = result["system"]
    lines = [
        "# VietnameseASR Streaming Performance Evaluation",
        "",
        f"- Host: `{system['hostname']}`",
        f"- Machine: `{system['machine']}`",
        f"- CPU: `{system['cpu']}`",
        f"- Model: `{config['model_type']}`",
        f"- Threads: `{config['threads']}`",
        f"- Decode: `{config['decoding_method']}`",
        f"- Provider: `{config['provider']}`",
        f"- Utterances: `{config['utterances']}`",
        "",
        "## Metrics",
        "",
        f"- WER: `{metrics['wer_percent']:.2f}%` "
        f"({metrics['word_errors']} / {metrics['ref_words']})",
        f"- Audio: `{metrics['audio_sec']:.2f}s`",
        f"- Wall time: `{metrics['wall_sec']:.2f}s`",
        f"- RTF wall: `{metrics['rtf_wall']:.4f}`",
        f"- Speed: `{metrics['audio_x_realtime_wall']:.2f}x realtime`",
        f"- Latency avg/p50/p90/p95/max: "
        f"`{metrics['latency_avg_ms']:.1f}` / "
        f"`{metrics['latency_p50_ms']:.1f}` / "
        f"`{metrics['latency_p90_ms']:.1f}` / "
        f"`{metrics['latency_p95_ms']:.1f}` / "
        f"`{metrics['latency_max_ms']:.1f}` ms",
        f"- Max RSS: `{metrics['max_rss_mb']:.1f} MB`",
        f"- Model init: `{metrics.get('init_sec', 0.0):.3f}s`",
    ]
    if system.get("tegrastats"):
        lines += ["", "## Tegrastats", "", f"`{system['tegrastats']}`"]
    lines += ["", "## Examples", ""]
    for ex in result["examples"]:
        lines += [
            f"### {ex['utt_id']}",
            "",
            f"- REF: {ex['ref']}",
            f"- HYP: {ex['hyp']}",
            f"- WER: `{100.0 * ex['wer']:.2f}%`, RTF: `{ex['rtf']:.4f}`",
            "",
        ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    args = get_args()
    rows = read_manifest(args.manifest, args.audio_root, args.limit)
    if not rows:
        raise ValueError(f"No utterances found in {args.manifest}")

    init_start = time.perf_counter()
    recognizer = make_recognizer(args)
    init_sec = time.perf_counter() - init_start
    result = summarize(args, rows, recognizer)
    result["metrics"]["init_sec"] = init_sec

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    if args.output_md:
        args.output_md.parent.mkdir(parents=True, exist_ok=True)
        write_markdown(args.output_md, result)

    m = result["metrics"]
    print(
        f"streaming {result['config']['model_type']} provider={args.provider} "
        f"threads={args.threads} WER={m['wer_percent']:.2f}% "
        f"RTF={m['rtf_wall']:.4f} speed={m['audio_x_realtime_wall']:.2f}x "
        f"avg={m['latency_avg_ms']:.1f}ms p95={m['latency_p95_ms']:.1f}ms "
        f"RSS={m['max_rss_mb']:.1f}MB"
    )


if __name__ == "__main__":
    main()
