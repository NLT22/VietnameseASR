#!/usr/bin/env python3
import argparse, csv, json, os, re, subprocess, time, wave
from pathlib import Path


def edit_distance(a, b):
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, y in enumerate(b, 1):
            cur[j] = min(prev[j] + 1, cur[j-1] + 1, prev[j-1] + (x != y))
        prev = cur
    return prev[-1]


def norm(s):
    return " ".join((s or "").lower().strip().split())


def wav_duration(path):
    with wave.open(str(path), "rb") as f:
        return f.getnframes() / float(f.getframerate())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", required=True)
    ap.add_argument("--precision", choices=["fp32", "int8"], default="fp32")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--audio-root", required=True)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--bin", default="/home/thayhoang/sherpa-onnx-gpu-build/sherpa-onnx/build-gpu/bin/sherpa-onnx")
    ap.add_argument("--output-json", required=True)
    ap.add_argument("--output-md", required=True)
    ap.add_argument(
        "--chunk-size",
        type=int,
        default=0,
        help="Run at most this many wavs per sherpa process. Useful on low-RAM Jetson Nano.",
    )
    args = ap.parse_args()

    rows = []
    with open(args.manifest, encoding="utf-8") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            rows.append(r)
    root = Path(args.audio_root)
    model = Path(args.model_dir)
    suffix = ".int8.onnx" if args.precision == "int8" else ".onnx"
    wavs = [str(root / r["audio_path"]) for r in rows]
    total_audio = sum(wav_duration(Path(w)) for w in wavs)

    env = os.environ.copy()
    lib = "/home/thayhoang/sherpa-onnx-gpu-build/sherpa-onnx/build-gpu/lib:/home/thayhoang/sherpa-onnx-gpu-build/sherpa-onnx/build-gpu/_deps/onnxruntime-src/lib"
    env["LD_LIBRARY_PATH"] = lib + (":" + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else "")
    enc_name = "encoder" + suffix
    dec_name = "decoder" + suffix
    join_name = "joiner" + suffix
    base_cmd = [
        args.bin,
        "--tokens=" + str(model / "tokens.txt"),
        "--encoder=" + str(model / enc_name),
        "--decoder=" + str(model / dec_name),
        "--joiner=" + str(model / join_name),
        "--modeling-unit=bpe",
        "--bpe-vocab=" + str(model / "bpe.vocab"),
        f"--provider={args.provider}",
        "--num-threads=2",
        "--decoding-method=modified_beam_search",
        "--max-active-paths=4",
        "--model-type=zipformer2",
    ]
    start = time.time()
    chunk_size = args.chunk_size if args.chunk_size and args.chunk_size > 0 else len(wavs)
    outputs = []
    returncodes = []
    for i in range(0, len(wavs), chunk_size):
        chunk_wavs = wavs[i : i + chunk_size]
        p = subprocess.run(
            [*base_cmd, *chunk_wavs],
            env=env,
            universal_newlines=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        returncodes.append(p.returncode)
        outputs.append(p.stdout)
        if p.returncode != 0:
            break
    elapsed = time.time() - start
    out = "\n".join(outputs)
    hyps = []
    for line in out.splitlines():
        line = line.strip()
        if line.startswith("{") and '"text"' in line:
            try:
                hyps.append(json.loads(line).get("text", ""))
            except Exception:
                pass
    if len(hyps) != len(rows):
        raise RuntimeError(
            f"Expected {len(rows)} JSON hypotheses, got {len(hyps)}. "
            f"Returns={returncodes}\nLast output:\n" + "\n".join(out.splitlines()[-40:])
        )

    total_words = total_err = 0
    details = []
    for r, hyp in zip(rows, hyps):
        ref_words = norm(r["text"]).split()
        hyp_words = norm(hyp).split()
        err = edit_distance(ref_words, hyp_words)
        total_words += len(ref_words)
        total_err += err
        details.append({"utt_id": r["utt_id"], "ref": norm(r["text"]), "hyp": norm(hyp), "errors": err, "ref_words": len(ref_words)})
    wer = total_err / total_words if total_words else 0.0
    result = {
        "provider": args.provider,
        "precision": args.precision,
        "num_utterances": len(rows),
        "total_audio_seconds": total_audio,
        "elapsed_seconds_including_startup": elapsed,
        "rtf_including_startup": elapsed / total_audio if total_audio else None,
        "wer": wer,
        "errors": total_err,
        "words": total_words,
        "returncodes": returncodes,
        "chunk_size": chunk_size,
        "details": details,
    }
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_json).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    Path(args.output_md).write_text(
        f"# sherpa-onnx binary eval\n\nProvider: {args.provider}\n\nPrecision: {args.precision}\n\nWER: {wer*100:.2f}% ({total_err}/{total_words})\n\nRTF incl startup: {elapsed/total_audio:.3f}\n\nElapsed: {elapsed:.2f}s\nAudio: {total_audio:.2f}s\n",
        encoding="utf-8",
    )
    print(json.dumps({k: result[k] for k in result if k != "details"}, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
