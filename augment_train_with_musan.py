#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path
from typing import Dict, List, Tuple

import librosa
import numpy as np
import soundfile as sf

AUDIO_EXTS = {".wav", ".flac", ".mp3", ".m4a", ".ogg", ".opus", ".aac", ".wma"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Offline MUSAN augmentation for train split: create n noisy copies per training utterance and append them to transcripts/train.tsv."
    )
    parser.add_argument(
        "--corpus-root",
        type=Path,
        default=Path("."),
        help="Root of vi_asr_corpus. Default: current directory",
    )
    parser.add_argument(
        "--musan-dir",
        type=Path,
        required=True,
        help="Root folder of MUSAN",
    )
    parser.add_argument(
        "--copies-per-utt",
        type=int,
        default=2,
        help="Number of noisy copies to generate for each training utterance",
    )
    parser.add_argument(
        "--snr-min",
        type=float,
        default=10.0,
        help="Minimum SNR in dB",
    )
    parser.add_argument(
        "--snr-max",
        type=float,
        default=20.0,
        help="Maximum SNR in dB",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--noise-categories",
        type=str,
        default="noise,speech,music",
        help="Comma-separated MUSAN subfolders to use, e.g. noise,speech,music",
    )
    parser.add_argument(
        "--output-subdir",
        type=str,
        default="train_aug_musan",
        help="Subdirectory under audio/ where augmented wavs are stored",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Rebuild augmented files and remove existing transcript rows for this augmentation prefix",
    )
    return parser.parse_args()


def list_audio_files(root: Path) -> List[Path]:
    files: List[Path] = []
    for ext in AUDIO_EXTS:
        files.extend(root.rglob(f"*{ext}"))
    return sorted(set(files))


def load_audio(path: Path, target_sr: int = 16000) -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix == ".wav":
        audio, sr = sf.read(str(path), always_2d=False)
        if isinstance(audio, np.ndarray) and audio.ndim == 2:
            audio = np.mean(audio, axis=1)
        if isinstance(audio, np.ndarray) and np.issubdtype(audio.dtype, np.integer):
            max_abs = np.iinfo(audio.dtype).max
            audio = audio.astype(np.float32) / max_abs
        else:
            audio = np.asarray(audio, dtype=np.float32)
    else:
        audio, sr = librosa.load(str(path), sr=None, mono=True)
        audio = np.asarray(audio, dtype=np.float32)

    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)

    if audio.ndim != 1:
        audio = np.ravel(audio)

    return np.clip(audio, -1.0, 1.0).astype(np.float32)


def ensure_noise_length(noise: np.ndarray, target_len: int, rng: random.Random) -> np.ndarray:
    if len(noise) == target_len:
        return noise
    if len(noise) > target_len:
        start = rng.randint(0, len(noise) - target_len)
        return noise[start:start + target_len]
    # loop/tiling if too short
    reps = (target_len + len(noise) - 1) // len(noise)
    tiled = np.tile(noise, reps)
    start = 0
    if len(tiled) > target_len:
        start = rng.randint(0, len(tiled) - target_len)
    return tiled[start:start + target_len]


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x), dtype=np.float64) + 1e-12))


def mix_with_snr(clean: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    clean_rms = rms(clean)
    noise_rms = rms(noise)

    if noise_rms < 1e-10:
        return clean.copy()

    target_noise_rms = clean_rms / (10 ** (snr_db / 20.0))
    scaled_noise = noise * (target_noise_rms / noise_rms)

    mixed = clean + scaled_noise
    peak = np.max(np.abs(mixed))
    if peak > 0.999:
        mixed = mixed / peak * 0.999
    return mixed.astype(np.float32)


def read_tsv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def write_tsv(path: Path, rows: List[Dict[str, str]]) -> None:
    rows = sorted(rows, key=lambda x: x["utt_id"])
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["utt_id", "speaker", "audio_path", "text"],
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    corpus_root = args.corpus_root.resolve()
    train_tsv = corpus_root / "transcripts" / "train.tsv"
    audio_root = corpus_root / "audio"
    out_audio_root = audio_root / args.output_subdir

    if not train_tsv.is_file():
        raise FileNotFoundError(f"Missing train.tsv: {train_tsv}")
    if not args.musan_dir.is_dir():
        raise FileNotFoundError(f"Missing MUSAN dir: {args.musan_dir}")

    categories = [c.strip() for c in args.noise_categories.split(",") if c.strip()]
    noise_files: List[Path] = []
    for cat in categories:
        cat_dir = args.musan_dir / cat
        if cat_dir.is_dir():
            noise_files.extend(list_audio_files(cat_dir))
    noise_files = sorted(set(noise_files))

    if not noise_files:
        raise RuntimeError(
            f"No MUSAN audio found in categories {categories} under {args.musan_dir}"
        )

    rows = read_tsv(train_tsv)
    aug_prefix = f"_musan"

    if args.overwrite:
        rows = [r for r in rows if aug_prefix not in r["utt_id"]]
        if out_audio_root.exists():
            import shutil
            shutil.rmtree(out_audio_root)

    out_audio_root.mkdir(parents=True, exist_ok=True)

    original_rows = [r for r in rows if aug_prefix not in r["utt_id"]]
    new_rows: List[Dict[str, str]] = []
    generated = 0

    # cache a small set of loaded noise arrays lazily
    noise_cache: Dict[Path, np.ndarray] = {}

    for row in original_rows:
        utt_id = row["utt_id"]
        speaker = row["speaker"]
        text = row["text"]
        src_audio = corpus_root / row["audio_path"]

        if not src_audio.is_file():
            raise FileNotFoundError(f"Missing training audio referenced by train.tsv: {src_audio}")

        clean = load_audio(src_audio, target_sr=16000)
        if len(clean) == 0:
            continue

        for i in range(1, args.copies_per_utt + 1):
            noise_path = rng.choice(noise_files)
            if noise_path not in noise_cache:
                noise_cache[noise_path] = load_audio(noise_path, target_sr=16000)

            noise = noise_cache[noise_path]
            noise_seg = ensure_noise_length(noise, len(clean), rng)
            snr_db = rng.uniform(args.snr_min, args.snr_max)
            mixed = mix_with_snr(clean, noise_seg, snr_db)

            new_utt_id = f"{utt_id}_musan{i:02d}"
            dst = out_audio_root / speaker / f"{new_utt_id}.wav"
            dst.parent.mkdir(parents=True, exist_ok=True)
            sf.write(str(dst), mixed, 16000, subtype="PCM_16")

            new_rows.append(
                {
                    "utt_id": new_utt_id,
                    "speaker": speaker,
                    "audio_path": dst.relative_to(corpus_root).as_posix(),
                    "text": text,
                }
            )
            generated += 1

    all_rows = rows + new_rows
    write_tsv(train_tsv, all_rows)

    print(f"Done. Generated {generated} noisy train utterances.")
    print(f"Updated: {train_tsv}")
    print(f"Stored augmented audio in: {out_audio_root}")
    print(f"Noise pool size: {len(noise_files)}")


if __name__ == "__main__":
    main()
