#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import re
import sys
import unicodedata
from pathlib import Path
from typing import Dict, List, Sequence

import librosa
import numpy as np
import soundfile as sf

AUDIO_EXTS = {".wav", ".flac", ".mp3", ".m4a", ".ogg", ".opus", ".aac", ".wma"}
PUNCT = r'''!"#$%&'()*+,./:;<=>?@[\\]^_`{|}~“”‘’…–—'''


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Append one speaker's recordings into an existing vi_asr_corpus safely."
    )
    parser.add_argument("--audio-dir", type=Path, required=True, help="Directory containing input audio files.")
    parser.add_argument("--prompts", type=Path, required=True, help="Prompt file: one line per audio file.")
    parser.add_argument("--speaker", type=str, required=True, help="Speaker id/name. Example: spk001")
    parser.add_argument("--output-root", type=Path, default=Path("vi_asr_corpus"), help="Corpus root.")
    parser.add_argument("--sample-rate", type=int, default=16000, help="Target sample rate. Default: 16000")
    parser.add_argument("--train-ratio", type=float, default=0.9, help="Train split ratio. Default: 0.9")
    parser.add_argument("--dev-ratio", type=float, default=0.05, help="Dev split ratio. Default: 0.05")
    parser.add_argument("--test-ratio", type=float, default=0.05, help="Test split ratio. Default: 0.05")
    parser.add_argument("--copy-only-wav", action="store_true", help="Only accept WAV input.")
    parser.add_argument("--keep-empty-prompts", action="store_true", help="Allow empty prompts.")
    parser.add_argument("--normalize-text", action="store_true", help="Normalize transcript text.")
    parser.add_argument(
        "--replace-speaker",
        action="store_true",
        help="Remove old data of this speaker only, then add current run. Does not touch other speakers.",
    )
    return parser.parse_args()


def sanitize_speaker_name(name: str) -> str:
    clean = re.sub(r"[^0-9A-Za-z_\-]", "_", name.strip())
    clean = re.sub(r"_+", "_", clean).strip("_")
    if not clean:
        raise ValueError("Speaker name becomes empty after sanitization.")
    return clean


def validate_ratios(train_ratio: float, dev_ratio: float, test_ratio: float) -> None:
    total = train_ratio + dev_ratio + test_ratio
    if not math.isclose(total, 1.0, rel_tol=1e-9, abs_tol=1e-9):
        raise ValueError(f"Split ratios must sum to 1.0, but got {total}")


def normalize_vietnamese_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = text.strip().lower()
    text = re.sub(f"[{re.escape(PUNCT)}]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def list_audio_files(audio_dir: Path) -> List[Path]:
    if not audio_dir.is_dir():
        raise FileNotFoundError(f"Audio directory not found: {audio_dir}")
    files = [p for p in audio_dir.iterdir() if p.is_file() and p.suffix.lower() in AUDIO_EXTS]
    files.sort(key=lambda p: p.name)
    if not files:
        raise RuntimeError(f"No supported audio files found in: {audio_dir}")
    return files


def read_prompts(prompt_path: Path, allow_empty: bool = False, normalize: bool = False) -> List[str]:
    if not prompt_path.is_file():
        raise FileNotFoundError(f"Prompt file not found: {prompt_path}")

    lines: List[str] = []
    with prompt_path.open("r", encoding="utf-8") as f:
        for idx, raw in enumerate(f, start=1):
            text = raw.rstrip("\n\r").strip()
            text = normalize_vietnamese_text(text) if normalize else re.sub(r"\s+", " ", text)
            if not text and not allow_empty:
                raise ValueError(f"Prompt line {idx} is empty after processing.")
            lines.append(text)

    if not lines:
        raise RuntimeError(f"Prompt file is empty: {prompt_path}")
    return lines


def make_split_names(n_items: int, train_ratio: float, dev_ratio: float) -> List[str]:
    n_train = int(n_items * train_ratio)
    n_dev = int(n_items * dev_ratio)
    n_test = n_items - n_train - n_dev
    split_names = ["train"] * n_train + ["dev"] * n_dev + ["test"] * n_test

    if n_items >= 3:
        if "dev" not in split_names:
            split_names[-2] = "dev"
        if "test" not in split_names:
            split_names[-1] = "test"
        if split_names.count("train") == 0:
            split_names[0] = "train"
    elif n_items == 2:
        split_names = ["train", "test"]
    elif n_items == 1:
        split_names = ["train"]

    return split_names


def load_audio(path: Path, target_sr: int, copy_only_wav: bool = False) -> np.ndarray:
    suffix = path.suffix.lower()
    if copy_only_wav and suffix != ".wav":
        raise ValueError(f"--copy-only-wav is set, but input is not WAV: {path}")

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

    if len(audio) == 0:
        raise ValueError(f"Audio has zero length: {path}")

    return np.clip(audio, -1.0, 1.0).astype(np.float32)


def write_wav(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), audio, sample_rate, subtype="PCM_16")


def ensure_corpus_dirs(root: Path) -> None:
    (root / "audio" / "train").mkdir(parents=True, exist_ok=True)
    (root / "audio" / "dev").mkdir(parents=True, exist_ok=True)
    (root / "audio" / "test").mkdir(parents=True, exist_ok=True)
    (root / "transcripts").mkdir(parents=True, exist_ok=True)


def read_tsv(tsv_path: Path) -> List[Dict[str, str]]:
    if not tsv_path.is_file():
        return []
    with tsv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        return list(reader)


def write_tsv(tsv_path: Path, rows: Sequence[Dict[str, str]]) -> None:
    tsv_path.parent.mkdir(parents=True, exist_ok=True)
    rows = sorted(rows, key=lambda x: x["utt_id"])
    with tsv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["utt_id", "speaker", "audio_path", "text"], delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def relative_posix(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def get_existing_rows(root: Path) -> Dict[str, List[Dict[str, str]]]:
    transcripts = root / "transcripts"
    return {
        "train": read_tsv(transcripts / "train.tsv"),
        "dev": read_tsv(transcripts / "dev.tsv"),
        "test": read_tsv(transcripts / "test.tsv"),
    }


def next_index_for_speaker(existing_rows: Dict[str, List[Dict[str, str]]], speaker: str) -> int:
    max_idx = 0
    prefix = f"{speaker}_"
    for split_rows in existing_rows.values():
        for row in split_rows:
            utt_id = row.get("utt_id", "")
            if not utt_id.startswith(prefix):
                continue
            try:
                idx = int(utt_id.split("_")[-1])
                max_idx = max(max_idx, idx)
            except ValueError:
                pass
    return max_idx + 1


def remove_speaker_rows(existing_rows: Dict[str, List[Dict[str, str]]], speaker: str) -> Dict[str, List[Dict[str, str]]]:
    filtered = {}
    for split, rows in existing_rows.items():
        filtered[split] = [r for r in rows if r.get("speaker") != speaker]
    return filtered


def remove_speaker_audio(root: Path, speaker: str) -> None:
    for split in ["train", "dev", "test"]:
        spk_dir = root / "audio" / split / speaker
        if spk_dir.exists():
            for p in spk_dir.glob("*.wav"):
                p.unlink()
            try:
                spk_dir.rmdir()
            except OSError:
                pass


def write_summary(
    root: Path,
    added_counts: Dict[str, int],
    total_rows: Dict[str, List[Dict[str, str]]],
    speaker: str,
    normalize_text: bool,
    added_durations: Dict[str, float],
    total_durations: Dict[str, float],
) -> None:
    summary_path = root / "README_PREPARED.txt"
    total_count = sum(len(v) for v in total_rows.values())
    total_seconds = sum(total_durations.values())
    added_seconds = sum(added_durations.values())

    with summary_path.open("w", encoding="utf-8") as f:
        f.write("Vietnamese ASR corpus prepared successfully.\n")
        f.write("This corpus supports repeated append runs for multiple speakers.\n\n")

        f.write(f"last_added_speaker: {speaker}\n")
        f.write(f"normalize_text: {normalize_text}\n\n")

        f.write("Last run:\n")
        f.write(f"  train_entries: {added_counts['train']}\n")
        f.write(f"  dev_entries: {added_counts['dev']}\n")
        f.write(f"  test_entries: {added_counts['test']}\n")
        f.write(f"  train_duration_sec: {added_durations['train']:.2f}\n")
        f.write(f"  dev_duration_sec: {added_durations['dev']:.2f}\n")
        f.write(f"  test_duration_sec: {added_durations['test']:.2f}\n")
        f.write(f"  total_added_duration_sec: {added_seconds:.2f}\n")
        f.write(f"  total_added_duration_hms: {format_seconds(added_seconds)}\n\n")

        f.write("Corpus total:\n")
        f.write(f"  total_entries: {total_count}\n")
        f.write(f"  train_entries: {len(total_rows['train'])}\n")
        f.write(f"  dev_entries: {len(total_rows['dev'])}\n")
        f.write(f"  test_entries: {len(total_rows['test'])}\n")
        f.write(f"  train_duration_sec: {total_durations['train']:.2f}\n")
        f.write(f"  dev_duration_sec: {total_durations['dev']:.2f}\n")
        f.write(f"  test_duration_sec: {total_durations['test']:.2f}\n")
        f.write(f"  total_duration_sec: {total_seconds:.2f}\n")
        f.write(f"  total_duration_hms: {format_seconds(total_seconds)}\n")

def format_seconds(seconds: float) -> str:
    total = int(round(seconds))
    h = total // 3600
    m = (total % 3600) // 60
    s = total % 60
    return f"{h:02d}:{m:02d}:{s:02d}"

def get_audio_duration_seconds(path: Path) -> float:
    info = sf.info(str(path))
    return float(info.duration)

def prepare_corpus(args: argparse.Namespace) -> None:
    validate_ratios(args.train_ratio, args.dev_ratio, args.test_ratio)

    speaker = sanitize_speaker_name(args.speaker)
    ensure_corpus_dirs(args.output_root)

    audio_files = list_audio_files(args.audio_dir)
    prompts = read_prompts(args.prompts, allow_empty=args.keep_empty_prompts, normalize=args.normalize_text)

    if len(audio_files) != len(prompts):
        raise RuntimeError(
            f"Mismatch between audio files and prompt lines: {len(audio_files)} files vs {len(prompts)} lines."
        )

    existing_rows = get_existing_rows(args.output_root)

    if args.replace_speaker:
        existing_rows = remove_speaker_rows(existing_rows, speaker)
        remove_speaker_audio(args.output_root, speaker)

    next_idx = next_index_for_speaker(existing_rows, speaker)
    split_names = make_split_names(len(audio_files), args.train_ratio, args.dev_ratio)

    added_counts = {"train": 0, "dev": 0, "test": 0}
    added_durations = {"train": 0.0, "dev": 0.0, "test": 0.0}

    for offset, (src_audio, text, split) in enumerate(zip(audio_files, prompts, split_names), start=0):
        utt_num = next_idx + offset
        utt_id = f"{speaker}_{utt_num:06d}"
        dst_wav = args.output_root / "audio" / split / speaker / f"{utt_id}.wav"

        audio = load_audio(src_audio, target_sr=args.sample_rate, copy_only_wav=args.copy_only_wav)
        write_wav(dst_wav, audio, args.sample_rate)

        duration_sec = get_audio_duration_seconds(dst_wav)
        existing_rows[split].append(
            {
                "utt_id": utt_id,
                "speaker": speaker,
                "audio_path": relative_posix(dst_wav, args.output_root),
                "text": text,
            }
        )
        added_counts[split] += 1
        added_durations[split] += duration_sec

    transcripts = args.output_root / "transcripts"
    for split in ["train", "dev", "test"]:
        write_tsv(transcripts / f"{split}.tsv", existing_rows[split])

        total_durations = {"train": 0.0, "dev": 0.0, "test": 0.0}
    
    for split in ["train", "dev", "test"]:
        split_audio_dir = args.output_root / "audio" / split
        if split_audio_dir.exists():
            for wav_path in split_audio_dir.rglob("*.wav"):
                total_durations[split] += get_audio_duration_seconds(wav_path)

    write_summary(
        args.output_root,
        added_counts,
        existing_rows,
        speaker,
        args.normalize_text,
        added_durations,
        total_durations,
    )

    print(f"Done. Corpus root: {args.output_root.resolve()}")
    print(f"Added speaker: {speaker}")
    print(
        f"Added this run -> train: {added_counts['train']}, "
        f"dev: {added_counts['dev']}, test: {added_counts['test']}"
    )
    print(
        f"Total corpus -> train: {len(existing_rows['train'])}, "
        f"dev: {len(existing_rows['dev'])}, test: {len(existing_rows['test'])}"
    )
    total_added_sec = sum(added_durations.values())
    total_corpus_sec = sum(total_durations.values())

    print(
        f"Added duration -> train: {added_durations['train']:.2f}s, "
        f"dev: {added_durations['dev']:.2f}s, test: {added_durations['test']:.2f}s, "
        f"total: {format_seconds(total_added_sec)}"
    )
    print(
        f"Corpus duration -> train: {format_seconds(total_durations['train'])}, "
        f"dev: {format_seconds(total_durations['dev'])}, "
        f"test: {format_seconds(total_durations['test'])}, "
        f"total: {format_seconds(total_corpus_sec)}"
    )

    # print("Safe mode: no global delete is performed.")


if __name__ == "__main__":
    try:
        prepare_corpus(parse_args())
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
