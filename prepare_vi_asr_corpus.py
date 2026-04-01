#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import re
import shutil
import sys
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import librosa
import numpy as np
import soundfile as sf

AUDIO_EXTS = {".wav", ".flac", ".mp3", ".m4a", ".ogg", ".opus", ".aac", ".wma"}
TEXT_EXTS = {".txt"}
PUNCT = r'''!"#$%&'()*+,./:;<=>?@[\\]^_`{|}~“”‘’…–—'''

PREFERRED_SCRIPT_NAMES = [
    "script.txt",
    "scripts.txt",
    "prompt.txt",
    "prompts.txt",
    "transcript.txt",
    "transcripts.txt",
    "text.txt",
]


@dataclass
class SpeakerPack:
    speaker: str
    speaker_dir: Path
    script_file: Path
    audio_files: List[Path]
    texts: List[str]


def parse_args() -> argparse.Namespace:
    default_root = Path(__file__).resolve().parent
    default_dataset_dir = default_root / "dataset"
    default_output_root = default_root

    parser = argparse.ArgumentParser(
        description="Prepare vi_asr_corpus in single-speaker mode or auto multi-speaker mode."
    )

    # mode
    parser.add_argument(
        "--auto",
        action="store_true",
        help="Auto mode: dataset-dir contains one subfolder per speaker.",
    )

    # single-speaker inputs
    parser.add_argument("--audio-dir", type=Path, help="Single-speaker input audio directory.")
    parser.add_argument("--prompts", type=Path, help="Single-speaker text file: one line per audio.")
    parser.add_argument("--speaker", type=str, help="Single-speaker id/name.")

    # auto mode inputs
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=default_dataset_dir,
        help="Auto mode dataset root. Default: ./dataset next to this script",
    )

    # output
    parser.add_argument(
        "--output-root",
        type=Path,
        default=default_output_root,
        help="Output corpus root. Default: directory containing this script",
    )

    # preprocessing
    parser.add_argument("--sample-rate", type=int, default=16000, help="Target sample rate.")
    parser.add_argument("--train-ratio", type=float, default=0.9, help="Train split ratio.")
    parser.add_argument("--dev-ratio", type=float, default=0.05, help="Dev split ratio.")
    parser.add_argument("--test-ratio", type=float, default=0.05, help="Test split ratio.")
    parser.add_argument(
        "--normalize-text",
        action="store_true",
        default=True,
        help="Normalize text. Default: enabled.",
    )
    parser.add_argument(
        "--no-normalize-text",
        action="store_false",
        dest="normalize_text",
        help="Disable text normalization.",
    )
    parser.add_argument(
        "--skip-empty-lines",
        action="store_true",
        default=True,
        help="Ignore empty lines in text/script files. Default: enabled.",
    )
    parser.add_argument(
        "--keep-empty-lines",
        action="store_true",
        help="Keep empty lines instead of skipping them.",
    )
    parser.add_argument(
        "--copy-only-wav",
        action="store_true",
        help="Only accept WAV input files.",
    )
    parser.add_argument(
        "--strict-script-detect",
        action="store_true",
        help="Auto mode only: require exactly one candidate script .txt in each speaker folder.",
    )

    # safe overwrite
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Only clear files/folders managed by this script inside output-root: audio/, transcripts/, README_PREPARED.txt. It does NOT delete the whole output-root.",
    )

    return parser.parse_args()


def natural_key(name: str):
    parts = re.split(r"(\d+)", name)
    out = []
    for p in parts:
        out.append(int(p) if p.isdigit() else p.lower())
    return out


def sanitize_speaker_name(name: str) -> str:
    clean = re.sub(r"[^0-9A-Za-z_\-]", "_", name.strip())
    clean = re.sub(r"_+", "_", clean).strip("_")
    if not clean:
        raise ValueError(f"Invalid speaker name after sanitization: {name!r}")
    return clean


def normalize_vietnamese_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = text.strip().lower()
    text = re.sub(f"[{re.escape(PUNCT)}]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def prepare_text(text: str, normalize: bool) -> str:
    text = unicodedata.normalize("NFC", text).strip()
    if normalize:
        return normalize_vietnamese_text(text)
    return re.sub(r"\s+", " ", text).strip()


def validate_ratios(train_ratio: float, dev_ratio: float, test_ratio: float) -> None:
    total = train_ratio + dev_ratio + test_ratio
    if not math.isclose(total, 1.0, rel_tol=1e-9, abs_tol=1e-9):
        raise ValueError(
            f"Split ratios must sum to 1.0, but got {train_ratio} + {dev_ratio} + {test_ratio} = {total}"
        )


def safe_prepare_output_root(output_root: Path, overwrite: bool) -> Tuple[Path, Path]:
    output_root.mkdir(parents=True, exist_ok=True)

    managed_paths = [
        output_root / "audio",
        output_root / "transcripts",
        output_root / "README_PREPARED.txt",
    ]

    if overwrite:
        for p in managed_paths:
            if p.is_dir():
                shutil.rmtree(p)
            elif p.is_file():
                p.unlink()

    audio_root = output_root / "audio"
    transcripts_root = output_root / "transcripts"
    for split in ["train", "dev", "test"]:
        (audio_root / split).mkdir(parents=True, exist_ok=True)
    transcripts_root.mkdir(parents=True, exist_ok=True)
    return audio_root, transcripts_root


def list_audio_files(audio_dir: Path, copy_only_wav: bool) -> List[Path]:
    if not audio_dir.is_dir():
        raise FileNotFoundError(f"Audio directory not found: {audio_dir}")
    files = [
        p for p in audio_dir.iterdir()
        if p.is_file() and p.suffix.lower() in AUDIO_EXTS and (not copy_only_wav or p.suffix.lower() == ".wav")
    ]
    files.sort(key=lambda p: natural_key(p.name))
    if not files:
        raise RuntimeError(f"No supported audio files found in: {audio_dir}")
    return files


def read_prompt_lines(prompt_path: Path, normalize: bool, skip_empty_lines: bool) -> List[str]:
    if not prompt_path.is_file():
        raise FileNotFoundError(f"Prompt/script file not found: {prompt_path}")

    lines: List[str] = []
    with prompt_path.open("r", encoding="utf-8") as f:
        for raw in f:
            text = raw.rstrip("\n\r")
            text = prepare_text(text, normalize=normalize)
            if not text and skip_empty_lines:
                continue
            lines.append(text)

    if not lines:
        raise RuntimeError(f"Prompt/script file is empty after processing: {prompt_path}")
    return lines


def detect_script_file(speaker_dir: Path, strict: bool) -> Path:
    preferred_hits = []
    for name in PREFERRED_SCRIPT_NAMES:
        p = speaker_dir / name
        if p.is_file():
            preferred_hits.append(p)

    if len(preferred_hits) == 1:
        return preferred_hits[0]
    if len(preferred_hits) > 1:
        raise RuntimeError(
            f"Multiple preferred script files found in {speaker_dir}: {[p.name for p in preferred_hits]}"
        )

    candidates = [p for p in speaker_dir.iterdir() if p.is_file() and p.suffix.lower() in TEXT_EXTS]
    if not candidates:
        raise FileNotFoundError(f"No .txt script file found in {speaker_dir}")

    candidates = sorted(candidates, key=lambda p: natural_key(p.name))
    if strict and len(candidates) != 1:
        raise RuntimeError(
            f"Expected exactly one script text file in {speaker_dir}, found {[p.name for p in candidates]}"
        )

    if len(candidates) > 1:
        ranked = sorted(
            candidates,
            key=lambda p: (
                0 if any(k in p.stem.lower() for k in ["script", "prompt", "transcript", "text"]) else 1,
                natural_key(p.name),
            ),
        )
        return ranked[0]

    return candidates[0]


def scan_auto_dataset(dataset_dir: Path, normalize: bool, skip_empty_lines: bool, copy_only_wav: bool, strict_script_detect: bool) -> List[SpeakerPack]:
    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    speaker_dirs = [p for p in dataset_dir.iterdir() if p.is_dir()]
    speaker_dirs.sort(key=lambda p: natural_key(p.name))
    if not speaker_dirs:
        raise RuntimeError(f"No speaker subdirectories found in {dataset_dir}")

    packs: List[SpeakerPack] = []
    for speaker_dir in speaker_dirs:
        speaker = sanitize_speaker_name(speaker_dir.name)
        script_file = detect_script_file(speaker_dir, strict=strict_script_detect)
        audio_files = list_audio_files(speaker_dir, copy_only_wav=copy_only_wav)
        texts = read_prompt_lines(script_file, normalize=normalize, skip_empty_lines=skip_empty_lines)

        if len(audio_files) != len(texts):
            raise RuntimeError(
                f"Mismatch in {speaker_dir.name}: {len(audio_files)} audio files vs {len(texts)} script lines.\n"
                f"Script file: {script_file}"
            )

        packs.append(
            SpeakerPack(
                speaker=speaker,
                speaker_dir=speaker_dir,
                script_file=script_file,
                audio_files=audio_files,
                texts=texts,
            )
        )

    return packs


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
        if "train" not in split_names:
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


def write_tsv(tsv_path: Path, rows: Sequence[Dict[str, str]]) -> None:
    with tsv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["utt_id", "speaker", "audio_path", "text"], delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def relative_posix(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def build_rows_for_pack(
    pack: SpeakerPack,
    output_root: Path,
    audio_root: Path,
    sample_rate: int,
    train_ratio: float,
    dev_ratio: float,
    copy_only_wav: bool,
) -> Tuple[Dict[str, List[Dict[str, str]]], Dict[str, int]]:
    rows_by_split: Dict[str, List[Dict[str, str]]] = {"train": [], "dev": [], "test": []}
    speaker_counts = {"train": 0, "dev": 0, "test": 0}

    split_names = make_split_names(len(pack.audio_files), train_ratio, dev_ratio)
    for idx, (src_audio, text, split) in enumerate(zip(pack.audio_files, pack.texts, split_names), start=1):
        utt_id = f"{pack.speaker}_{idx:06d}"
        dst_wav = audio_root / split / pack.speaker / f"{utt_id}.wav"

        audio = load_audio(src_audio, target_sr=sample_rate, copy_only_wav=copy_only_wav)
        write_wav(dst_wav, audio, sample_rate)

        rows_by_split[split].append(
            {
                "utt_id": utt_id,
                "speaker": pack.speaker,
                "audio_path": relative_posix(dst_wav, output_root),
                "text": text,
            }
        )
        speaker_counts[split] += 1

    return rows_by_split, speaker_counts


def run_single_mode(args: argparse.Namespace, skip_empty_lines: bool) -> None:
    if args.audio_dir is None or args.prompts is None or args.speaker is None:
        raise ValueError("Single-speaker mode requires --audio-dir, --prompts, and --speaker.")

    speaker = sanitize_speaker_name(args.speaker)
    audio_files = list_audio_files(args.audio_dir, copy_only_wav=args.copy_only_wav)
    texts = read_prompt_lines(args.prompts, normalize=args.normalize_text, skip_empty_lines=skip_empty_lines)

    if len(audio_files) != len(texts):
        raise RuntimeError(
            f"Mismatch: {len(audio_files)} audio files vs {len(texts)} prompt lines.\n"
            f"Prompt file: {args.prompts}"
        )

    pack = SpeakerPack(
        speaker=speaker,
        speaker_dir=args.audio_dir,
        script_file=args.prompts,
        audio_files=audio_files,
        texts=texts,
    )

    audio_root, transcripts_root = safe_prepare_output_root(args.output_root, args.overwrite)
    rows_by_split, speaker_counts = build_rows_for_pack(
        pack=pack,
        output_root=args.output_root,
        audio_root=audio_root,
        sample_rate=args.sample_rate,
        train_ratio=args.train_ratio,
        dev_ratio=args.dev_ratio,
        copy_only_wav=args.copy_only_wav,
    )

    for split in ["train", "dev", "test"]:
        rows_by_split[split].sort(key=lambda x: x["utt_id"])
        write_tsv(transcripts_root / f"{split}.tsv", rows_by_split[split])

    total_files = len(audio_files)
    with (args.output_root / "README_PREPARED.txt").open("w", encoding="utf-8") as f:
        f.write("Vietnamese ASR corpus prepared in single-speaker mode.\n\n")
        f.write(f"speaker: {speaker}\n")
        f.write(f"audio_dir: {args.audio_dir.resolve()}\n")
        f.write(f"prompts: {args.prompts.resolve()}\n")
        f.write(f"output_root: {args.output_root.resolve()}\n")
        f.write(f"sample_rate: {args.sample_rate}\n")
        f.write(f"normalize_text: {args.normalize_text}\n")
        f.write(f"skip_empty_lines: {skip_empty_lines}\n")
        f.write(f"total_files: {total_files}\n")
        f.write(f"train_total: {speaker_counts['train']}\n")
        f.write(f"dev_total: {speaker_counts['dev']}\n")
        f.write(f"test_total: {speaker_counts['test']}\n")

    print(f"Done. Corpus created at: {args.output_root.resolve()}")
    print("Mode: single-speaker")
    print(f"Speaker: {speaker}")
    print(
        f"Split totals -> train: {speaker_counts['train']}, "
        f"dev: {speaker_counts['dev']}, test: {speaker_counts['test']}"
    )


def run_auto_mode(args: argparse.Namespace, skip_empty_lines: bool) -> None:
    packs = scan_auto_dataset(
        dataset_dir=args.dataset_dir,
        normalize=args.normalize_text,
        skip_empty_lines=skip_empty_lines,
        copy_only_wav=args.copy_only_wav,
        strict_script_detect=args.strict_script_detect,
    )

    audio_root, transcripts_root = safe_prepare_output_root(args.output_root, args.overwrite)

    rows_by_split: Dict[str, List[Dict[str, str]]] = {"train": [], "dev": [], "test": []}
    summary: List[str] = []
    total_files = 0

    for pack in packs:
        speaker_rows, speaker_counts = build_rows_for_pack(
            pack=pack,
            output_root=args.output_root,
            audio_root=audio_root,
            sample_rate=args.sample_rate,
            train_ratio=args.train_ratio,
            dev_ratio=args.dev_ratio,
            copy_only_wav=args.copy_only_wav,
        )

        for split in ["train", "dev", "test"]:
            rows_by_split[split].extend(speaker_rows[split])

        total_files += len(pack.audio_files)
        summary.append(
            f"{pack.speaker}\tfiles={len(pack.audio_files)}\ttrain={speaker_counts['train']}\tdev={speaker_counts['dev']}\ttest={speaker_counts['test']}\tscript={pack.script_file.name}"
        )

    for split in ["train", "dev", "test"]:
        rows_by_split[split].sort(key=lambda x: x["utt_id"])
        write_tsv(transcripts_root / f"{split}.tsv", rows_by_split[split])

    with (args.output_root / "README_PREPARED.txt").open("w", encoding="utf-8") as f:
        f.write("Vietnamese ASR corpus prepared in auto multi-speaker mode.\n\n")
        f.write(f"dataset_dir: {args.dataset_dir.resolve()}\n")
        f.write(f"output_root: {args.output_root.resolve()}\n")
        f.write(f"sample_rate: {args.sample_rate}\n")
        f.write(f"normalize_text: {args.normalize_text}\n")
        f.write(f"skip_empty_lines: {skip_empty_lines}\n")
        f.write(f"total_files: {total_files}\n")
        f.write(f"train_total: {len(rows_by_split['train'])}\n")
        f.write(f"dev_total: {len(rows_by_split['dev'])}\n")
        f.write(f"test_total: {len(rows_by_split['test'])}\n\n")
        f.write("Per-speaker summary:\n")
        for line in summary:
            f.write(line + "\n")

    print(f"Done. Corpus created at: {args.output_root.resolve()}")
    print("Mode: auto")
    print(f"Speakers: {len(packs)}")
    print(f"Total utterances: {total_files}")
    print(
        f"Split totals -> train: {len(rows_by_split['train'])}, "
        f"dev: {len(rows_by_split['dev'])}, test: {len(rows_by_split['test'])}"
    )


def main():
    args = parse_args()
    validate_ratios(args.train_ratio, args.dev_ratio, args.test_ratio)
    skip_empty_lines = False if args.keep_empty_lines else args.skip_empty_lines

    if args.auto:
        run_auto_mode(args, skip_empty_lines)
    else:
        run_single_mode(args, skip_empty_lines)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
