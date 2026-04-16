#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import shutil
import sys
from pathlib import Path
from typing import Dict, Iterable, List

from scipy.io import wavfile

try:
    import noisereduce as nr
except ModuleNotFoundError as e:
    raise SystemExit(
        "Missing dependency: noisereduce. Install it with: pip install noisereduce"
    ) from e


ROOT = Path(__file__).resolve().parents[1]
FIELDS = ["utt_id", "speaker", "audio_path", "text"]


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create an offline noisereduce audio/transcript variant."
    )
    parser.add_argument(
        "--input-transcript-dir",
        type=Path,
        default=Path("transcripts"),
        help="Input transcript directory containing train/dev/test.tsv.",
    )
    parser.add_argument(
        "--output-transcript-dir",
        type=Path,
        default=Path("transcripts_nr"),
        help="Output transcript directory for the noise-reduced variant.",
    )
    parser.add_argument(
        "--output-audio-root",
        type=Path,
        default=Path("audio_nr"),
        help="Output audio root for the noise-reduced variant.",
    )
    parser.add_argument(
        "--splits",
        type=str,
        default="train dev test",
        help='Space-separated splits to process. Default: "train dev test".',
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Remove existing output audio/transcript directories before processing.",
    )
    return parser.parse_args()


def read_tsv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def write_tsv(path: Path, rows: Iterable[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def output_audio_path(src_rel: Path, split: str, output_audio_root: Path) -> Path:
    parts = src_rel.parts
    if parts and parts[0] == "audio":
        return output_audio_root.joinpath(*parts[1:])
    return output_audio_root / split / src_rel.name


def reduce_one(src: Path, dst: Path) -> None:
    rate, data = wavfile.read(str(src))
    reduced_noise = nr.reduce_noise(y=data, sr=rate)
    dst.parent.mkdir(parents=True, exist_ok=True)
    wavfile.write(str(dst), rate, reduced_noise)


def process_split(
    split: str,
    input_transcript_dir: Path,
    output_transcript_dir: Path,
    output_audio_root: Path,
) -> int:
    src_tsv = input_transcript_dir / f"{split}.tsv"
    if not src_tsv.is_file():
        raise FileNotFoundError(f"Missing transcript: {src_tsv}")

    rows = read_tsv(src_tsv)
    out_rows: List[Dict[str, str]] = []

    for row in rows:
        src_rel = Path(row["audio_path"])
        src_audio = ROOT / src_rel
        if not src_audio.is_file():
            raise FileNotFoundError(f"Missing audio referenced by {src_tsv}: {src_audio}")

        dst_audio = output_audio_path(src_rel, split, output_audio_root)
        reduce_one(src_audio, dst_audio)

        out_row = dict(row)
        try:
            out_row["audio_path"] = dst_audio.relative_to(ROOT).as_posix()
        except ValueError:
            out_row["audio_path"] = dst_audio.as_posix()
        out_rows.append(out_row)

    write_tsv(output_transcript_dir / f"{split}.tsv", out_rows)
    return len(out_rows)


def main() -> None:
    args = get_args()
    args.input_transcript_dir = (
        args.input_transcript_dir
        if args.input_transcript_dir.is_absolute()
        else ROOT / args.input_transcript_dir
    )
    args.output_transcript_dir = (
        args.output_transcript_dir
        if args.output_transcript_dir.is_absolute()
        else ROOT / args.output_transcript_dir
    )
    args.output_audio_root = (
        args.output_audio_root
        if args.output_audio_root.is_absolute()
        else ROOT / args.output_audio_root
    )
    splits = tuple(s for s in args.splits.split() if s)
    if not splits:
        raise ValueError("--splits must contain at least one split name")

    if args.overwrite:
        for path in (args.output_audio_root, args.output_transcript_dir):
            if path.exists():
                shutil.rmtree(path)

    total = 0
    for split in splits:
        count = process_split(
            split=split,
            input_transcript_dir=args.input_transcript_dir,
            output_transcript_dir=args.output_transcript_dir,
            output_audio_root=args.output_audio_root,
        )
        total += count
        print(f"{split}: wrote {count} noise-reduced utterances")

    print(f"Done. Total utterances: {total}")
    print(f"Audio root: {args.output_audio_root}")
    print(f"Transcript dir: {args.output_transcript_dir}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
