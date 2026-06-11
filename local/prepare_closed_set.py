#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from prepare_vi_asr_corpus import scan_auto_dataset  # noqa: E402


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create intentional closed-set splits containing every original recording."
    )
    parser.add_argument("--dataset-dir", type=Path, default=ROOT / "dataset")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "closed_set/transcripts")
    return parser.parse_args()


def main() -> None:
    args = get_args()
    packs = scan_auto_dataset(
        dataset_dir=args.dataset_dir,
        normalize=True,
        skip_empty_lines=True,
        copy_only_wav=False,
        strict_script_detect=False,
    )

    rows = []
    for pack in packs:
        for index, (audio_path, text) in enumerate(
            zip(pack.audio_files, pack.texts), start=1
        ):
            rows.append(
                {
                    "utt_id": f"{pack.speaker}_{index:06d}",
                    "speaker": pack.speaker,
                    "audio_path": audio_path.resolve().relative_to(ROOT).as_posix(),
                    "text": text,
                }
            )

    rows.sort(key=lambda row: row["utt_id"])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "dev", "test"):
        with (args.output_dir / f"{split}.tsv").open(
            "w", encoding="utf-8", newline=""
        ) as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["utt_id", "speaker", "audio_path", "text"],
                delimiter="\t",
            )
            writer.writeheader()
            writer.writerows(rows)

    print(f"Wrote {len(rows)} original recordings to each closed-set split.")
    print(f"Output: {args.output_dir}")


if __name__ == "__main__":
    main()
