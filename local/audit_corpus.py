#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import re
from collections import Counter
from pathlib import Path
from statistics import mean, median


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit transcript splits for ASR experiments.")
    parser.add_argument("--transcript-dir", type=Path, default=Path("transcripts"))
    return parser.parse_args()


def read_split(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def words(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower(), flags=re.UNICODE)


def main() -> None:
    args = get_args()
    splits = {
        split: read_split(args.transcript_dir / f"{split}.tsv")
        for split in ("train", "dev", "test")
    }

    for split, rows in splits.items():
        originals = [r for r in rows if "_musan" not in r["utt_id"]]
        lengths = [len(words(r["text"])) for r in originals]
        unique_texts = {r["text"] for r in originals}
        print(
            f"{split}: rows={len(rows)}, originals={len(originals)}, "
            f"unique_texts={len(unique_texts)}, speakers={dict(Counter(r['speaker'] for r in originals))}"
        )
        print(
            f"  words/utterance: mean={mean(lengths):.1f}, "
            f"median={median(lengths):.1f}, min={min(lengths)}, max={max(lengths)}"
        )

    train_vocab = {w for r in splits["train"] for w in words(r["text"])}
    train_texts = {r["text"] for r in splits["train"]}
    for split in ("dev", "test"):
        split_words = [w for r in splits[split] for w in words(r["text"])]
        oov = [w for w in split_words if w not in train_vocab]
        overlap = sum(r["text"] in train_texts for r in splits[split])
        print(
            f"{split} vs train: token_oov={len(oov)}/{len(split_words)} "
            f"({len(oov) / len(split_words):.1%}), oov_types={len(set(oov))}, "
            f"exact_text_overlap={overlap}"
        )

    for left, right in (("train", "dev"), ("train", "test"), ("dev", "test")):
        left_texts = {r["text"] for r in splits[left]}
        right_texts = {r["text"] for r in splits[right]}
        print(f"text_overlap {left}-{right}: {len(left_texts & right_texts)}")


if __name__ == "__main__":
    main()
