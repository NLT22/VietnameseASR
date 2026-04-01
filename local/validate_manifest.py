#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
from pathlib import Path

from lhotse import CutSet, load_manifest_lazy
from lhotse.cut import Cut
from lhotse.dataset.speech_recognition import validate_for_asr


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path, nargs="?", default=None,
                        help="Path to one cut manifest")
    parser.add_argument("--manifest-dir", type=Path, default=Path("fbank"),
                        help="Directory containing train_cuts.jsonl.gz, dev_cuts.jsonl.gz, test_cuts.jsonl.gz")
    parser.add_argument("--all", action="store_true",
                        help="Validate train/dev/test cut manifests in manifest-dir")
    return parser.parse_args()


def validate_one_supervision_per_cut(c: Cut):
    if len(c.supervisions) != 1:
        raise ValueError(f"{c.id} has {len(c.supervisions)} supervisions")


def validate_supervision_and_cut_time_bounds(c: Cut):
    tol = 2e-3
    s = c.supervisions[0]
    if s.start < -tol:
        raise ValueError(f"{c.id}: supervision start {s.start} must not be negative")
    if s.start > tol:
        raise ValueError(f"{c.id}: supervision start {s.start} is not at beginning of cut")
    if c.start + s.end > c.end + tol:
        raise ValueError(f"{c.id}: supervision end exceeds cut end")


def validate_manifest_file(manifest: Path):
    logging.info(f"Validating {manifest}")
    if not manifest.is_file():
        raise FileNotFoundError(f"{manifest} does not exist")
    cut_set = load_manifest_lazy(manifest)
    assert isinstance(cut_set, CutSet)

    for c in cut_set:
        validate_one_supervision_per_cut(c)
        validate_supervision_and_cut_time_bounds(c)

    validate_for_asr(cut_set)
    logging.info(f"OK: {manifest}")


def main():
    args = get_args()
    if args.all:
        for name in ["train_cuts.jsonl.gz", "dev_cuts.jsonl.gz", "test_cuts.jsonl.gz"]:
            validate_manifest_file(args.manifest_dir / name)
    else:
        if args.manifest is None:
            raise ValueError("Provide MANIFEST or use --all")
        validate_manifest_file(args.manifest)


if __name__ == "__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    logging.basicConfig(format=formatter, level=logging.INFO)
    main()
