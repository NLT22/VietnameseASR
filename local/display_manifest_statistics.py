#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from lhotse import load_manifest_lazy


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path, nargs="?", default=None,
                        help="Path to one cut manifest")
    parser.add_argument("--manifest-dir", type=Path, default=Path("fbank"))
    parser.add_argument("--all", action="store_true",
                        help="Describe train/dev/test cut manifests in manifest-dir")
    return parser.parse_args()


def describe_one(path: Path):
    print(f"\n===== {path} =====")
    cuts = load_manifest_lazy(path)
    cuts.describe()


def main():
    args = get_args()
    if args.all:
        for name in ["train_cuts.jsonl.gz", "dev_cuts.jsonl.gz", "test_cuts.jsonl.gz"]:
            describe_one(args.manifest_dir / name)
    else:
        if args.manifest is None:
            raise ValueError("Provide MANIFEST or use --all")
        describe_one(args.manifest)


if __name__ == "__main__":
    main()
