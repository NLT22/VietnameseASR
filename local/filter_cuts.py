#!/usr/bin/env python3
# Copyright 2022 Xiaomi Corp. (authors: Fangjun Kuang)
# Adapted for vi_asr_corpus (Zipformer2 subsampling formula)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Remove cuts that are too short, too long, or have too many tokens relative to
their length (which would make pruned RNN-T fail at T < S).

Run display_manifest_statistics.py first to pick appropriate min/max thresholds
for your corpus.

Usage:
  cd /path/to/icefall/egs/vi_asr_corpus

  python3 local/filter_cuts.py \
    --bpe-model data/lang_bpe_100/bpe.model \
    --in-cuts fbank/train_cuts.jsonl.gz \
    --out-cuts fbank/train_cuts_filtered.jsonl.gz

  # Tune thresholds:
  python3 local/filter_cuts.py \
    --bpe-model data/lang_bpe_100/bpe.model \
    --in-cuts fbank/train_cuts.jsonl.gz \
    --out-cuts fbank/train_cuts_filtered.jsonl.gz \
    --min-duration 0.5 \
    --max-duration 15.0
"""

import argparse
import logging
from pathlib import Path

import sentencepiece as spm
from lhotse import CutSet, load_manifest_lazy
from lhotse.cut import Cut


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--bpe-model",
        type=Path,
        required=True,
        help="Path to the bpe.model used for tokenizing transcripts.",
    )

    parser.add_argument(
        "--in-cuts",
        type=Path,
        required=True,
        help="Path to the input cut manifest (jsonl.gz).",
    )

    parser.add_argument(
        "--out-cuts",
        type=Path,
        required=True,
        help="Path to save the filtered cut manifest.",
    )

    parser.add_argument(
        "--min-duration",
        type=float,
        default=1.0,
        help="Remove cuts shorter than this many seconds.",
    )

    parser.add_argument(
        "--max-duration",
        type=float,
        default=20.0,
        help="Remove cuts longer than this many seconds.",
    )

    return parser.parse_args()


def filter_cuts(
    cut_set: CutSet,
    sp: spm.SentencePieceProcessor,
    min_duration: float,
    max_duration: float,
) -> CutSet:
    total = 0
    removed = 0

    def keep(c: Cut) -> bool:
        nonlocal removed, total
        total += 1

        if c.duration < min_duration or c.duration > max_duration:
            logging.warning(
                f"Remove cut {c.id}: duration={c.duration:.2f}s "
                f"(allowed [{min_duration}, {max_duration}])"
            )
            removed += 1
            return False

        # Pruned RNN-T requires T >= S after subsampling.
        # Zipformer2 subsampling formula: T = ((F - 7) // 2 + 1) // 2
        # where F = number of feature frames (at 10ms frame shift, F = duration * 100)
        if c.num_frames is not None:
            num_frames = c.num_frames
        else:
            num_frames = int(c.duration * 100)

        T = ((num_frames - 7) // 2 + 1) // 2

        tokens = sp.encode(c.supervisions[0].text, out_type=str)
        S = len(tokens)

        if T < S:
            logging.warning(
                f"Remove cut {c.id}: T={T} < S={S} tokens after subsampling. "
                f"frames={num_frames}, text='{c.supervisions[0].text}'"
            )
            removed += 1
            return False

        return True

    ans = cut_set.filter(keep).to_eager()
    ratio = removed / total * 100 if total > 0 else 0
    logging.info(
        f"Removed {removed}/{total} cuts ({ratio:.2f}%). "
        f"Remaining: {total - removed} cuts."
    )
    return ans


def main():
    args = get_args()
    logging.info(vars(args))

    if args.out_cuts.is_file():
        logging.info(f"{args.out_cuts} already exists — skipping. Delete it to re-run.")
        return

    assert args.in_cuts.is_file(), f"Input cuts not found: {args.in_cuts}"
    assert args.bpe_model.is_file(), f"BPE model not found: {args.bpe_model}"

    sp = spm.SentencePieceProcessor()
    sp.load(str(args.bpe_model))

    cut_set = load_manifest_lazy(args.in_cuts)
    assert isinstance(cut_set, CutSet)

    cut_set = filter_cuts(
        cut_set,
        sp,
        min_duration=args.min_duration,
        max_duration=args.max_duration,
    )

    logging.info(f"Saving to {args.out_cuts}")
    args.out_cuts.parent.mkdir(parents=True, exist_ok=True)
    cut_set.to_file(args.out_cuts)
    logging.info("Done.")


if __name__ == "__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    logging.basicConfig(format=formatter, level=logging.INFO)
    main()
