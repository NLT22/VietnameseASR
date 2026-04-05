#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Optional

import sentencepiece as spm
import torch
from lhotse import CutSet, Fbank, FbankConfig, LilcomChunkyWriter, RecordingSet, SupervisionSet

torch.set_num_threads(1)
torch.set_num_interop_threads(1)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bpe-model", type=str, default="data/lang_bpe_100/bpe.model",
                        help="Path to bpe.model. Used to filter cuts by token/frame ratio.")
    parser.add_argument("--manifest-dir", type=str, default="manifests_fixed",
                        help="Directory containing *_recordings.jsonl.gz and *_supervisions.jsonl.gz")
    parser.add_argument("--output-dir", type=str, default="fbank",
                        help="Directory to save features and *_cuts.jsonl.gz")
    parser.add_argument("--dataset", type=str, default=None,
                        help='Space-separated parts to compute fbank, e.g. "train dev test". Default: all')
    parser.add_argument("--perturb-speed", action="store_true",
                        help="Apply 0.9x and 1.1x speed perturbation on train split")
    parser.add_argument("--num-mel-bins", type=int, default=80)
    parser.add_argument("--min-duration", type=float, default=0.3)
    parser.add_argument("--max-duration", type=float, default=20.0)
    return parser.parse_args()


def filter_cuts(
    cut_set: CutSet,
    sp: spm.SentencePieceProcessor,
    min_duration: float,
    max_duration: float,
) -> CutSet:
    def keep(c):
        if c.duration < min_duration or c.duration > max_duration:
            return False
        if len(c.supervisions) != 1:
            return False
        text = c.supervisions[0].text.strip()
        if not text:
            return False
        return True

    return cut_set.filter(keep)


def compute_fbank_for_split(
    split: str,
    manifest_dir: Path,
    output_dir: Path,
    extractor: Fbank,
    sp: Optional[spm.SentencePieceProcessor],
    perturb_speed: bool,
    min_duration: float,
    max_duration: float,
):
    rec_path = manifest_dir / f"{split}_recordings.jsonl.gz"
    sup_path = manifest_dir / f"{split}_supervisions.jsonl.gz"

    if not rec_path.is_file():
        raise FileNotFoundError(f"Missing recordings manifest: {rec_path}")
    if not sup_path.is_file():
        raise FileNotFoundError(f"Missing supervisions manifest: {sup_path}")

    cuts_path = output_dir / f"{split}_cuts.jsonl.gz"
    if cuts_path.is_file():
        logging.info(f"{cuts_path} already exists - skipping.")
        return

    recordings = RecordingSet.from_file(rec_path)
    supervisions = SupervisionSet.from_file(sup_path)

    logging.info(f"Building cuts for split={split}")
    cut_set = CutSet.from_manifests(recordings=recordings, supervisions=supervisions)

    if sp is not None:
        logging.info(f"Filtering cuts for split={split}")
        cut_set = filter_cuts(cut_set, sp=sp, min_duration=min_duration, max_duration=max_duration)

    if split == "train" and perturb_speed:
        logging.info("Applying speed perturbation 0.9x and 1.1x to train split")
        cut_set = cut_set + cut_set.perturb_speed(0.9) + cut_set.perturb_speed(1.1)

    num_jobs = min(15, os.cpu_count() or 1)

    cut_set = cut_set.compute_and_store_features(
        extractor=extractor,
        storage_path=str(output_dir / split),
        num_jobs=num_jobs,
        storage_type=LilcomChunkyWriter,
    )
    cut_set.to_file(cuts_path)
    logging.info(f"Saved {cuts_path}")


def main():
    args = get_args()
    logging.info(vars(args))

    manifest_dir = Path(args.manifest_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_parts = ("train", "dev", "test") if args.dataset is None else tuple(args.dataset.split())

    sp = None
    if args.bpe_model:
        bpe_path = Path(args.bpe_model)
        if not bpe_path.is_file():
            raise FileNotFoundError(f"Missing bpe model: {bpe_path}")
        sp = spm.SentencePieceProcessor()
        sp.load(str(bpe_path))

    extractor = Fbank(FbankConfig(num_mel_bins=args.num_mel_bins))

    for split in dataset_parts:
        compute_fbank_for_split(
            split=split,
            manifest_dir=manifest_dir,
            output_dir=output_dir,
            extractor=extractor,
            sp=sp,
            perturb_speed=args.perturb_speed,
            min_duration=args.min_duration,
            max_duration=args.max_duration,
        )


if __name__ == "__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    logging.basicConfig(format=formatter, level=logging.INFO)
    main()
