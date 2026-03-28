# Copyright      2021  Piotr Żelasko
# Copyright      2022  Xiaomi Corporation     (Author: Mingshuang Luo)
#
# Modified for custom vi_asr_corpus dataset

import argparse
import inspect
import logging
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from lhotse import CutSet, Fbank, FbankConfig, load_manifest, load_manifest_lazy
from lhotse.dataset import (
    CutConcatenate,
    CutMix,
    DynamicBucketingSampler,
    K2SpeechRecognitionDataset,
    PrecomputedFeatures,
    SimpleCutSampler,
    SpecAugment,
)
from lhotse.dataset.input_strategies import AudioSamples, OnTheFlyFeatures
from lhotse.utils import fix_random_seed
from torch.utils.data import DataLoader

from icefall.utils import str2bool


class _SeedWorkers:
    def __init__(self, seed: int):
        self.seed = seed

    def __call__(self, worker_id: int):
        fix_random_seed(self.seed + worker_id)


class ViAsrDataModule:
    """
    DataModule for custom Vietnamese ASR dataset.

    Expected manifest_dir structure:
      manifest_dir/
        train_cuts.jsonl.gz
        dev_cuts.jsonl.gz
        test_cuts.jsonl.gz

    Optionally, if MUSAN augmentation is enabled:
      manifest_dir/
        musan_cuts.jsonl.gz
    """

    def __init__(self, args: argparse.Namespace):
        self.args = args

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser):
        group = parser.add_argument_group(
            title="ASR data related options",
            description=(
                "These options control how Lhotse CutSets are turned into "
                "PyTorch DataLoaders."
            ),
        )

        group.add_argument(
            "--manifest-dir",
            type=Path,
            default=Path("fbank"),
            help="Path to directory containing train/dev/test cut manifests.",
        )

        group.add_argument(
            "--max-duration",
            type=float,
            default=200.0,
            help="Maximum pooled recordings duration (seconds) in a batch.",
        )

        group.add_argument(
            "--bucketing-sampler",
            type=str2bool,
            default=True,
            help="Use DynamicBucketingSampler if true, otherwise SimpleCutSampler.",
        )

        group.add_argument(
            "--num-buckets",
            type=int,
            default=30,
            help="Number of buckets for DynamicBucketingSampler.",
        )

        group.add_argument(
            "--concatenate-cuts",
            type=str2bool,
            default=False,
            help="Concatenate utterances to reduce padding.",
        )

        group.add_argument(
            "--duration-factor",
            type=float,
            default=1.0,
            help="Max duration of concatenated cut relative to longest cut in batch.",
        )

        group.add_argument(
            "--gap",
            type=float,
            default=1.0,
            help="Padding gap in seconds inserted between concatenated cuts.",
        )

        group.add_argument(
            "--on-the-fly-feats",
            type=str2bool,
            default=False,
            help="Use on-the-fly feature extraction instead of precomputed features.",
        )

        group.add_argument(
            "--shuffle",
            type=str2bool,
            default=True,
            help="Shuffle training data each epoch.",
        )

        group.add_argument(
            "--drop-last",
            type=str2bool,
            default=True,
            help="Whether sampler drops the last incomplete batch.",
        )

        group.add_argument(
            "--return-cuts",
            type=str2bool,
            default=True,
            help="Return cuts in batch['supervisions']['cut'].",
        )

        group.add_argument(
            "--num-workers",
            type=int,
            default=2,
            help="Number of DataLoader workers.",
        )

        group.add_argument(
            "--enable-spec-aug",
            type=str2bool,
            default=True,
            help="Use SpecAugment for training.",
        )

        group.add_argument(
            "--spec-aug-time-warp-factor",
            type=int,
            default=80,
            help="Time warp factor for SpecAugment. < 1 disables time warp.",
        )

        group.add_argument(
            "--enable-musan",
            type=str2bool,
            default=False,
            help=(
                "Enable MUSAN noise mixing. Requires manifest_dir/musan_cuts.jsonl.gz"
            ),
        )

        group.add_argument(
            "--input-strategy",
            type=str,
            default="PrecomputedFeatures",
            help="AudioSamples or PrecomputedFeatures",
        )

    def train_dataloaders(
        self,
        cuts_train: CutSet,
        sampler_state_dict: Optional[Dict[str, Any]] = None,
    ) -> DataLoader:
        transforms = []

        if self.args.enable_musan:
            musan_path = self.args.manifest_dir / "musan_cuts.jsonl.gz"
            if not musan_path.is_file():
                raise FileNotFoundError(
                    f"--enable-musan is True but MUSAN manifest not found: {musan_path}"
                )

            logging.info("Enable MUSAN")
            logging.info(f"Loading MUSAN cuts from {musan_path}")
            cuts_musan = load_manifest(musan_path)
            transforms.append(
                CutMix(cuts=cuts_musan, p=0.5, snr=(10, 20), preserve_id=True)
            )
        else:
            logging.info("Disable MUSAN")

        if self.args.concatenate_cuts:
            logging.info(
                f"Using cut concatenation with duration factor "
                f"{self.args.duration_factor} and gap {self.args.gap}."
            )
            transforms = [
                CutConcatenate(
                    duration_factor=self.args.duration_factor,
                    gap=self.args.gap,
                )
            ] + transforms

        input_transforms = []
        if self.args.enable_spec_aug:
            logging.info("Enable SpecAugment")
            logging.info(f"Time warp factor: {self.args.spec_aug_time_warp_factor}")

            num_frame_masks = 10
            num_frame_masks_parameter = inspect.signature(
                SpecAugment.__init__
            ).parameters["num_frame_masks"]
            if num_frame_masks_parameter.default == 1:
                num_frame_masks = 2

            logging.info(f"Num frame mask: {num_frame_masks}")
            input_transforms.append(
                SpecAugment(
                    time_warp_factor=self.args.spec_aug_time_warp_factor,
                    num_frame_masks=num_frame_masks,
                    features_mask_size=27,
                    num_feature_masks=2,
                    frames_mask_size=100,
                )
            )
        else:
            logging.info("Disable SpecAugment")

        logging.info("Creating train dataset")
        train = K2SpeechRecognitionDataset(
            input_strategy=eval(self.args.input_strategy)(),
            cut_transforms=transforms,
            input_transforms=input_transforms,
            return_cuts=self.args.return_cuts,
        )

        if self.args.on_the_fly_feats:
            train = K2SpeechRecognitionDataset(
                cut_transforms=transforms,
                input_strategy=OnTheFlyFeatures(Fbank(FbankConfig(num_mel_bins=80))),
                input_transforms=input_transforms,
                return_cuts=self.args.return_cuts,
            )

        if self.args.bucketing_sampler:
            logging.info("Using DynamicBucketingSampler.")
            
            train_sampler = DynamicBucketingSampler(
                cuts_train,
                max_duration=self.args.max_duration,
                shuffle=self.args.shuffle,
                num_buckets=self.args.num_buckets,
                buffer_size=self.args.num_buckets * 5000,
                drop_last=self.args.drop_last,
            )
        else:
            logging.info("Using SimpleCutSampler.")
            train_sampler = SimpleCutSampler(
                cuts_train,
                max_duration=self.args.max_duration,
                shuffle=self.args.shuffle,
            )

        if sampler_state_dict is not None:
            logging.info("Loading sampler state dict")
            train_sampler.load_state_dict(sampler_state_dict)

        seed = torch.randint(0, 100000, ()).item()
        worker_init_fn = _SeedWorkers(seed)

        train_dl = DataLoader(
            train,
            sampler=train_sampler,
            batch_size=None,
            num_workers=self.args.num_workers,
            persistent_workers=False,
            worker_init_fn=worker_init_fn,
        )
        return train_dl

    def valid_dataloaders(self, cuts_valid: CutSet) -> DataLoader:
        transforms = []
        if self.args.concatenate_cuts:
            transforms = [
                CutConcatenate(
                    duration_factor=self.args.duration_factor,
                    gap=self.args.gap,
                )
            ] + transforms

        logging.info("Creating validation dataset")
        if self.args.on_the_fly_feats:
            validate = K2SpeechRecognitionDataset(
                cut_transforms=transforms,
                input_strategy=OnTheFlyFeatures(Fbank(FbankConfig(num_mel_bins=80))),
                return_cuts=self.args.return_cuts,
            )
        else:
            validate = K2SpeechRecognitionDataset(
                cut_transforms=transforms,
                return_cuts=self.args.return_cuts,
            )

        # valid_sampler = DynamicBucketingSampler(
        #     cuts_valid,
        #     max_duration=self.args.max_duration,
        #     shuffle=False,
        # )

        # Sài thì dataset nhỏ
        logging.info("Using SimpleCutSampler for validation.")
        valid_sampler = SimpleCutSampler(
            cuts_valid,
            max_duration=self.args.max_duration,
            shuffle=False,
        )

        valid_dl = DataLoader(
            validate,
            sampler=valid_sampler,
            batch_size=None,
            num_workers=self.args.num_workers,
            persistent_workers=False,
        )
        return valid_dl

    def test_dataloaders(self, cuts: CutSet) -> DataLoader:
        logging.info("Creating test dataset")
        test = K2SpeechRecognitionDataset(
            input_strategy=OnTheFlyFeatures(Fbank(FbankConfig(num_mel_bins=80)))
            if self.args.on_the_fly_feats
            else eval(self.args.input_strategy)(),
            return_cuts=self.args.return_cuts,
        )

        # sampler = DynamicBucketingSampler(
        #     cuts,
        #     max_duration=self.args.max_duration,
        #     shuffle=False,
        # )

        # Sài thì dataset nhỏ
        logging.info("Using SimpleCutSampler for test.")
        sampler = SimpleCutSampler(
            cuts,
            max_duration=self.args.max_duration,
            shuffle=False,
        )

        test_dl = DataLoader(
            test,
            batch_size=None,
            sampler=sampler,
            num_workers=self.args.num_workers,
        )
        return test_dl

    @lru_cache()
    def load_manifest(self, manifest_path: Path) -> CutSet:
        logging.info(f"Loading manifest: {manifest_path}")
        return load_manifest_lazy(manifest_path)

    @lru_cache()
    def train_cuts(self) -> CutSet:
        path = self.args.manifest_dir / "train_cuts.jsonl.gz"
        logging.info(f"Loading train cuts from {path}")
        return load_manifest_lazy(path)

    @lru_cache()
    def dev_cuts(self) -> CutSet:
        path = self.args.manifest_dir / "dev_cuts.jsonl.gz"
        logging.info(f"Loading dev cuts from {path}")
        return load_manifest_lazy(path)

    @lru_cache()
    def test_cuts(self) -> CutSet:
        path = self.args.manifest_dir / "test_cuts.jsonl.gz"
        logging.info(f"Loading test cuts from {path}")
        return load_manifest_lazy(path)