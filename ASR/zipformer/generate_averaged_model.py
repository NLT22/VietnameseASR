#!/usr/bin/env python3
# Copyright 2021-2022 Xiaomi Corporation (Author: Yifan Yang)
#
# See ../../../../LICENSE for clarification regarding multiple authors
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
Average N consecutive checkpoints offline, without running decode.py.

Usage (epoch-based):
  cd /path/to/icefall/egs/vi_asr_corpus
  python3 ASR/zipformer/generate_averaged_model.py \
      --epoch 50 \
      --avg 10 \
      --exp-dir ASR/zipformer/exp_bpe100_small \
      --tokens data/lang_bpe_100/tokens.txt

  Output: ASR/zipformer/exp_bpe100_small/epoch-50-avg-10.pt

Usage (iter-based):
  python3 ASR/zipformer/generate_averaged_model.py \
      --iter 22000 \
      --avg 5 \
      --exp-dir ASR/zipformer/exp_bpe100_small \
      --tokens data/lang_bpe_100/tokens.txt

  Output: ASR/zipformer/exp_bpe100_small/iter-22000-avg-5.pt
"""

import argparse
from pathlib import Path

import k2
import torch
from train import add_model_arguments, get_model, get_params

from icefall.checkpoint import average_checkpoints_with_averaged_model, find_checkpoints


def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--epoch",
        type=int,
        default=50,
        help="Checkpoint epoch to end averaging at. Epoch counts from 1.",
    )

    parser.add_argument(
        "--iter",
        type=int,
        default=0,
        help="If positive, --epoch is ignored and iter-based checkpoints are used.",
    )

    parser.add_argument(
        "--avg",
        type=int,
        default=10,
        help="Number of checkpoints to average (consecutive, ending at --epoch/--iter).",
    )

    parser.add_argument(
        "--exp-dir",
        type=str,
        default="ASR/zipformer/exp_bpe100_small",
        help="Experiment directory containing epoch-N.pt checkpoints.",
    )

    parser.add_argument(
        "--tokens",
        type=str,
        default="data/lang_bpe_100/tokens.txt",
        help="Path to tokens.txt (used to infer vocab_size).",
    )

    parser.add_argument(
        "--context-size",
        type=int,
        default=2,
        help="Decoder context size. 1=bigram, 2=trigram.",
    )

    add_model_arguments(parser)

    return parser


@torch.no_grad()
def main():
    parser = get_parser()
    args = parser.parse_args()
    args.exp_dir = Path(args.exp_dir)

    params = get_params()
    params.update(vars(args))

    if params.iter > 0:
        params.suffix = f"iter-{params.iter}-avg-{params.avg}"
    else:
        params.suffix = f"epoch-{params.epoch}-avg-{params.avg}"

    print(f"exp_dir : {params.exp_dir}")
    print(f"suffix  : {params.suffix}")

    device = torch.device("cpu")

    symbol_table = k2.SymbolTable.from_file(params.tokens)
    params.blank_id = symbol_table["<blk>"]
    params.unk_id = symbol_table["<unk>"]
    params.vocab_size = len(symbol_table)

    print(f"vocab_size: {params.vocab_size}")
    model = get_model(params)

    if params.iter > 0:
        filenames = find_checkpoints(params.exp_dir, iteration=-params.iter)[
            : params.avg + 1
        ]
        if len(filenames) == 0:
            raise ValueError(
                f"No checkpoints found for --iter {params.iter}, --avg {params.avg}"
            )
        elif len(filenames) < params.avg + 1:
            raise ValueError(
                f"Not enough checkpoints ({len(filenames)}) found for"
                f" --iter {params.iter}, --avg {params.avg}"
            )
        filename_start = filenames[-1]
        filename_end = filenames[0]
        print(
            f"Averaging iter checkpoints from {filename_start} (excluded) to {filename_end}"
        )
        model.to(device)
        model.load_state_dict(
            average_checkpoints_with_averaged_model(
                filename_start=filename_start,
                filename_end=filename_end,
                device=device,
            )
        )
        filename = params.exp_dir / f"iter-{params.iter}-avg-{params.avg}.pt"
        torch.save({"model": model.state_dict()}, filename)
    else:
        assert params.avg > 0, params.avg
        start = params.epoch - params.avg
        assert start >= 1, start
        filename_start = f"{params.exp_dir}/epoch-{start}.pt"
        filename_end = f"{params.exp_dir}/epoch-{params.epoch}.pt"
        print(
            f"Averaging epoch range from {start} (excluded) to {params.epoch}"
        )
        model.to(device)
        model.load_state_dict(
            average_checkpoints_with_averaged_model(
                filename_start=filename_start,
                filename_end=filename_end,
                device=device,
            )
        )
        filename = params.exp_dir / f"epoch-{params.epoch}-avg-{params.avg}.pt"
        torch.save({"model": model.state_dict()}, filename)

    num_param = sum([p.numel() for p in model.parameters()])
    print(f"Number of model parameters: {num_param}")
    print(f"Saved averaged model to: {filename}")


if __name__ == "__main__":
    main()
