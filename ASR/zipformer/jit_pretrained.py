#!/usr/bin/env python3
# Copyright 2021-2023 Xiaomi Corporation (Author: Fangjun Kuang, Zengwei Yao)
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
Load a TorchScript model (exported by export.py --jit 1) and transcribe wav files.

First export the model:
  cd /path/to/icefall/egs/vi_asr_corpus
  python3 ASR/zipformer/export.py \
    --exp-dir ASR/zipformer/exp_bpe100_small \
    --tokens data/lang_bpe_100/tokens.txt \
    --epoch 50 \
    --avg 10 \
    --jit 1

Then run inference:
  python3 ASR/zipformer/jit_pretrained.py \
    --nn-model-filename ASR/zipformer/exp_bpe100_small/jit_script.pt \
    --tokens data/lang_bpe_100/tokens.txt \
    /path/to/audio1.wav \
    /path/to/audio2.wav
"""

import argparse
import logging
import math
from typing import List

import k2
import torch
import torchaudio
from torch.nn.utils.rnn import pad_sequence


def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--nn-model-filename",
        type=str,
        required=True,
        help="Path to the TorchScript model jit_script.pt",
    )

    parser.add_argument(
        "--tokens",
        type=str,
        required=True,
        help="Path to tokens.txt",
    )

    parser.add_argument(
        "sound_files",
        type=str,
        nargs="+",
        help="Wav files to transcribe. Must be 16 kHz mono.",
    )

    return parser


def read_sound_files(
    filenames: List[str], expected_sample_rate: float = 16000
) -> List[torch.Tensor]:
    """Read wav files into a list of 1-D float32 tensors."""
    ans = []
    for f in filenames:
        wave, sample_rate = torchaudio.load(f)
        assert sample_rate == expected_sample_rate, (
            f"Expected {expected_sample_rate} Hz, got {sample_rate} Hz: {f}"
        )
        ans.append(wave[0].contiguous())
    return ans


def compute_features(waves: List[torch.Tensor], device: torch.device) -> List[torch.Tensor]:
    """Compute 80-dim log-Mel filterbank features matching icefall's fbank config.

    Uses torchaudio.compliance.kaldi.fbank which replicates Kaldi's fbank
    with the same defaults used in lhotse/icefall:
      - 80 mel bins, snip_edges=False, dither=0, high_freq=-400 (relative to Nyquist)
    """
    features = []
    for wave in waves:
        # torchaudio.compliance.kaldi.fbank expects (channel, time)
        feats = torchaudio.compliance.kaldi.fbank(
            wave.unsqueeze(0).to(device),
            num_mel_bins=80,
            dither=0,
            snip_edges=False,
            high_freq=-400,
            sample_frequency=16000,
            use_log_fbank=True,
        )
        features.append(feats)
    return features


def greedy_search(
    model: torch.jit.ScriptModule,
    encoder_out: torch.Tensor,
    encoder_out_lens: torch.Tensor,
) -> List[List[int]]:
    assert encoder_out.ndim == 3
    assert encoder_out.size(0) >= 1

    packed_encoder_out = torch.nn.utils.rnn.pack_padded_sequence(
        input=encoder_out,
        lengths=encoder_out_lens.cpu(),
        batch_first=True,
        enforce_sorted=False,
    )

    device = encoder_out.device
    blank_id = model.decoder.blank_id
    context_size = model.decoder.context_size

    batch_size_list = packed_encoder_out.batch_sizes.tolist()
    N = encoder_out.size(0)

    assert torch.all(encoder_out_lens > 0), encoder_out_lens
    assert N == batch_size_list[0], (N, batch_size_list)

    hyps = [[blank_id] * context_size for _ in range(N)]

    decoder_input = torch.tensor(hyps, device=device, dtype=torch.int64)
    decoder_out = model.decoder(
        decoder_input,
        need_pad=torch.tensor([False]),
    ).squeeze(1)

    offset = 0
    for batch_size in batch_size_list:
        start = offset
        end = offset + batch_size
        current_encoder_out = packed_encoder_out.data[start:end]
        offset = end

        decoder_out = decoder_out[:batch_size]

        logits = model.joiner(current_encoder_out, decoder_out)
        assert logits.ndim == 2, logits.shape

        y = logits.argmax(dim=1).tolist()
        emitted = False
        for i, v in enumerate(y):
            if v != blank_id:
                hyps[i].append(v)
                emitted = True
        if emitted:
            decoder_input = [h[-context_size:] for h in hyps[:batch_size]]
            decoder_input = torch.tensor(decoder_input, device=device, dtype=torch.int64)
            decoder_out = model.decoder(
                decoder_input,
                need_pad=torch.tensor([False]),
            ).squeeze(1)

    sorted_ans = [h[context_size:] for h in hyps]
    ans = []
    unsorted_indices = packed_encoder_out.unsorted_indices.tolist()
    for i in range(N):
        ans.append(sorted_ans[unsorted_indices[i]])

    return ans


@torch.no_grad()
def main():
    parser = get_parser()
    args = parser.parse_args()
    logging.info(vars(args))

    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda", 0)
    logging.info(f"device: {device}")

    model = torch.jit.load(args.nn_model_filename)
    model.eval()
    model.to(device)

    logging.info(f"Reading sound files: {args.sound_files}")
    waves = read_sound_files(filenames=args.sound_files)
    waves = [w.to(device) for w in waves]

    logging.info("Computing features")
    features = compute_features(waves, device)
    feature_lengths = [f.size(0) for f in features]

    features = pad_sequence(
        features,
        batch_first=True,
        padding_value=math.log(1e-10),
    )
    feature_lengths = torch.tensor(feature_lengths, device=device)

    logging.info("Encoding")
    encoder_out, encoder_out_lens = model.encoder(
        features=features,
        feature_lengths=feature_lengths,
    )

    hyps = greedy_search(
        model=model,
        encoder_out=encoder_out,
        encoder_out_lens=encoder_out_lens,
    )

    token_table = k2.SymbolTable.from_file(args.tokens)

    def token_ids_to_text(token_ids: List[int]) -> str:
        text = ""
        for i in token_ids:
            text += token_table[i]
        return text.replace("▁", " ").strip()

    s = "\n"
    for filename, hyp in zip(args.sound_files, hyps):
        words = token_ids_to_text(hyp)
        s += f"{filename}:\n{words}\n"
    logging.info(s)


if __name__ == "__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    logging.basicConfig(format=formatter, level=logging.INFO)
    main()
