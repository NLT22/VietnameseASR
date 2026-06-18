#!/usr/bin/env python3
"""Smoke-test the TorchScript mic model on wavs listed in transcript TSV files."""

import argparse
import csv
import math
import sys
from pathlib import Path
from typing import Dict, List

import k2
import soundfile as sf
import torch
import torchaudio
from torch.nn.utils.rnn import pad_sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mic_streaming_asr import (  # noqa: E402
    FULL_MODEL,
    SAMPLE_RATE,
    TOKENS_PATH,
    compute_fbank,
    jit_beam_search,
    jit_greedy_search,
    token_ids_to_text,
)


def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--transcript", default="transcripts/test.tsv")
    parser.add_argument("--model", default=FULL_MODEL)
    parser.add_argument("--tokens", default=TOKENS_PATH)
    parser.add_argument("--decode-method", choices=["beam", "greedy"], default="beam")
    parser.add_argument("--beam", type=int, default=4)
    parser.add_argument(
        "--max-per-text",
        type=int,
        default=1,
        help="Keep at most this many utterances for each unique reference text.",
    )
    parser.add_argument(
        "--utt-id",
        action="append",
        default=[],
        help="Decode only this utterance ID. Can be passed multiple times.",
    )
    return parser


def read_rows(path: Path, max_per_text: int, utt_ids: List[str]) -> List[Dict[str, str]]:
    rows = []
    counts: Dict[str, int] = {}
    requested = set(utt_ids)
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            if requested and row["utt_id"] not in requested:
                continue
            text = row["text"]
            counts[text] = counts.get(text, 0) + 1
            if not requested and counts[text] > max_per_text:
                continue
            rows.append(row)
    return rows


def load_wave(path: Path, device: torch.device) -> torch.Tensor:
    data, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    wave = torch.from_numpy(data[:, 0]).unsqueeze(0)
    if sample_rate != SAMPLE_RATE:
        wave = torchaudio.functional.resample(wave, sample_rate, SAMPLE_RATE)
    return wave[0].contiguous().to(device)


def edit_distance(ref: List[str], hyp: List[str]) -> int:
    dp = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, start=1):
        prev = dp[0]
        dp[0] = i
        for j, h in enumerate(hyp, start=1):
            old = dp[j]
            dp[j] = min(
                dp[j] + 1,
                dp[j - 1] + 1,
                prev + (0 if r == h else 1),
            )
            prev = old
    return dp[-1]


@torch.no_grad()
def decode_one(model, wave: torch.Tensor, token_table, args, device: torch.device) -> str:
    # Match mic full-context behavior: add tail padding, fbank whole utterance.
    tail_padding = torch.zeros(int(0.3 * SAMPLE_RATE), device=device)
    feats = compute_fbank(torch.cat([wave, tail_padding]), device)
    features = pad_sequence(
        [feats],
        batch_first=True,
        padding_value=math.log(1e-10),
    )
    feature_lens = torch.tensor([feats.size(0)], device=device)
    encoder_out, encoder_out_lens = model.encoder(
        features=features,
        feature_lengths=feature_lens,
    )
    if args.decode_method == "greedy":
        token_ids = jit_greedy_search(model, encoder_out, encoder_out_lens, device)
    else:
        token_ids = jit_beam_search(model, encoder_out, args.beam, device)
    return token_ids_to_text(token_ids, token_table)


def main():
    args = get_parser().parse_args()
    transcript = ROOT / args.transcript
    model_path = ROOT / args.model
    tokens = ROOT / args.tokens

    rows = read_rows(transcript, args.max_per_text, args.utt_id)
    if not rows:
        raise SystemExit(f"No utterances selected from {transcript}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.jit.load(str(model_path), map_location=device)
    model.eval().to(device)
    token_table = k2.SymbolTable.from_file(str(tokens))

    total_err = 0
    total_words = 0

    print(f"model: {model_path.relative_to(ROOT)}")
    print(f"transcript: {transcript.relative_to(ROOT)}")
    print(f"decode: {args.decode_method}, beam={args.beam}")
    print()

    for row in rows:
        wav = ROOT / row["audio_path"]
        hyp = decode_one(model, load_wave(wav, device), token_table, args, device)
        ref = row["text"]
        err = edit_distance(ref.split(), hyp.split())
        total_err += err
        total_words += len(ref.split())

        print(f"utt_id: {row['utt_id']}")
        print(f"wav:    {row['audio_path']}")
        print(f"REF:    {ref}")
        print(f"HYP:    {hyp}")
        print(f"WER:    {100.0 * err / max(1, len(ref.split())):.2f}")
        print()

    print(f"TOTAL_WER: {100.0 * total_err / max(1, total_words):.2f}")


if __name__ == "__main__":
    main()
