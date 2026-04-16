#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import sentencepiece as spm


ROOT = Path(__file__).resolve().parents[1]


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=30,
        help="Vocabulary size used to locate bpe.model at data/lang_bpe_<vocab_size>/bpe.model",
    )
    parser.add_argument(
        "--text",
        type=str,
        default="hôm nay tôi học nhận dạng tiếng nói",
        help="Text to tokenize",
    )
    return parser.parse_args()


def main():
    args = get_args()

    model_path = ROOT / f"data/lang_bpe_{args.vocab_size}/bpe.model"

    sp = spm.SentencePieceProcessor()
    ok = sp.load(str(model_path))
    if not ok:
        raise FileNotFoundError(f"Cannot load SentencePiece model: {model_path}")

    print(f"Model: {model_path}")
    print(f"Text: {args.text}")
    print("Pieces:", sp.encode(args.text, out_type=str))
    print("Ids:", sp.encode(args.text, out_type=int))


if __name__ == "__main__":
    main()
