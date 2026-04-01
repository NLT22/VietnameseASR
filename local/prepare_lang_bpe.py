#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import k2
import sentencepiece as spm


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang-dir", type=Path, required=True,
                        help="Directory containing bpe.model")
    return parser.parse_args()


def write_words_txt(lang_dir: Path):
    words = ["<eps>", "!SIL", "<SPOKEN_NOISE>", "<UNK>", "#0", "<s>", "</s>"]
    with open(lang_dir / "words.txt", "w", encoding="utf-8") as f:
        for i, w in enumerate(words):
            f.write(f"{w} {i}\n")


def main():
    args = get_args()
    lang_dir = args.lang_dir
    model_file = lang_dir / "bpe.model"
    if not model_file.is_file():
        raise FileNotFoundError(f"Missing {model_file}")

    sp = spm.SentencePieceProcessor()
    sp.load(str(model_file))

    if not (lang_dir / "tokens.txt").is_file():
        with open(lang_dir / "tokens.txt", "w", encoding="utf-8") as f:
            for i in range(sp.vocab_size()):
                f.write(f"{sp.id_to_piece(i)} {i}\n")

    write_words_txt(lang_dir)

    _ = k2.SymbolTable.from_file(lang_dir / "tokens.txt")
    _ = k2.SymbolTable.from_file(lang_dir / "words.txt")
    print(f"Prepared minimal lang dir at: {lang_dir}")


if __name__ == "__main__":
    main()
