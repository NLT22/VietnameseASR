#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import sentencepiece as spm


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=30,
        help="Vocabulary size for BPE training. Default: 30",
    )
    parser.add_argument(
        "--input-txt",
        type=Path,
        default=Path("lang/transcript_words.txt"),
        help="Training text file. Default: lang/transcript_words.txt",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Project root. Default: repository recipe directory",
    )
    return parser.parse_args()


def generate_tokens(lang_dir: Path):
    sp = spm.SentencePieceProcessor()
    sp.load(str(lang_dir / "bpe.model"))
    token2id: Dict[str, int] = {sp.id_to_piece(i): i for i in range(sp.vocab_size())}

    with open(lang_dir / "tokens.txt", "w", encoding="utf-8") as f:
        for sym, i in token2id.items():
            f.write(f"{sym} {i}\n")


def main():
    args = get_args()

    root = args.root
    input_txt = root / args.input_txt
    vocab_size = args.vocab_size

    if not input_txt.is_file():
        raise FileNotFoundError(f"Missing training text: {input_txt}")

    lang_dir = root / f"data/lang_bpe_{vocab_size}"
    lang_dir.mkdir(parents=True, exist_ok=True)

    model_type = "bpe"
    model_prefix = str(lang_dir / "bpe")

    user_defined_symbols = ["<blk>", "<sos/eos>"]
    unk_id = len(user_defined_symbols)  # <blk>=0, <sos/eos>=1, <unk>=2

    spm.SentencePieceTrainer.train(
        input=str(input_txt),
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        model_type=model_type,
        character_coverage=1.0,
        user_defined_symbols=user_defined_symbols,
        unk_id=unk_id,
        bos_id=-1,
        eos_id=-1,
        pad_id=-1,
    )

    generate_tokens(lang_dir)

    print("Done")
    print(f"Saved to: {lang_dir}")


if __name__ == "__main__":
    main()
