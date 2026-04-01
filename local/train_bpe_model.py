#!/usr/bin/env python3
from pathlib import Path
from typing import Dict
import shutil
import sentencepiece as spm


def generate_tokens(lang_dir: Path):
    sp = spm.SentencePieceProcessor()
    sp.load(str(lang_dir / "bpe.model"))
    token2id: Dict[str, int] = {sp.id_to_piece(i): i for i in range(sp.vocab_size())}

    with open(lang_dir / "tokens.txt", "w", encoding="utf-8") as f:
        for sym, i in token2id.items():
            f.write(f"{sym} {i}\n")


ROOT = Path("./")
input_txt = ROOT / "lang" / "transcript_words.txt"

vocab_size = 100
lang_dir = ROOT / f"data/lang_bpe_{vocab_size}"
lang_dir.mkdir(parents=True, exist_ok=True)

model_type = "bpe"
model_prefix = str(lang_dir / "bpe")

user_defined_symbols = ["<blk>", "<sos/eos>"]
unk_id = len(user_defined_symbols)   # <blk>=0, <sos/eos>=1, <unk>=2

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

# bpe.model và bpe.vocab đã nằm đúng chỗ vì model_prefix=lang_dir/bpe
generate_tokens(lang_dir)

print("Done")
print(f"Saved to: {lang_dir}")