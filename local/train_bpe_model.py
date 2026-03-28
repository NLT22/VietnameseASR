# local/train_bpe_model.py
from pathlib import Path
import sentencepiece as spm

ROOT = Path("./")
input_txt = ROOT / "lang" / "transcript_words.txt"
lang_dir = ROOT / "data" / "lang_bpe_500"
lang_dir.mkdir(parents=True, exist_ok=True)

spm.SentencePieceTrainer.Train(
    input=str(input_txt),
    model_prefix=str(lang_dir / "bpe"),
    vocab_size=500,
    model_type="bpe",
    character_coverage=1.0,
    bos_id=-1,
    eos_id=-1,
    unk_id=0,
    pad_id=-1,
    user_defined_symbols=[]
)
print("Done")
