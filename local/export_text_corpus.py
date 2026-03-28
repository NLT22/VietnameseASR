from pathlib import Path
import csv

ROOT = Path("./")
train_tsv = ROOT / "transcripts" / "train.tsv"
out_txt = ROOT / "lang" / "transcript_words.txt"
out_txt.parent.mkdir(parents=True, exist_ok=True)

with open(train_tsv, "r", encoding="utf-8") as f, open(out_txt, "w", encoding="utf-8") as g:
    reader = csv.DictReader(f, delimiter="\t")
    for row in reader:
        text = " ".join(row["text"].strip().split())
        g.write(text + "\n")
