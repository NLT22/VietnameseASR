from pathlib import Path
import csv
import argparse

ROOT = Path("./")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--transcript-dir",
        type=Path,
        default=Path("transcripts"),
        help="Directory containing train.tsv.",
    )
    parser.add_argument(
        "--output-text",
        type=Path,
        default=Path("lang/transcript_words.txt"),
        help="Output text corpus path.",
    )
    return parser.parse_args()


def main():
    args = get_args()
    train_tsv = ROOT / args.transcript_dir / "train.tsv"
    out_txt = ROOT / args.output_text
    out_txt.parent.mkdir(parents=True, exist_ok=True)

    with open(train_tsv, "r", encoding="utf-8") as f, open(out_txt, "w", encoding="utf-8") as g:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            text = " ".join(row["text"].strip().split())
            g.write(text + "\n")


if __name__ == "__main__":
    main()
