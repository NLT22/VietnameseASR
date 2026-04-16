from pathlib import Path
import csv
import argparse

ROOT = Path(__file__).resolve().parents[1]


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
    transcript_dir = args.transcript_dir if args.transcript_dir.is_absolute() else ROOT / args.transcript_dir
    output_text = args.output_text if args.output_text.is_absolute() else ROOT / args.output_text
    train_tsv = transcript_dir / "train.tsv"
    out_txt = output_text
    out_txt.parent.mkdir(parents=True, exist_ok=True)

    with open(train_tsv, "r", encoding="utf-8") as f, open(out_txt, "w", encoding="utf-8") as g:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            text = " ".join(row["text"].strip().split())
            g.write(text + "\n")


if __name__ == "__main__":
    main()
