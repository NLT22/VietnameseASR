from pathlib import Path
import csv
import soundfile as sf

ROOT = Path("./")
def audit(split):
    tsv = ROOT / "transcripts" / f"{split}.tsv"
    bad = 0
    total = 0

    with open(tsv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            total += 1
            wav = ROOT / row["audio_path"]
            text = row["text"].strip()

            if not wav.is_file():
                print("Missing:", wav)
                bad += 1
                continue

            info = sf.info(str(wav))
            if info.samplerate != 16000:
                print("Bad sr:", wav, info.samplerate)
                bad += 1
            if info.channels != 1:
                print("Bad channels:", wav, info.channels)
                bad += 1
            if info.duration < 0.3 or info.duration > 30:
                print("Strange duration:", wav, info.duration)
            if not text:
                print("Empty text:", wav)
                bad += 1

    print(split, "total =", total, "bad =", bad)

for s in ["train", "dev", "test"]:
    audit(s)
