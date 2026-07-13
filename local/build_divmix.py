#!/usr/bin/env python3
"""
Assembles the divmix training set: real recordings repeated `--mult` times,
plus cross-speaker clones and diverse-voice clones. dev/test are written as
the clean real recordings (no repeats, no clones). Used by run.sh stage -2.
"""
import argparse
import collections
import csv
import os

VI = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

FIELDNAMES = ["utt_id", "speaker", "audio_path", "text"]


def _read_tsv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def _write_tsv(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDNAMES, delimiter="\t")
        w.writeheader()
        w.writerows(rows)


def build_divmix(real_mult, cross_tsv, div_tsv, out_tsv):
    """real x N (all 1000) + cross-speaker clones + diverse clones."""
    real = _read_tsv(os.path.join(VI, "transcripts_matched", "train.tsv"))
    cross = [r for r in _read_tsv(cross_tsv) if r["audio_path"].startswith("audio_crossspk/")]
    div = _read_tsv(div_tsv)

    rows = []
    for k in range(real_mult):
        for r in real:
            rr = dict(r)
            rr["utt_id"] = f'{r["utt_id"]}_r{k}'
            rows.append(rr)
    rows += cross + div

    _write_tsv(out_tsv, rows)

    missing = sum(1 for r in rows if not os.path.exists(os.path.join(VI, r["audio_path"])))
    voices = collections.Counter(r["speaker"] for r in rows)
    print(f"divmix: {len(rows)} rows ({len(real) * real_mult} real x{real_mult}), "
          f"missing_audio={missing}, voices={dict(voices)}")

    out_dir = os.path.dirname(out_tsv)
    for split in ("dev", "test"):
        _write_tsv(os.path.join(out_dir, f"{split}.tsv"), real)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mult", type=int, default=8)
    ap.add_argument("--cross", required=True)
    ap.add_argument("--div", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    build_divmix(a.mult, a.cross, a.div, a.out)


if __name__ == "__main__":
    main()
