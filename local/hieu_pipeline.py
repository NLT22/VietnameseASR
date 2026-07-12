#!/usr/bin/env python3
"""
Helpers for the Hieu-integration retrain pipeline. Each subcommand is one step;
the orchestrator run_hieu_pipeline.sh chains them. Kept in python (not bash) so
the % sign and Vietnamese text parse correctly -- the vocab analyzer's regex bug
was exactly this.
"""
import argparse, csv, glob, json, os, re, sys, unicodedata
import numpy as np
import soundfile as sf

VI = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def best_vocab():
    """Parse results/vocab_experiment.md + the LR baseline; print the best vocab.
    Baseline vocab 100 = 2.08%. Switch only if another vocab beats it clearly
    (>0.5 pt, since the 578-word held-out set has ~1 pt noise)."""
    res = {100: 2.08}
    # vocab_corrections.md overrides/supplements vocab_experiment.md (used when an
    # export bug forced a re-sweep). Both same format; take the min per vocab.
    for fn in ("vocab_experiment.md", "vocab_corrections.md"):
        p = os.path.join(VI, "results", fn)
        if not os.path.isfile(p):
            continue
        cur = None
        for line in open(p):
            m = re.search(r"VOCAB=(\d+)", line)
            if m:
                cur = int(m.group(1)); res.setdefault(cur, 999.0)
            mm = re.match(r"\|\s*\d+\s*\|\s*([\d.]+)%", line)   # | epoch | WER% |
            if mm and cur:
                res[cur] = min(res[cur], float(mm.group(1)))
    best = min(res, key=res.get)
    # prefer 100 unless clearly beaten
    if res[best] > res[100] - 0.5:
        best = 100
    sys.stderr.write(f"vocab held-out mins: {res}; chosen={best}\n")
    print(best)


def ingest(src_dir, peak=0.7):
    """Normalize Hieu wavs to `peak` and install to dataset/Hieu/ (+ script)."""
    out = os.path.join(VI, "dataset", "Hieu")
    bk = os.path.join(VI, "dataset_backup_hieu_none")   # first real Hieu; note it
    os.makedirs(out, exist_ok=True)
    n = 0
    for f in glob.glob(os.path.join(src_dir, "*.wav")):
        w, sr = sf.read(f, dtype="float32")
        w = w.mean(1) if w.ndim > 1 else w
        pk = np.abs(w).max()
        if pk > 0:
            w = w / pk * peak
        sf.write(os.path.join(out, os.path.basename(f)), w, sr, subtype="PCM_16")
        n += 1
    txt = glob.glob(os.path.join(src_dir, "*.txt"))[0]
    lines = [l.rstrip("\n").rstrip("\r") for l in open(txt, encoding="utf-8")]
    open(os.path.join(out, "script.txt"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print(f"installed {n} Hieu wavs -> {out}")


def hieu_sentences(out_tsv):
    """Write a source tsv of Hieu's rows only (for incremental cloning)."""
    rows = [r for r in csv.DictReader(
        open(os.path.join(VI, "transcripts_matched_u20", "train.tsv")), delimiter="\t")
        if r["speaker"] == "Hieu"]
    os.makedirs(os.path.dirname(out_tsv), exist_ok=True)
    w = csv.DictWriter(open(out_tsv, "w", newline=""),
                       fieldnames=["utt_id", "speaker", "audio_path", "text"], delimiter="\t")
    w.writeheader(); w.writerows(rows)
    print(f"wrote {len(rows)} Hieu sentences -> {out_tsv}")


def merge_tsv(base, extra, out):
    """Concatenate two clone tsvs, de-duping by utt_id."""
    def rd(p): return list(csv.DictReader(open(p), delimiter="\t")) if os.path.isfile(p) else []
    seen, rows = set(), []
    for r in rd(base) + rd(extra):
        if r["utt_id"] in seen: continue
        seen.add(r["utt_id"]); rows.append(r)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    w = csv.DictWriter(open(out, "w", newline=""),
                       fieldnames=["utt_id", "speaker", "audio_path", "text"], delimiter="\t")
    w.writeheader(); w.writerows(rows)
    print(f"merged -> {out}: {len(rows)} rows")


def build_divmix(real_mult, cross_tsv, div_tsv, out_tsv):
    """real x N (all 1000) + cross-speaker clones + diverse clones."""
    def rd(p): return list(csv.DictReader(open(p), delimiter="\t"))
    real = rd(os.path.join(VI, "transcripts_matched_u20", "train.tsv"))
    cross = [r for r in rd(cross_tsv) if r["audio_path"].startswith("audio_crossspk/")]
    div = rd(div_tsv)
    rows = []
    for k in range(real_mult):
        for r in real:
            rr = dict(r); rr["utt_id"] = f'{r["utt_id"]}_r{k}'; rows.append(rr)
    rows += cross + div
    os.makedirs(os.path.dirname(out_tsv), exist_ok=True)
    w = csv.DictWriter(open(out_tsv, "w", newline=""),
                       fieldnames=["utt_id", "speaker", "audio_path", "text"], delimiter="\t")
    w.writeheader(); w.writerows(rows)
    miss = sum(1 for r in rows if not os.path.exists(os.path.join(VI, r["audio_path"])))
    import collections
    spk = collections.Counter(r["speaker"] for r in rows)
    print(f"divmix: {len(rows)} rows ({len(real)*real_mult} real x{real_mult}), "
          f"missing_audio={miss}, voices={dict(spk)}")
    for s in ("dev", "test"):
        d = os.path.join(os.path.dirname(out_tsv), f"{s}.tsv")
        wr = csv.DictWriter(open(d, "w", newline=""),
                            fieldnames=["utt_id", "speaker", "audio_path", "text"], delimiter="\t")
        wr.writeheader(); wr.writerows(real)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd")
    sub.add_parser("best-vocab")
    p = sub.add_parser("ingest"); p.add_argument("--src", required=True); p.add_argument("--peak", type=float, default=0.7)
    p = sub.add_parser("hieu-sentences"); p.add_argument("--out", required=True)
    p = sub.add_parser("merge-tsv"); p.add_argument("--base", required=True); p.add_argument("--extra", required=True); p.add_argument("--out", required=True)
    p = sub.add_parser("build-divmix"); p.add_argument("--mult", type=int, default=8); p.add_argument("--cross", required=True); p.add_argument("--div", required=True); p.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.cmd == "best-vocab": best_vocab()
    elif a.cmd == "ingest": ingest(a.src, a.peak)
    elif a.cmd == "hieu-sentences": hieu_sentences(a.out)
    elif a.cmd == "merge-tsv": merge_tsv(a.base, a.extra, a.out)
    elif a.cmd == "build-divmix": build_divmix(a.mult, a.cross, a.div, a.out)
    else: ap.print_help()
