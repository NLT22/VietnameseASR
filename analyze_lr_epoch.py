#!/usr/bin/env python3
"""
Autonomous analysis of the LR x epoch experiment. Runs detached AFTER the
orchestrator writes DONE. Produces the decision-ready conclusion with NO agent:
  - parse coarse held-out sweep -> best epoch per LR
  - fine sweep +-2 epochs around each LR's best
  - for the overall best (LR, epoch): per-speaker WER on the 800 real recordings
    (answers the Trung de-clipping question)
  - append a Conclusions section to results/lr_epoch_experiment.md
"""
import csv, glob, os, re, subprocess, sys, time
import numpy as np

VI = "/media/pc/c88ba509-53f0-4c97-9e44-e33483754b08/icefall/egs/VietnameseASR"
VENV = "/media/pc/c88ba509-53f0-4c97-9e44-e33483754b08/test-icefall/bin"
GWEN = "/media/pc/c88ba509-53f0-4c97-9e44-e33483754b08/gwen-tts/.venv-gwen/bin/python"
RES = os.path.join(VI, "results", "lr_epoch_experiment.md")
os.chdir(VI)
sys.path.insert(0, os.path.join(VI, "deploy", "jetson_nano"))

LRS = [("0.01", "newT_lr001"), ("0.02", "newT_lr002"), ("0.045", "newT_lr0045")]


def log(msg):
    with open(RES, "a") as f:
        f.write(msg + "\n")


def export(tag, epoch):
    exp = f"ASR/zipformer/exp_bpe100_small_streaming_divmix_x8_{tag}"
    out = f"deploy/jetson_nano/sweep_{tag}_e{epoch}"
    if os.path.isfile(f"{out}/encoder.int8.onnx"):
        return out
    if not os.path.isfile(f"{exp}/epoch-{epoch}.pt"):
        return None
    env = dict(os.environ, PATH=VENV + ":" + os.environ["PATH"])
    subprocess.run(["bash", "local/export_for_jetson.sh", "--exp-dir", exp,
                    "--epoch", str(epoch), "--avg", "10", "--streaming", "1",
                    "--use-averaged-model", "1", "--out-dir", os.path.join(VI, out)],
                   env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for f in glob.glob(f"{out}/*.onnx"):
        if not f.endswith(".int8.onnx"):
            os.remove(f)
    return out if os.path.isfile(f"{out}/encoder.int8.onnx") else None


def heldout(model_dir):
    out = subprocess.run([GWEN, "eval_heldout_speaker.py", "--model-dir", model_dir],
                         capture_output=True, text=True).stdout
    ov = re.search(r"OVERALL\s+WER\s+([\d.]+)", out)
    return float(ov.group(1)) if ov else None


def real_per_speaker(model_dir):
    """Per-speaker WER on the 800 real recordings (the Trung answer)."""
    from jetson_beam_decode import OnnxTransducer, beam_search, load_tokens, detok, edit_distance
    from transcribe_beam_wav import load_wav_any
    from jetson_asr import compute_fbank
    m = OnnxTransducer(model_dir)
    id2 = load_tokens(os.path.join(model_dir, "tokens.txt"))
    rows = list(csv.DictReader(open("transcripts_matched_u20/test.tsv"), delimiter="\t"))
    bys, te, tn = {}, 0, 0
    for r in rows:
        ids = beam_search(m, m.encode(compute_fbank(load_wav_any(r["audio_path"]))), 4)
        e = edit_distance(r["text"].split(), detok(ids, id2).split())
        n = len(r["text"].split())
        te += e; tn += n
        s = bys.setdefault(r["speaker"], [0, 0]); s[0] += e; s[1] += n
    return {k: 100 * v[0] / v[1] for k, v in bys.items()}, 100 * te / tn


def parse_coarse():
    """From the md tables, get {tag: [(epoch, wer), ...]}."""
    txt = open(RES).read()
    res = {}
    cur = None
    for line in txt.splitlines():
        m = re.search(r"TRAIN LR=([\d.]+) ->.*_(newT_lr\w+)", line)
        if m:
            cur = m.group(2); res[cur] = []
        mm = re.match(r"\|\s*(\d+)\s*\|\s*([\d.]+)\s*\|", line)
        if mm and cur:
            res[cur].append((int(mm.group(1)), float(mm.group(2))))
    return res


def main():
    # wait for the orchestrator
    while not (os.path.isfile(RES) and re.search(r"^DONE", open(RES).read(), re.M)):
        time.sleep(120)

    log("\n---\n\n## Analysis (autonomous)\n")
    coarse = parse_coarse()

    # fine sweep +-2 around each LR's coarse best
    best_overall = (None, None, 999)
    for lr, tag in LRS:
        pts = coarse.get(tag, [])
        if not pts:
            log(f"- **LR {lr}**: no results"); continue
        be, bw = min(pts, key=lambda x: x[1])
        log(f"### LR {lr} — fine sweep around epoch {be} (coarse best {bw}%)\n")
        log("| epoch | held-out WER |\n| ---: | ---: |")
        fine = list(pts)
        for e in (be - 2, be - 1, be + 1, be + 2):
            if not (30 <= e <= 60) or any(p[0] == e for p in fine):
                continue
            md = export(tag, e)
            w = heldout(md) if md else None
            if w is not None:
                fine.append((e, w)); log(f"| {e} | {w} |")
        fbe, fbw = min(fine, key=lambda x: x[1])
        log(f"\n=> LR {lr} best: **epoch {fbe}, held-out {fbw}%**\n")
        if fbw < best_overall[2]:
            best_overall = (tag, fbe, fbw)

    # Trung answer at the overall best
    tag, ep, w = best_overall
    lr = dict((t, l) for l, t in LRS).get(tag, "?")
    log(f"## Winner: LR {lr}, epoch {ep} (held-out {w}%)\n")
    md = export(tag, ep)
    if md:
        per, overall = real_per_speaker(md)
        log(f"Per-speaker WER on the 800 real recordings (real-set overall {overall:.2f}%):\n")
        log("| speaker | WER |\n| --- | ---: |")
        for s in sorted(per):
            log(f"| {s} | {per[s]:.2f}% |")
        tr = per.get("Trung")
        others = np.mean([per[s] for s in per if s != "Trung"])
        log(f"\n**Trung {tr:.2f}%** vs others' mean {others:.2f}%. "
            f"Previous (de-clipped Trung) was 4.18-5.74%. "
            f"{'Gap CLOSED -> de-clipping was hurting him.' if tr < others*1.5 else 'Gap PERSISTS -> not the de-clipping (his voice/conditions).'}\n")
    log("ANALYSIS_DONE\n")


if __name__ == "__main__":
    main()
