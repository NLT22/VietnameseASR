#!/usr/bin/env python3
"""
Repair hard-clipped recordings by cubic-interpolating the flat-topped runs.

Clipped samples were never stored, so this reconstructs a plausible peak from
the clean samples bracketing each run -- it does not recover the original.
Valid only while runs stay short relative to the pitch period; a run longer
than ~1/4 period is fabrication, not interpolation, so we refuse those.

Usage:
  python3 local/declip.py --src DIR --dst DIR [--peak 0.7]
  python3 local/declip.py --self-check
"""
import argparse
import glob
import os

import numpy as np
import soundfile as sf

SAT = 0.999
PAD = 8           # clean samples each side used to fit the cubic
MAX_RUN = 100     # ponytail: 1/4 pitch period @48k/120Hz; longer = refuse, re-record


def declip(w, sr, max_run=MAX_RUN):
    """Return (repaired, n_runs, n_refused). w is mono float32 in [-1, 1]."""
    w = w.astype(np.float64).copy()
    sat = (np.abs(w) >= SAT).view(np.int8)
    edges = np.flatnonzero(np.diff(np.concatenate(([0], sat, [0]))))
    starts, ends = edges[0::2], edges[1::2]
    refused = 0
    scale = max_run * sr / 48000.0
    for s, e in zip(starts, ends):
        if e - s > scale:
            refused += 1
            continue
        l0, l1 = max(0, s - PAD), s
        r0, r1 = e, min(len(w), e + PAD)
        if l1 - l0 < 2 or r1 - r0 < 2:
            continue
        xi = np.concatenate([np.arange(l0, l1), np.arange(r0, r1)])
        if np.abs(w[xi]).max() >= SAT:   # neighbours also clipped: skip
            continue
        # Cubic Hermite across the gap. Boundary slopes (not just values) are what
        # make the curve arc over the ceiling; a plain fit through the outside
        # samples underestimates the peak and gets pinned flat again.
        n = e - s
        yl, yr = w[s - 1], w[e]
        dl = w[s - 1] - w[s - 2]
        dr = w[e + 1] - w[e]
        t = (np.arange(n) + 1.0) / (n + 1.0)
        h00 = 2 * t**3 - 3 * t**2 + 1
        h10 = t**3 - 2 * t**2 + t
        h01 = -2 * t**3 + 3 * t**2
        h11 = t**3 - t**2
        m = n + 1.0
        fill = h00 * yl + h10 * (dl * m) + h01 * yr + h11 * (dr * m)
        sign = np.sign(w[s])
        # the true peak lay beyond the ceiling; never pull it back inside
        fill = sign * np.maximum(np.abs(fill), SAT)
        w[s:e] = fill
    return w, len(starts), refused


def process(src, dst, peak):
    os.makedirs(dst, exist_ok=True)
    files = sorted(glob.glob(os.path.join(src, "*.wav")))
    tot_runs = tot_ref = 0
    for f in files:
        w, sr = sf.read(f, dtype="float32")
        if w.ndim > 1:
            w = w.mean(1)
        w, n, r = declip(w, sr)
        tot_runs += n
        tot_ref += r
        m = np.abs(w).max()
        if m > 0:
            w = w * (peak / m)
        sf.write(os.path.join(dst, os.path.basename(f)), w.astype(np.float32), sr,
                 subtype="PCM_16")
    print(f"{len(files)} files, {tot_runs} clipped runs repaired, {tot_ref} refused "
          f"(too long), peak-normalized to {peak}")


def self_check():
    sr = 48000
    t = np.arange(sr // 10) / sr
    clean = 0.9 * np.sin(2 * np.pi * 120 * t) + 0.3 * np.sin(2 * np.pi * 360 * t)
    clean /= np.abs(clean).max()
    drive = 1.08                                   # ~Trung's level: runs << pitch period
    hot = np.clip(clean * drive, -1.0, 1.0)
    fixed, n, refused = declip(hot, sr)
    assert n > 0, "self-check needs actual clipping"
    assert refused == 0, f"unexpected refusals: {refused}"
    ref = clean * drive
    lost = np.abs(ref) >= 1.0
    err_before = np.abs(hot - ref)[lost].mean()
    err_after = np.abs(fixed - ref)[lost].mean()
    assert err_after < err_before, f"declip made it worse: {err_after} vs {err_before}"
    assert np.abs(fixed).max() > 1.0, "peaks should exceed the old ceiling"

    # the MAX_RUN guard must refuse gaps too long to honestly interpolate
    _, _, refused_hot = declip(np.clip(clean * 3.0, -1.0, 1.0), sr)
    assert refused_hot > 0, "guard failed to refuse over-long clipped runs"

    print(f"self-check OK: {n} runs, peak error {err_before:.4f} -> {err_after:.4f} "
          f"({100*(1-err_after/err_before):.0f}% closer to truth); "
          f"guard refused {refused_hot} over-long runs")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--src")
    p.add_argument("--dst")
    p.add_argument("--peak", type=float, default=0.7)
    p.add_argument("--self-check", action="store_true")
    a = p.parse_args()
    if a.self_check:
        self_check()
    else:
        process(a.src, a.dst, a.peak)
