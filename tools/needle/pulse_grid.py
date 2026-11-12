r"""Pulse -> a DJ beat grid: one straight tempo + phase line through the beat activation, then the bar line on it.

Why: picking beats one by one loses beats in breakdowns, and every lost beat shifts the bar count. House and
techno hold one tempo, so the deck's grid is a straight line anyway (that is what sync needs).

  1. tempo hint   median interval between the activation's peaks (no tempo range cap; the network decides)
  2. fine search  +-4 % around the hint in 0.005-BPM steps: the tempo whose grid lines up with the beat
                  activation best (activation-weighted phasor), then a least-squares line through the peaks
                  within 35 ms of it -> period and phase to well under 1 ms
  3. check        if fewer than 60 % of the peaks sit on that line the track is not steady-tempo: fall back to
                  the peaks (and say so)
  4. bar line     downbeat activation read at every grid beat; the bar position (0-3) chosen per beat by a
                  sliding 16-bar comb (a 32-bar window; tracks with a 2-beat fill still recover)

Runs on the M85 in a few ms per track (the network's output is 50 frames/s).

    python pulse_grid.py [--model p2_w24]  -> Raveform test set: peaks vs grid, beat / downbeat / bar line
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from pulse import FPS, Pulse, f_measure, peaks, predict_track      # noqa: E402
from pulse_rave import Rave, bar_fit                                # noqa: E402
from pulse_raveform import tempo_ok                                 # noqa: E402


def fit_grid(beat_logit, n_frames):
    """-> (grid beat times s, bpm, steady?)."""
    act = torch.sigmoid(beat_logit).numpy()
    pk = peaks(beat_logit)
    if len(pk) < 16:
        return pk, 0.0, False
    hint = 60 / np.median(np.diff(pk))
    cand = np.arange(hint * 0.96, hint * 1.04, 0.005)
    w = act[np.round(pk * FPS).astype(int)] ** 2                    # confident beats count more
    z = (w[None, :] * np.exp(2j * np.pi * pk[None, :] * cand[:, None] / 60)).sum(1)
    i = int(np.argmax(np.abs(z)))
    period = 60 / cand[i]
    phase = (np.angle(z[i]) / (2 * np.pi) * period) % period
    for _ in range(3):                                              # refine on the peaks near the grid
        k = np.round((pk - phase) / period)
        keep = np.abs(pk - (phase + k * period)) < 0.035
        if keep.sum() < 16:
            break
        period, phase = np.polyfit(k[keep], pk[keep], 1)
    k = np.round((pk - phase) / period)
    on = np.abs(pk - (phase + k * period)) < 0.035
    if on.mean() < 0.6:
        return pk, 60 / period, False
    first = phase - np.floor(phase / period) * period
    grid = np.arange(first, n_frames / FPS, period)
    # the grid spans only where the track has beats (not silence before / after)
    return grid[(grid >= pk[0] - period / 2) & (grid <= pk[-1] + period / 2)], 60 / period, True


def grid_downbeats(grid, down_logit, bars=16):
    """Bar line on a gap-free grid: a sliding comb over the same bar position (see bar_fit)."""
    return bar_fit(grid, down_logit, bars=bars)


def bar_share(ref_d, est_d, tol=0.07):
    if len(ref_d) == 0 or len(est_d) == 0:
        return 0.0
    return float(np.mean([np.min(np.abs(est_d - r)) <= tol for r in ref_d]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="p2_w24", help="a runs_pulse name, or beat_this (the teacher, same grid fit)")
    a = ap.parse_args()
    if a.model == "beat_this":
        from beat_this.inference import Spect2Frames
        s2f = Spect2Frames(checkpoint_path=r"F:\needle\beat_this\final0.ckpt", device="cuda", float16=False)

        def m(x):                                                   # same interface as Pulse on a (1, 1, 128, T) batch
            b, d = s2f(x[0, 0].T)
            return torch.stack([b, d])[None]
        predict = lambda mel: m(torch.from_numpy(mel)[None, None].cuda()).float().cpu()[0]   # noqa: E731
    else:
        run = HERE / "runs_pulse" / a.model
        ck = torch.load(run / ("best.pt" if (run / "best.pt").exists() else "last.pt"), map_location="cpu", weights_only=False)
        net = Pulse(ck["width"])
        net.load_state_dict(ck["model"])
        net = net.cuda().eval()
        predict = lambda mel: predict_track(net, mel, "cuda")                                # noqa: E731
    te = Rave("test")
    rows = {k: [] for k in ("peaks + bar fit", "grid + bar fit (16)", "grid + bar fit (32)", "grid, one bar phase")}
    steady = 0
    for t in te.ids:
        b, d = te.lab[t]
        mel = np.asarray(te.mels[t], dtype=np.float32)
        with torch.inference_mode():
            y = predict(mel)
        pk = peaks(y[0])
        grid, bpm, ok = fit_grid(y[0], mel.shape[1])
        steady += ok
        cands = {"peaks + bar fit": (pk, bar_fit(pk, y[1], bars=16)),
                 "grid + bar fit (16)": (grid, grid_downbeats(grid, y[1], 16)),
                 "grid + bar fit (32)": (grid, grid_downbeats(grid, y[1], 32)),
                 "grid, one bar phase": (grid, grid_downbeats(grid, y[1], 10000))}
        for k, (eb, ed) in cands.items():
            rows[k].append((f_measure(b, eb), f_measure(d, ed), tempo_ok(b, eb), bar_share(d, ed)))
    print(f"Raveform test set, {len(te.ids)} tracks, {a.model}; steady tempo (grid used) on {steady}")
    print(f"{'':22s} beat F  downbeat F  tempo right  downbeats on the right beat  tracks >90 % bars right")
    for k, v in rows.items():
        v = np.array(v, dtype=float)
        print(f"{k:22s} {v[:, 0].mean():.3f}   {v[:, 1].mean():.3f}       {v[:, 2].mean() * 100:3.0f}%         "
              f"{v[:, 3].mean():.3f}                       {np.mean(v[:, 3] > 0.9) * 100:.0f}%")


if __name__ == "__main__":
    main()
