r"""Desktop simulation of the DECK001 "stem loop" workflow:

    the DJ finds a section on headphones, sets a loop (bars on the beat grid), picks "stemsplit drums"
    and saves it on a hot cue. On the deck that is: separate the loop plus some context with the
    device-style chunked Needle, cut the stem on the beat grid, crossfade the seam, store it.

    .venv\Scripts\python.exe stem_loop.py <track> --start 92.5 --bars 4 [--bpm 124] \
        --checkpoint runs\b20_jam_plain\best.pt --checkpoint runs\b20_jam_frontier\best.pt \
        [--shifts 1 4] [--stem drums] [--teacher <teacher folder with drums.flac>] [--out F:\needle\stem_loops]

--start is where the DJ pressed loop-in; it snaps to the nearest beat. Without --bpm the tempo is
estimated from the audio (house range). Writes, per checkpoint and shift count:
    <model>_s<shifts>_<stem>.wav       the saved hot-cue sample (one loop)
    <model>_s<shifts>_<stem>_x4.wav    the same, played 4 times, to hear the seam
    <model>_s<shifts>_rest_x4.wav      everything else (mix minus the stem): what was taken out
plus mix_x4.wav (the loop of the full mix), teacher_<stem>_x4.wav if --teacher is given, and
hotcue.json (what the deck would store). Runs on the CPU by default, so a training run on the GPU
is not disturbed.
"""
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from chunked import separate_chunked, torch_predictor
from needle import SOURCES, Needle, NeedleConfig

SR = 44100


def read_section(path, start, length):
    """(2, length) float32 from `start` (samples, may be < 0 -> zeros), resampled to 44.1 kHz if needed."""
    sr = sf.info(str(path)).samplerate
    f = sr / SR                                                          # read in the file's own rate
    a, b = max(0, int(start * f)), max(0, int((start + length) * f) + 16)
    x, _ = sf.read(str(path), start=a, stop=b, dtype="float32", always_2d=True)
    x = x.T
    if x.shape[0] == 1:
        x = np.repeat(x, 2, 0)
    x = x[:2]
    if sr != SR:
        from scipy.signal import resample_poly
        g = math.gcd(SR, sr)
        x = resample_poly(x, SR // g, sr // g, axis=1).astype(np.float32)
    lead = max(0, -start)                                                # before the track start: silence
    out = np.zeros((2, length), np.float32)
    m = min(length - lead, x.shape[1])
    if m > 0:
        out[:, lead:lead + m] = x[:, :m]
    return out


def onset_envelope(x, hop=512, n_fft=2048):
    """Spectral flux of the mono mix, one value per hop."""
    m = x.mean(0)
    w = np.hanning(n_fft)
    frames = np.lib.stride_tricks.sliding_window_view(m, n_fft)[::hop] * w
    mag = np.log1p(np.abs(np.fft.rfft(frames, axis=1)))
    flux = np.maximum(0, np.diff(mag, axis=0)).sum(1)
    flux -= np.convolve(flux, np.ones(16) / 16, "same")                 # remove the slow trend
    return np.maximum(flux, 0)


def estimate_bpm(x, lo=100.0, hi=140.0, hop=512):
    """Tempo by autocorrelation of the onset envelope, searched in the house range."""
    env = onset_envelope(x, hop)
    ac = np.correlate(env, env, "full")[len(env) - 1:]
    fps = SR / hop
    bpms = np.arange(lo, hi, 0.05)
    lags = fps * 60 / bpms
    score = np.interp(lags, np.arange(len(ac)), ac) + 0.5 * np.interp(2 * lags, np.arange(len(ac)), ac)
    return float(bpms[np.argmax(score)])


def snap_to_beat(x, x_start, wanted, bpm, hop=512):
    """Beat phase from the onset envelope of the analysed audio; returns the beat position (samples,
    absolute) nearest to `wanted`."""
    env = onset_envelope(x, hop)
    period = SR * 60 / bpm / hop
    phases = np.linspace(0, period, 64, endpoint=False)
    best = max(phases, key=lambda p: np.interp(np.arange(p, len(env) - 1, period), np.arange(len(env)), env).sum())
    beat0 = x_start + int(round(best * hop + 1024))                     # frame centre -> samples
    period_s = SR * 60 / bpm
    k = round((wanted - beat0) / period_s)
    return int(round(beat0 + k * period_s))


def loop_with_seam(y, pre, n, fade):
    """y: (2, N) with `pre` samples of audio before the loop start. One loop of n samples whose last
    `fade` samples crossfade (equal power) into the audio just before the loop start, so looping it
    plays loop end -> pre-roll -> loop start without a click."""
    loop = y[:, pre:pre + n].copy()
    t = np.linspace(0, np.pi / 2, fade, dtype=np.float32)
    loop[:, -fade:] = loop[:, -fade:] * np.cos(t) + y[:, pre - fade:pre] * np.sin(t)
    return loop


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("track")
    ap.add_argument("--start", type=float, required=True, help="loop-in, seconds (snaps to the beat)")
    ap.add_argument("--bars", type=int, default=4)
    ap.add_argument("--bpm", type=float)
    ap.add_argument("--checkpoint", action="append", required=True)
    ap.add_argument("--shifts", type=int, nargs="+", default=[1, 4])
    ap.add_argument("--stem", default="drums", choices=SOURCES)
    ap.add_argument("--teacher", help="teacher stem folder for this track (drums.flac ...), for comparison")
    ap.add_argument("--out", default=r"F:\needle\stem_loops")
    ap.add_argument("--fade-ms", type=float, default=5.0)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()

    # tempo and beat from 30 s of audio around the loop-in point
    ana_start = int((a.start - 10) * SR)
    ana = read_section(a.track, ana_start, 30 * SR)
    bpm = a.bpm or estimate_bpm(ana)
    loop_in = snap_to_beat(ana, ana_start, int(a.start * SR), bpm)
    n = int(round(a.bars * 4 * 60 / bpm * SR))
    fade = int(a.fade_ms / 1000 * SR)

    tag = f"{Path(a.track).stem[:60]}_{loop_in / SR:.2f}s_{a.bars}bars"
    out = Path(a.out) / tag
    out.mkdir(parents=True, exist_ok=True)
    report = {"track": str(a.track), "bpm": round(bpm, 2), "bpm_estimated": a.bpm is None, "loop_in_s": round(loop_in / SR, 4),
              "bars": a.bars, "samples": n, "seconds": round(n / SR, 3), "stem": a.stem, "fade_ms": a.fade_ms, "models": []}

    torch.set_num_threads(8)
    for ck_path in a.checkpoint:
        ck = torch.load(ck_path, map_location="cpu", weights_only=False)
        cfg = NeedleConfig(**ck["config"])
        model = Needle(cfg)
        model.load_state_dict(ck["model"])
        model.to(a.device).eval()
        predict = torch_predictor(model, batch=16)
        name = Path(ck_path).parent.name
        # context: one chunk of audio before and after the loop, so the loop edges never sit at a
        # chunk border and the seam crossfade has real pre-roll
        pad = cfg.frames * cfg.hop
        mix = read_section(a.track, loop_in - pad, n + 2 * pad)
        step = (cfg.frames - 8) * cfg.hop
        chunks = math.ceil((n + 2 * pad) / step) + 1
        if not out.joinpath("mix_x4.wav").exists():
            sf.write(out / "mix_x4.wav", np.tile(loop_with_seam(mix, pad, n, fade), 4).T, SR, subtype="FLOAT")
        for s in a.shifts:
            t0 = time.time()
            stems = separate_chunked(predict, torch.from_numpy(mix), cfg, 8, batch=16, shifts=s).numpy()
            dt = time.time() - t0
            y = stems[SOURCES.index(a.stem)]
            one = loop_with_seam(y, pad, n, fade)
            rest = loop_with_seam(mix - y, pad, n, fade)
            base = f"{name}_s{s}"
            sf.write(out / f"{base}_{a.stem}.wav", one.T, SR, subtype="PCM_16")
            sf.write(out / f"{base}_{a.stem}_x4.wav", np.tile(one, 4).T, SR, subtype="FLOAT")
            sf.write(out / f"{base}_rest_x4.wav", np.tile(rest, 4).T, SR, subtype="FLOAT")
            report["models"].append({"model": name, "step": ck.get("step"), "shifts": s, "chunks": chunks * s,
                                     "cpu_seconds_desktop": round(dt, 2), "file": f"{base}_{a.stem}.wav"})
            print(f"{base}: {chunks * s} chunks, {dt:.2f} s on the desktop CPU", flush=True)
    if a.teacher:
        st = Path(a.teacher) / f"{a.stem}.flac"
        y = read_section(st, loop_in - pad, n + 2 * pad)
        sf.write(out / f"teacher_{a.stem}_x4.wav", np.tile(loop_with_seam(y, pad, n, fade), 4).T, SR, subtype="FLOAT")
    (out / "hotcue.json").write_text(json.dumps(report, indent=1))
    print(out, f"bpm {bpm:.2f}{' (estimated)' if a.bpm is None else ''}, loop-in {loop_in / SR:.3f} s, {n / SR:.2f} s")


if __name__ == "__main__":
    main()
