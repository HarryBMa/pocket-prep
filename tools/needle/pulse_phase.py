r"""Phase refinement for Pulse's grid (blueprint 2.2): after the grid fit, move the grid onto the drum hits.

Pulse's network marks beats on a 20-ms frame lattice, and it learned where "the beat" is from two kinds of
labels that do not agree: Raveform's hand labels (rekordbox's grid, corrected by hand) and the teacher's
(Beat This!). Measured against the kicks of drum stems, its grid sits about 9 ms late on Jamendo / Ibiza
house and techno and about 5 ms early on Raveform's tracks. The drum hits themselves are in the audio:

  1. three bands of the mono signal at 22050 Hz: below 150 Hz, 150-2000 Hz, above 2 kHz (4th-order
     Butterworth run forwards and backwards, so nothing is delayed), each as power in blocks of 10 samples
     (2205 per second)
  2. per band: power smoothed (11.8 ms below 150 Hz, as beatgrid_lab.kick_onsets does; 5.9 ms above),
     log10, its rise over 4.5 ms; the onset function o is the three rises added (BAND = "all"), or the
     low band's alone ("low", what blueprint 2.2 asks for)
  3. around every grid beat b: the highest point of o within +-40 ms, its time t to a fraction of a block
     (parabola through three points), its height h
  4. the beats whose h is at least TAU x the track's strong onsets (the 90th percentile of h) count, with
     weight h; offset = their weighted median of t - b; spread = their weighted median of |t - b - offset|
  5. grid phase += offset + SHIFT, unless the track is faster than MAX_BPM or the spread is over MAX_SPREAD

Why "all" and not the low band alone: measured on 100 Raveform training tracks and 48 tracks with a drum
stem (13 Nov 2026), the low band's onset agrees with the stems' kicks (same band) but not with the hand
labels: against those the grid ends up worse than before (within 5 ms: 69 % -> 41 %). In EDM the sub of a
kick, or a side-chained bass coming back, often rises 5-20 ms after the click that an editor lines the grid
up with. The bands above 150 Hz carry the click; all three together agree with both references.
SHIFT is the convention: with it the refined grids sit, in the median, where Raveform's hand labels do.
MAX_BPM: on drum & bass the strongest onset near a beat is often a grace note before it; left alone.

The firmware's C++ (pulse_c\pulse_beat.cpp, Options::refinePhase, off by default) is a port of refine() with
BAND = "all"; `fixture` and `check` hold the two against each other.

Commands (CPU only):
  prep     per track of a set: the bands, the kick onsets (beatgrid_lab.kick_onsets), the reference beats and
           the grid of a C++ Pulse build (pulse_cli.exe from pulse_passfail.py); --refined adds that build's
           grid with the phase switch on  -> F:\needle\pulse_phase\<set>\<id>.npz
  eval     before / after on a prepared set, against the kick onsets and against the reference beats (hand
           labels, or the line through a drum stem's kicks); --cpp: the C++ refined grid as "after"
  check    C++ against refine() on real tracks: the same 22.05 kHz samples through both, and the audio at
           its own rate through the C++ resampler against refine() on the prepared bands
  fixture  the table of expected values in pulse_c\test_pulse_beat.cpp (refine() on its test signal)

    .venv\Scripts\python.exe pulse_phase.py prep --set rave_test --model p4_w24_l6 --bands
    .venv\Scripts\python.exe pulse_phase.py eval --set rave_test --model p4_w24_l6 [--band low]
    .venv\Scripts\python.exe pulse_phase.py check --set rave_test --model p4_w24_l6 --tracks 40
"""
import argparse
import json
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import scipy.signal as ss

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

SR = 22050                  # Pulse's own rate
DECIM = 10
FS = SR // DECIM            # 2205 blocks per second
CUTOFF = 150.0              # Hz, the low band's upper edge
WINDOW = 0.040              # s to either side of a grid beat
TAU = 0.3                   # a beat counts when its onset is at least this share of the track's strong ones
SMOOTH = {"low": 26, "mid": 13, "high": 13}     # blocks the power is averaged over
LAG = 10                    # blocks the rise is taken over (4.5 ms)
BAND = "all"                # "all" (the three bands' rises added) or "low" (blueprint 2.2)
SHIFT = {"all": 0.0023, "low": -0.0020}         # s, per band choice: refined grids -> the hand labels' convention
                                                # (`convention` on rave_cal, 98 Raveform training tracks, 13 Nov 2026)
MAX_BPM = 150.0             # faster tracks are left as the network put them
MAX_SPREAD = 0.006          # s: a track whose onsets scatter more than this around the offset is left alone


def low_band(x22):
    """Mono signal at 22050 Hz -> its band below 150 Hz, every 10th sample (2205 Hz). The filter runs forwards,
    then backwards (zero phase: nothing is delayed), each time from rest: plain enough to be the same in C++."""
    sos = ss.butter(4, CUTOFF, fs=SR, output="sos")
    y = ss.sosfilt(sos, np.asarray(x22, dtype=np.float64))
    return ss.sosfilt(sos, y[::-1])[::-1][::DECIM]


def band_envelopes(x22):
    """Mono signal at 22050 Hz -> RMS envelopes at 2205 Hz of its 150-2000 Hz band and of everything above 2 kHz
    (zero-phase filters; each value is the RMS of 10 samples), as float16 relative to the track's own maximum."""
    out = []
    x = np.asarray(x22, dtype=np.float64)
    for sos in (ss.butter(4, [CUTOFF, 2000.0], "bandpass", fs=SR, output="sos"), ss.butter(4, 2000.0, "highpass", fs=SR, output="sos")):
        y = ss.sosfilt(sos, ss.sosfilt(sos, x)[::-1])[::-1]
        n = len(y) // DECIM
        e = np.sqrt((y[: n * DECIM] ** 2).reshape(n, DECIM).mean(1))
        out.append((e / (e.max() or 1.0)).astype(np.float16))
    return out


def rise(power, smooth):
    """How much a band's smoothed power (log10) goes up over LAG blocks, never below 0. Element n is the rise
    from block n to block n + LAG."""
    p = np.convolve(power, np.ones(smooth) / smooth, "same")
    e = np.log10(p + 1e-4 * np.percentile(p, 99) + 1e-20)               # floor: 40 dB under the loud passages
    return np.maximum(0.0, e[LAG:] - e[:-LAG])


def onset_function(low, env_mid=None, env_high=None, band=BAND):
    """-> (o, centre): o[n] is the onset strength at the time (n + centre) / FS."""
    low = np.asarray(low, dtype=np.float64)
    o = rise(low * low, SMOOTH["low"])
    if band == "low":
        return o, LAG / 2
    if env_mid is None or env_high is None:
        raise ValueError('band "all" needs the envelopes above 150 Hz (prep --bands)')
    n = min(len(o), len(env_mid) - LAG, len(env_high) - LAG)
    for name, env in (("mid", env_mid), ("high", env_high)):
        e = np.asarray(env, dtype=np.float64)
        o = o[:n] + rise(e * e, SMOOTH[name])[:n]
    return o, LAG / 2 + 0.3                                             # an envelope block is centred 0.45 block later than a low-band sample


class Refined:
    """delta: s to add to the grid's first beat (0.0 when the track is left alone); raw: the same before the
    guards; spread: s, weighted median of |onset - beat - offset|; used / beats: how many beats counted, of how
    many; drift: s per s, slope of onset - beat over the track (a tempo that is slightly off shows here)."""
    def __init__(self, delta, raw, spread, used, beats, drift):
        self.delta, self.raw, self.spread, self.used, self.beats, self.drift = delta, raw, spread, used, beats, drift


def weighted_median(v, w):
    i = np.argsort(v)
    c = np.cumsum(w[i])
    return float(v[i][np.searchsorted(c, c[-1] / 2)])


def beat_onsets(o, centre, bpm, first, window=WINDOW):
    """Per grid beat: (beat time, time of the strongest onset within +-window, its height). Beats whose
    window leaves the track, or whose highest point is at the window's edge (a rise that peaks outside), are
    left out."""
    period = 60.0 / bpm
    half = int(round(window * FS))
    k0 = int(np.ceil((window + 2 / FS - first) / period))
    k1 = int(np.floor(((len(o) - 2) / FS - window - first) / period))
    b = first + np.arange(k0, k1 + 1) * period
    c = np.round(b * FS - centre).astype(int)                           # o's index nearest to the beat
    idx = c[:, None] + np.arange(-half, half + 1)[None, :]
    inside = (idx[:, 0] >= 0) & (idx[:, -1] < len(o))
    b, idx = b[inside], idx[inside]
    seg = o[idx]
    j = seg.argmax(1)
    ok = (j > 0) & (j < seg.shape[1] - 1)
    b, idx, seg, j = b[ok], idx[ok], seg[ok], j[ok]
    r = np.arange(len(j))
    y0, y1, y2 = seg[r, j - 1], seg[r, j], seg[r, j + 1]
    den = y0 - 2 * y1 + y2
    frac = np.where(den < 0, 0.5 * (y0 - y2) / np.where(den < 0, den, -1.0), 0.0)
    t = (idx[r, j] + centre + frac) / FS
    return b, t, y1


def refine(low, bpm, first, env_mid=None, env_high=None, band=BAND, tau=TAU, window=WINDOW, shift=None,
           max_bpm=MAX_BPM, max_spread=MAX_SPREAD):
    """The grid (bpm, first beat in s) against the track's bands (low_band, band_envelopes) -> Refined."""
    shift = SHIFT[band] if shift is None else shift
    o, centre = onset_function(low, env_mid, env_high, band)
    b, t, h = beat_onsets(o, centre, bpm, first, window)
    if len(b) < 8:
        return Refined(0.0, 0.0, 0.0, 0, len(b), 0.0)
    keep = h >= tau * np.percentile(h, 90)
    d, w = (t - b)[keep], h[keep]
    if len(d) < 8:
        return Refined(0.0, 0.0, 0.0, int(keep.sum()), len(b), 0.0)
    offset = weighted_median(d, w)
    spread = weighted_median(np.abs(d - offset), w)
    near = np.abs(d - offset) < max(3 * spread, 0.003)                  # the slope from the beats near the line
    drift = float(np.polyfit(b[keep][near], d[near], 1, w=w[near])[0]) if near.sum() >= 8 else 0.0
    raw = offset + shift
    return Refined(raw if bpm <= max_bpm and spread <= max_spread else 0.0, raw, spread, int(keep.sum()), len(b), drift)


# ── measuring ────────────────────────────────────────────────────────────────

def folded(times, bpm, first):
    """Grid minus event, per event, folded to +-half a beat (s): positive = the grid beat comes late."""
    p = 60.0 / bpm
    return -(((np.asarray(times) - first + p / 2) % p) - p / 2)


def track_offset(times, bpm, first, reach=0.070, near=0.020):
    """One track's signed offset (s) of the grid against a set of events (kick onsets or human beats): of the
    events within 70 ms of a grid beat, the median of grid - event over those within 20 ms of where most of
    them sit. On a full mix kick_onsets also finds bass notes between the beats (often more of them than
    kicks: a kick on top of a held bass note does not rise by 5 dB); those are further than 70 ms from a beat
    of a grid at the right tempo. None with fewer than 16 events there."""
    d = folded(times, bpm, first)
    d = d[np.abs(d) <= reach]
    if len(d) < 16:
        return None
    hist, edges = np.histogram(d, bins=np.arange(-reach, reach + 0.002, 0.002))
    hist = np.convolve(hist, np.ones(5), "same")                        # 10 ms
    mode = edges[hist.argmax()] + 0.001
    sel = np.abs(d - mode) < near
    return float(np.median(d[sel])) if sel.sum() >= 16 else None


def table(rows, key):
    """rows of per-track offsets (s) -> the line asked for: median signed, spread, shares within 3 / 5 / 10 ms."""
    v = np.array([r[key] for r in rows if r.get(key) is not None]) * 1000
    if len(v) == 0:
        return "no tracks"
    q1, q3 = np.percentile(v, [25, 75])
    return (f"n={len(v):3d}  median {np.median(v):+6.1f} ms  quartiles {q1:+6.1f} .. {q3:+6.1f} (IQR {q3 - q1:5.1f})  "
            f"|offset| median {np.median(np.abs(v)):5.1f}  within 3 / 5 / 10 ms: {np.mean(np.abs(v) <= 3):4.0%} / "
            f"{np.mean(np.abs(v) <= 5):4.0%} / {np.mean(np.abs(v) <= 10):4.0%}")


def offsets(z, bpm, first, delta, raw):
    """Per reference in a prepared track: the grid's offset before, after (moved by delta) and after without
    the guards (moved by raw)."""
    out = {}
    for name, key in (("kick", "kicks"), ("ref", "beats"), ("stem", "kicks_stem")):
        if key in z:
            out[f"{name}_before"] = track_offset(z[key], bpm, first)
            out[f"{name}_after"] = track_offset(z[key], bpm, first + delta)
            out[f"{name}_raw"] = track_offset(z[key], bpm, first + raw)
    return out


def measure(z, grid, band=BAND, **kw):
    """One prepared track (npz dict) and its grid (bpm, first) -> the refinement's numbers and, per reference
    in the file, the grid's offset before and after."""
    bpm, first = grid
    r = refine(z["low"].astype(np.float64), bpm, first, z.get("env_mid"), z.get("env_high"), band, **kw)
    out = {"delta": r.delta, "raw": r.raw, "spread": r.spread, "used": r.used, "beats": r.beats, "drift": r.drift}
    out.update(offsets(z, bpm, first, r.delta, r.raw))
    return out


def measure_cpp(z, g):
    """measure() with the firmware's own numbers in refine()'s place. g: pulse_cli's line with the phase switch
    on (..., phaseShift, phaseOffset, phaseSpread, phaseBeats); its firstBeat is the grid after the shift."""
    bpm, shift, raw = float(g[2]), float(g[14]), float(g[15])
    out = {"delta": shift, "raw": raw, "spread": float(g[16]), "used": int(g[17])}
    out.update(offsets(z, bpm, float(g[3]) - shift, shift, raw))
    return out


# ── prep (on the PC: needs the audio, beatgrid_lab and a C++ build) ──────────────

def set_tracks(name):
    """-> [(id, audio file, beats, downbeats, drum stem or None)]"""
    from pulse import DATA
    rave = DATA / "beat_rave"
    index = json.loads((rave / "index.json").read_text())
    test = set((rave / "test.txt").read_text().split())
    if name == "rave_test":
        ids = sorted(t for t in index if t in test)
    elif name == "rave_cal":                                            # 100 training tracks: for settings, not for scores
        ids = sorted(t for t in index if t not in test)
        random.Random(11).shuffle(ids)
        ids = sorted(ids[:100])
    elif name == "mixed":                                               # pulse_passfail.py's 459, the teacher's beats as "beats"
        teacher = DATA / "beat_teacher"
        tindex = json.loads((teacher / "index.json").read_text(encoding="utf-8"))
        out = []
        for v in json.loads((DATA / "pulse_passfail" / "mixed459_ids.json").read_text(encoding="utf-8")):
            z = np.load(teacher / f"{v['id']}.npz")
            out.append((v["key"], tindex[v["id"]]["source"], z["beats"].astype(np.float64), z["downbeats"].astype(np.float64), None))
        return out
    elif name == "lab":
        # beatgrid_lab.py's tracks with a precise reference: the kicks of the teacher's drum stem lie on one
        # straight line (within 3 ms, 64 or more of them). "beats" = that line; kicks_stem = the stem's onsets.
        from beatgrid_lab import SETS
        res = json.loads((DATA / "beatgrid" / "results.json").read_text(encoding="utf-8"))
        out = []
        for key, r in sorted(res.items()):
            ref = r.get("ref")
            if not ref or ref["jitter_ms"] >= 3 or ref["n"] < 64:
                continue
            set_name, track = key.split("/", 1)
            line = ref["phase"] + np.arange(0, int(1800 * ref["bpm"] / 60)) * 60 / ref["bpm"]
            out.append((f"{set_name}__{track}", r["source"], line, np.zeros(0), str(SETS[set_name] / track / "drums.flac")))
        return out
    elif name == "stems":
        # 300 tracks of Needle's teacher pass (E:\needle-teacher\jamendo_v2: a drum stem per track). The reference
        # line is fitted in prep (beatgrid_lab.reference_grid on the stem's kick onsets, Pulse's tempo as the hint)
        # and kept with its quality ("ref": bpm, phase, onsets on the line, onsets, F1, jitter ms).
        root = Path(r"E:\needle-teacher\jamendo_v2")
        dirs = sorted(p.parent for p in root.glob("*/done") if (p.parent / "drums.flac").exists() and (p.parent / "source.txt").exists())
        random.Random(5).shuffle(dirs)
        return [(d.name, (d / "source.txt").read_text(encoding="utf-8").splitlines()[0], None, np.zeros(0), str(d / "drums.flac"))
                for d in sorted(dirs[:300])]
    else:
        raise SystemExit(f"unknown set {name}")
    out = []
    for t in ids:
        z = np.load(rave / f"{t}.npz")
        out.append((t, index[t]["file"], z["beats"].astype(np.float64), z["downbeats"].astype(np.float64), None))
    return out


def run_cli(cli, pcm_file, rate, threads, rule, refine_phase=False):
    """pulse_cli.exe on a file of interleaved stereo float32 -> its line as floats (the columns it prints)."""
    cmd = [str(cli), str(pcm_file), str(rate), str(threads), "0", str(int(rule))] + (["1"] if refine_phase else [])
    v = [float(s) for s in subprocess.run(cmd, capture_output=True, text=True, timeout=900).stdout.split()]
    if len(v) < (18 if refine_phase else 14):
        raise SystemExit(f"{cli} printed {len(v)} values: " + ("a build from before the phase switch; rebuild it "
                         "(pulse_passfail.py --model M --no-export --export-only)" if len(v) == 14 else "it failed"))
    return v


def prep(a):
    import soundfile as sf
    import soxr
    from beatgrid_lab import kick_onsets, reference_grid
    from pulse import DATA
    out = DATA / "pulse_phase" / a.set
    out.mkdir(parents=True, exist_ok=True)
    cli = DATA / "pulse_passfail" / a.model / "pulse_cli.exe"
    tmp, t0 = out / f"_audio_{a.model}.f32", time.time()
    tracks = set_tracks(a.set)
    for n, (t, file, beats, downs, stem) in enumerate(tracks):
        path = out / f"{t}.npz"
        old = dict(np.load(path)) if path.exists() else {}
        if f"grid_{a.model}" in old and ("env_high" in old or not a.bands) and (f"refined_{a.model}" in old or not a.refined):
            continue
        try:
            x, sr = sf.read(file, dtype="float32", always_2d=True)
        except Exception as ex:                                         # noqa: BLE001
            print("no audio:", t, ex, flush=True)
            continue
        x = x[:, :2] if x.shape[1] >= 2 else np.repeat(x, 2, 1)
        if f"grid_{a.model}" not in old:
            np.ascontiguousarray(x).tofile(tmp)
            g = []
            for rule in (0, 1):                                         # the double-time rule off, on
                g.append(run_cli(cli, tmp, sr, a.threads, rule)[:14])
            old[f"grid_{a.model}"] = np.array(g)                        # rows: rule off / on; columns as pulse_cli prints
        g = old[f"grid_{a.model}"]
        if a.refined and f"refined_{a.model}" not in old:               # the rule on and the phase switch on
            np.ascontiguousarray(x).tofile(tmp)
            old[f"refined_{a.model}"] = np.array(run_cli(cli, tmp, sr, a.threads, 1, refine_phase=True))
        if a.bands and "env_high" not in old:                           # the other bands, for comparing with the low one
            mono = x.mean(1)
            x22 = soxr.resample(mono, sr, SR) if sr != SR else mono
            old["env_mid"], old["env_high"] = band_envelopes(x22)
        if "low" not in old:
            mono = x.mean(1)
            x22 = soxr.resample(mono, sr, SR) if sr != SR else mono
            low = low_band(x22)
            scale = float(np.abs(low).max()) or 1.0
            old.update(low=(low / scale).astype(np.float16), scale=np.array(scale), kicks=kick_onsets(x, sr), downbeats=downs,
                       seconds=np.array(len(x) / sr))
            if stem:
                d, dsr = sf.read(stem, dtype="float32", always_2d=True)
                old["kicks_stem"] = kick_onsets(d, dsr)
            if beats is None:                                           # the reference: a line through the stem's kicks
                ref = reference_grid(old["kicks_stem"], g[1][2] if g[1][0] and g[1][2] > 0 else 124.0)
                old["ref"] = np.array([ref[k] for k in ("bpm", "phase", "n", "of", "f1", "jitter_ms")] if ref else [0.0] * 6)
                beats = ref["phase"] + np.arange(int(1800 * ref["bpm"] / 60)) * 60 / ref["bpm"] if ref else np.zeros(0)
            old["beats"] = beats[beats < len(x) / sr]
        np.savez_compressed(path, **old)
        if n % 10 == 0:
            print(f"[{n + 1}/{len(tracks)}] {t}  {time.time() - t0:.0f} s", flush=True)
    tmp.unlink(missing_ok=True)
    print("PREP DONE", a.set, a.model, flush=True)


def load_set(folder, model):
    out = []
    for p in sorted(Path(folder).glob("*.npz")):
        z = dict(np.load(p))
        if f"grid_{model}" in z:
            out.append((p.stem, z))
    return out


def human_line(b):
    k = np.arange(len(b))
    period, phase = np.polyfit(k, b, 1)
    return 60 / period, phase, float(np.std(b - (phase + k * period)) * 1000)


REFS = {"rave_test": "Raveform's hand labels", "rave_cal": "Raveform's hand labels", "mixed": "the teacher's beats (20-ms steps)",
        "lab": "the line through the drum stem's kicks", "stems": "the line through the drum stem's kicks"}


def track_rows(folder, model, rule=1, band=BAND, cpp=False, **kw):
    """Per track of a prepared set: is the reference a steady grid, is Pulse's tempo right, and if so measure()
    (cpp: measure_cpp(), on the tracks that prep --refined has been through)."""
    rows = []
    for t, z in load_set(folder, model):
        if len(z["beats"]) < 16 or (cpp and f"refined_{model}" not in z):
            continue
        g = z[f"refined_{model}"] if cpp else z[f"grid_{model}"][1 if rule else 0]
        ok, steady, bpm, first = g[0], g[1], g[2], g[3]
        rb, _, jit = human_line(z["beats"])
        row = {"id": t, "ref_bpm": rb, "bpm": float(bpm), "ref_steady": bool(jit < 15), "tempo_right": bool(ok and steady and abs(bpm / rb - 1) < 0.02),
               "on": float(g[5]), "act": float(g[11]), "margin": float(g[12]), "parity": float(g[9]), "halved": bool(g[10])}
        if "ref" in z:                                                  # stems: how good the line through the kicks is
            row["ref_steady"] = bool(z["ref"][5] < 3 and z["ref"][2] >= 64)
        if row["tempo_right"]:
            row.update(measure_cpp(z, g) if cpp else measure(z, (bpm, first), band, **kw))
        if cpp:                                                         # the same grid as the stored one, before the shift?
            s = z[f"grid_{model}"][1]
            bar = 240.0 / bpm if bpm > 0 else 1.0
            row["same_grid"] = bool(s[0] == ok and s[1] == steady and abs(s[2] - bpm) < 1e-5
                                    and abs((first - g[14] - s[3] + bar / 2) % bar - bar / 2) < 2e-6)
        rows.append(row)
    return rows


def set_folder(a):
    if a.dir:
        return Path(a.dir)
    from pulse import DATA                                              # needs torch; --dir does not
    return DATA / "pulse_phase" / a.set


def evaluate(a):
    folder = set_folder(a)
    if a.cpp and (a.band != "all" or not a.rule):
        raise SystemExit('--cpp is the firmware\'s refinement: band "all", the double-time rule on')
    rows = track_rows(folder, a.model, a.rule, a.band, a.cpp)
    good = [r for r in rows if r["ref_steady"] and r["tempo_right"]]
    moved = sum(r["delta"] != 0 for r in good)
    how = "C++ (pulse_cli, phase switch on)" if a.cpp else f"refine() here, band {a.band}"
    if a.cpp:
        print(f"the C++ grid before the shift is the stored one on {sum(r['same_grid'] for r in rows)} of {len(rows)} tracks")
    print(f"{a.set}, {a.model}, double-time rule {'on' if a.rule else 'off'}, {how}: {len(rows)} tracks, {len(good)} with a steady "
          f"reference and the right tempo; grid moved on {moved}, left alone on {len(good) - moved} "
          f"(over {MAX_BPM:g} BPM: {sum(r['bpm'] > MAX_BPM for r in good)}, spread over {MAX_SPREAD * 1000:g} ms: "
          f"{sum(r['spread'] > MAX_SPREAD and r['bpm'] <= MAX_BPM for r in good)})")
    for ref, name in (("kick", "kick onsets (beatgrid_lab.kick_onsets on the mix)"), ("ref", REFS.get(a.set, "the reference beats")),
                      ("stem", "kick onsets of the drum stem")):
        if any(f"{ref}_before" in r for r in good):
            print(f"  against {name}")
            print("    before           " + table(good, f"{ref}_before"))
            print("    after            " + table(good, f"{ref}_after"))
            print("    after, no guards " + table(good, f"{ref}_raw"))
    d = np.array([r["raw"] for r in good]) * 1000
    sp = np.array([r["spread"] for r in good]) * 1000
    line = (f"  the measured offset (with SHIFT {SHIFT[a.band] * 1000:+.1f} ms): median {np.median(d):+.1f} ms, quartiles {np.percentile(d, 25):+.1f} .. "
            f"{np.percentile(d, 75):+.1f}; spread median {np.median(sp):.1f} ms (90th percentile {np.percentile(sp, 90):.1f})")
    if not a.cpp:
        u = np.array([r["used"] / max(1, r["beats"]) for r in good])
        dr = np.array([abs(r["drift"]) for r in good]) * 60 * 1000
        line += (f"; beats that counted: median {np.median(u):.0%}; drift median {np.median(dr):.1f} ms per minute "
                 f"(90th percentile {np.percentile(dr, 90):.1f})")
    print(line)
    name = f"{a.set}_{a.model}_{'on' if a.rule else 'off'}_{'cpp' if a.cpp else a.band}.json"
    (folder.parent / name).write_text(json.dumps(rows, indent=1))


def convention(a):
    """SHIFT per band choice: minus the median, over the calibration tracks, of where the refined grid would sit
    against the hand labels without it. It belongs to the onset function, not to the model that made the grid."""
    folder = set_folder(a)
    for band in ("all", "low"):
        rows = [r for r in track_rows(folder, a.model, a.rule, band, shift=0.0) if r["ref_steady"] and r["tempo_right"] and r.get("ref_raw") is not None]
        v = np.array([r["ref_raw"] for r in rows])
        print(f'band "{band}": {len(v)} tracks of {a.set}; the unshifted refined grid sits {np.median(v) * 1000:+.2f} ms from the hand labels '
              f'(median) -> SHIFT["{band}"] = {-np.median(v):.4f}')


def check(a):
    """The C++ port against refine(), on tracks of a prepared set (drawn with a fixed seed):
      same samples    this file's 22.05 kHz signal (soxr) through pulse_cli and through refine(): what the port
                      itself differs by (the float16 envelopes here, float32 filter outputs there)
      the deck's way  the audio at its own rate through pulse_cli (its own resampler) against refine() on the
                      prepared bands: what a track differs by on the deck
    Both use the grid pulse_cli found for that input (before its shift), so only the phase stage is compared."""
    import soundfile as sf
    import soxr
    from pulse import DATA
    folder = DATA / "pulse_phase" / a.set
    cli = DATA / "pulse_passfail" / a.model / "pulse_cli.exe"
    files = {t: file for t, file, *_ in set_tracks(a.set)}
    ready = [(t, z) for t, z in load_set(folder, a.model) if "env_high" in z and t in files]
    random.Random(3).shuffle(ready)
    tmp, rows, t0 = folder / f"_check_{a.model}.f32", {"same samples": [], "the deck's way": []}, time.time()
    for n, (t, z) in enumerate(ready[: a.tracks]):
        x, sr = sf.read(files[t], dtype="float32", always_2d=True)
        x = x[:, :2] if x.shape[1] >= 2 else np.repeat(x, 2, 1)
        mono = x.mean(1)
        x22 = soxr.resample(mono, sr, SR) if sr != SR else mono
        for name, pcm, rate in (("same samples", np.repeat(x22[:, None], 2, 1), SR), ("the deck's way", x, sr)):
            np.ascontiguousarray(pcm, dtype=np.float32).tofile(tmp)
            v = run_cli(cli, tmp, rate, a.threads, 1, refine_phase=True)
            if not (v[0] and v[1]):
                continue
            if name == "same samples":
                r = refine(low_band(x22), v[2], v[3] - v[14], *band_envelopes(x22))
            else:
                r = refine(z["low"].astype(np.float64), v[2], v[3] - v[14], z["env_mid"], z["env_high"])
            rows[name].append({"id": t, "bpm": v[2], "offset": v[15] - r.raw, "spread": v[16] - r.spread, "beats": int(v[17]) - r.used,
                               "shift": v[14] - r.delta, "moved_cpp": v[14] != 0, "moved_py": r.delta != 0})
        if n % 5 == 0:
            print(f"[{n + 1}/{min(a.tracks, len(ready))}] {t}  {time.time() - t0:.0f} s", flush=True)
    tmp.unlink(missing_ok=True)
    print(f"{a.set}, {a.model}: C++ (pulse_cli) minus refine(), per track")
    for name, rs in rows.items():
        if not rs:
            continue
        off = np.abs([r["offset"] for r in rs]) * 1e6
        sp = np.abs([r["spread"] for r in rs]) * 1e6
        flips = [r for r in rs if r["moved_cpp"] != r["moved_py"]]
        print(f"  {name:14s} {len(rs):3d} tracks: |offset| median {np.median(off):.2f} us, 90th percentile {np.percentile(off, 90):.2f}, max {off.max():.2f}; "
              f"|spread| median {np.median(sp):.2f} us, max {sp.max():.2f}; beat counts differ on {sum(r['beats'] != 0 for r in rs)}; "
              f"moved by one and left alone by the other: {len(flips)}" + "".join(f"\n      {r['id']} {r['bpm']:.2f} BPM" for r in flips))
    (folder.parent / f"check_{a.set}_{a.model}.json").write_text(json.dumps(rows, indent=1))
    print("CHECK DONE", flush=True)


FIXTURE_GRIDS = ((128.0, 0.0), (128.0, 0.011), (128.0, 0.45505), (128.0, 0.238375), (64.0, 0.006), (127.9, 0.002))


def fixture(_):
    """The rows of kPhaseFixture in pulse_c\\test_pulse_beat.cpp: grid tempo, a grid beat, and refine()'s offset
    (before the guards), spread and beat count on the test signal that test builds from the same formula."""
    from pulse_cexport import test_signal
    x = test_signal()
    low, (mid, high) = low_band(x), band_envelopes(x)
    for bpm, first in FIXTURE_GRIDS:
        r = refine(low, bpm, first, mid, high)
        print(f"    {{{bpm!r}, {first!r}, {r.raw!r}, {r.spread!r}, {r.used}}},")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["prep", "eval", "convention", "check", "fixture"])
    ap.add_argument("--set", default="rave_test", help="rave_test (the 200 test tracks), rave_cal (100 training tracks), "
                    "mixed (pulse_passfail.py's 459, the teacher's beats in place of human ones), lab (beatgrid_lab.py's "
                    "tracks with a precise kick line from the teacher's drum stems, that line in place of human beats) or "
                    "stems (300 more tracks with a drum stem, the line fitted here)")
    ap.add_argument("--model", default="p4_w24_l6", help="the C++ build in F:\\needle\\pulse_passfail\\<model>")
    ap.add_argument("--rule", type=int, default=1, help="eval: the grid with the double-time rule on (1) or off (0)")
    ap.add_argument("--band", default=BAND, choices=["all", "low"], help="eval: the three bands (default) or the low band alone")
    ap.add_argument("--bands", action="store_true", help="prep: also keep the envelopes of the bands above 150 Hz")
    ap.add_argument("--refined", action="store_true", help="prep: also keep the C++ grid with the phase switch on")
    ap.add_argument("--cpp", action="store_true", help="eval: the C++ refined grid (prep --refined) in place of refine() here")
    ap.add_argument("--tracks", type=int, default=40, help="check: how many tracks")
    ap.add_argument("--threads", type=int, default=3, help="prep / check: threads pulse_cli may use")
    ap.add_argument("--dir", help="eval / convention: the prepared set's folder, if it is not F:\\needle\\pulse_phase\\<set>")
    a = ap.parse_intermixed_args()
    {"prep": prep, "eval": evaluate, "convention": convention, "check": check, "fixture": fixture}[a.cmd](a)
