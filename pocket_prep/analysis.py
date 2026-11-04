"""Track analysis: tempo and beat grid, key, energy, phrases and waveforms.

Everything a player needs so it doesn't have to analyse the track itself.
Results are saved as JSON per track in the analysis folder; the export step
turns them into rekordbox ANLZ files.

Methods (kept simple on purpose, so they can be improved step by step):
  beat grid  librosa's beat tracker, then a straight-line fit so the grid
             has one constant tempo (like rekordbox's default grid)
  downbeat   the beat phase (1 of 4) where the bass hits hardest
  key        average chroma matched against Krumhansl-Kessler key profiles
  phrases    energy per 8-bar block: where the intro ends, drops, the outro
  waveforms  low/mid/high band levels at 150 columns per second
"""
from __future__ import annotations

import base64
import json
import time

import librosa
import numpy as np
import scipy.signal as sps

from . import config, db

SR = 22050
HOP = 147            # 22050 / 147 = 150 waveform columns per second

PITCHES = ["C", "Db", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]
MINOR_NAMES = ["Cm", "C#m", "Dm", "Ebm", "Em", "Fm", "F#m", "Gm", "G#m", "Am", "Bbm", "Bm"]
CAMELOT_MAJOR = [8, 3, 10, 5, 12, 7, 2, 9, 4, 11, 6, 1]
CAMELOT_MINOR = [5, 12, 7, 2, 9, 4, 11, 6, 1, 8, 3, 10]
KK_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
KK_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])


def _b64(a: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(a).tobytes()).decode()


def load_analysis(track_id: int) -> dict | None:
    p = config.ANALYSIS_DIR / f"{track_id}.json"
    try:
        return json.loads(p.read_text("utf-8"))
    except (OSError, ValueError):
        return None


def waveform_bytes(a: dict, name: str) -> bytes:
    return base64.b64decode(a["waveforms"][name])


# ── Building blocks ──────────────────────────────────────────────────────────

def band_envelopes(y: np.ndarray):
    """Peak level and low/mid/high RMS for every 1/150 s column."""
    lo = sps.sosfilt(sps.butter(4, 200, "low", fs=SR, output="sos"), y)
    hi = sps.sosfilt(sps.butter(4, 2500, "high", fs=SR, output="sos"), y)
    mid = y - lo - hi
    n = len(y) // HOP

    def frames(x):
        return x[: n * HOP].reshape(n, HOP)

    peak = np.abs(frames(y)).max(1)
    rms = [np.sqrt((frames(x) ** 2).mean(1)) for x in (lo, mid, hi)]
    return peak, rms[0], rms[1], rms[2]


def refine_grid(y: np.ndarray, bpm: float) -> tuple[float, float]:
    """Fine-tune tempo and phase: try tempos 0.01 BPM apart around the estimate
    and keep the grid whose beats land on the most onset energy.
    Returns (bpm, time of the first beat in seconds)."""
    hop = 128
    env = librosa.onset.onset_strength(y=y, sr=SR, hop_length=hop)
    fps = SR / hop
    n = len(env)

    def score(b, phases):
        period = 60.0 / b * fps
        k = np.arange(0, max((n - 1 - period) / period, 1))
        idx = (phases[:, None] + k[None, :] * period).astype(int)
        return env[np.clip(idx, 0, n - 1)].sum(1)

    best = (-1.0, bpm, 0.0)
    for b in np.arange(bpm - 1.0, bpm + 1.0001, 0.01):
        phases = np.linspace(0, 60.0 / b * fps, 32, endpoint=False)
        s = score(b, phases)
        j = int(np.argmax(s))
        if s[j] > best[0]:
            best = (float(s[j]), float(b), float(phases[j]))
    _, b, ph = best
    # Most club tracks sit on a whole BPM: prefer it when it scores almost as well.
    whole = float(round(b))
    if abs(whole - b) < 0.3:
        phases = np.arange(0, 60.0 / whole * fps, 60.0 / whole * fps / 64)
        s = score(whole, phases)
        if s.max() >= best[0] * 0.985:
            b, ph = whole, float(phases[int(np.argmax(s))])
    # Phase to the nearest single onset frame.
    period_f = 60.0 / b * fps
    fine = np.arange(ph - period_f / 32, ph + period_f / 32, 1.0)
    ph = float(fine[int(np.argmax(score(b, fine)))]) % period_f
    return round(b, 2), ph / fps


def beat_grid(y: np.ndarray, low_env: np.ndarray, bpm_min: float, bpm_max: float) -> dict:
    onset = librosa.onset.onset_strength(y=y, sr=SR, hop_length=512)
    tempo, frames = librosa.beat.beat_track(onset_envelope=onset, sr=SR, hop_length=512)
    times = librosa.frames_to_time(frames, sr=SR, hop_length=512)
    if len(times) < 8:
        period = 60.0 / 120.0
        offset = 0.0
    else:
        # Fit beat_time = offset + n × period, dropping outliers twice.
        period = float(np.median(np.diff(times)))
        n = np.round((times - times[0]) / period)
        keep = np.ones(len(times), bool)
        for _ in range(3):
            A = np.vstack([np.ones(keep.sum()), n[keep]]).T
            offset, period = np.linalg.lstsq(A, times[keep], rcond=None)[0]
            keep = np.abs(times - (offset + period * n)) < 0.06
            if keep.sum() < 8:
                break
    # Keep the tempo inside the DJ range by halving or doubling.
    bpm = 60.0 / period
    while bpm < bpm_min:
        bpm *= 2
    while bpm > bpm_max:
        bpm /= 2
    bpm, first = refine_grid(y, bpm)
    period = 60.0 / bpm

    # Downbeat: which of the four beat phases has the strongest bass.
    duration = len(y) / SR
    beat_times = np.arange(first, duration, period)
    idx = np.clip((beat_times * 150).astype(int), 0, len(low_env) - 1)
    win = np.array([low_env[max(i - 3, 0): i + 4].max() for i in idx]) if len(idx) else np.zeros(0)
    scores = [win[p::4].sum() for p in range(4)] if len(win) >= 4 else [1, 0, 0, 0]
    phase = int(np.argmax(scores))
    return {"bpm": bpm, "period_ms": period * 1000, "first_beat_ms": first * 1000, "downbeat_index": phase}


def beats_from_grid(a: dict) -> list[tuple[int, int, float]]:
    """Every beat as (beat number 1-4, time in ms, bpm) for the ANLZ PQTZ section."""
    out = []
    t, i = a["first_beat_ms"], 0
    while t < a["duration_ms"]:
        out.append((((i - a["downbeat_index"]) % 4) + 1, int(round(t)), a["bpm"]))
        t += a["period_ms"]
        i += 1
    return out


def detect_key(y: np.ndarray) -> tuple[str, str]:
    chroma = librosa.feature.chroma_cqt(y=y, sr=SR, hop_length=4096)
    prof = chroma.mean(1)
    best, name, cam = -2.0, "", ""
    for pc in range(12):
        for minor, ref in ((False, KK_MAJOR), (True, KK_MINOR)):
            r = np.corrcoef(prof, np.roll(ref, pc))[0, 1]
            if r > best:
                best = r
                name = MINOR_NAMES[pc] if minor else PITCHES[pc]
                cam = f"{(CAMELOT_MINOR if minor else CAMELOT_MAJOR)[pc]}{'A' if minor else 'B'}"
    return name, cam


def phrases(a: dict, lo: np.ndarray, mid: np.ndarray) -> dict:
    """Intro end, drops and outro start, on 8-bar boundaries (for the autopilot)."""
    energy = lo + mid
    bar_ms = a["period_ms"] * 4
    first_down = a["first_beat_ms"] + a["downbeat_index"] * a["period_ms"]
    block_ms = bar_ms * 8
    starts = np.arange(first_down, a["duration_ms"] - block_ms / 2, block_ms)
    if len(starts) < 3:
        return {"first_downbeat_ms": first_down, "intro_end_ms": None, "outro_start_ms": None, "drops": []}
    levels = []
    for s in starts:
        i0, i1 = int(s * 0.15), int(min(s + block_ms, a["duration_ms"]) * 0.15)
        levels.append(float(energy[i0:i1].mean()) if i1 > i0 else 0.0)
    levels = np.array(levels) / (max(levels) or 1)
    loud = levels >= 0.7
    intro_end = next((float(starts[i]) for i in range(1, len(starts)) if loud[i] and not loud[: i].any()), None)
    outro = None
    for i in range(len(starts) - 1, 0, -1):
        if loud[i - 1] and not loud[i:].any():
            outro = float(starts[i])
            break
    drops = [float(starts[i]) for i in range(1, len(starts)) if levels[i] - levels[i - 1] > 0.25]
    return {"first_downbeat_ms": first_down, "intro_end_ms": intro_end, "outro_start_ms": outro, "drops": drops}


def energy_rating(peak: np.ndarray, lo, mid, hi, onset_rate: float) -> int:
    loud = 20 * np.log10(np.sqrt(((lo + mid + hi) ** 2).mean()) + 1e-9)
    bright = float(hi.mean() / ((lo + mid + hi).mean() + 1e-9))
    score = 0.5 * np.clip((loud + 30) / 24, 0, 1) + 0.3 * np.clip(onset_rate / 6, 0, 1) + 0.2 * np.clip(bright * 3, 0, 1)
    return int(np.clip(round(1 + score * 9), 1, 10))


# ── Waveforms in rekordbox's formats ─────────────────────────────────────────

def waveforms(peak, lo, mid, hi) -> dict:
    ref = np.percentile(peak, 99.5) or 1.0
    h = np.clip(peak / ref, 0, 1)
    total = lo + mid + hi + 1e-9
    white = np.clip(hi / total * 2, 0, 1)

    # PWV3: 1 byte per column, height 0-31, whiteness 0-7.
    pwv3 = ((np.round(white * 7).astype(np.uint8) << 5) | np.round(h * 31).astype(np.uint8)).astype(np.uint8)

    # PWV5: 2 bytes per column, RGB 3 bits each (bass=red, mids=green, highs=blue), height 5 bits.
    m = np.maximum(np.maximum(lo, mid), hi) + 1e-9
    r, g, b = (np.round(x / m * 7).astype(np.uint16) for x in (lo, mid, hi))
    pwv5 = ((r << 13) | (g << 10) | (b << 7) | (np.round(h * 31).astype(np.uint16) << 2)).astype(">u2")

    def columns(n):
        edges = np.linspace(0, len(h), n + 1).astype(int)
        return [(slice(edges[i], max(edges[i + 1], edges[i] + 1))) for i in range(n)]

    cols400 = columns(400)
    pwav = np.array([(int(round(white[s].mean() * 7)) << 5) | int(round(h[s].max() * 31)) for s in cols400], np.uint8)
    pwv2 = np.array([int(round(h[s].max() * 15)) for s in columns(100)], np.uint8)

    # PWV4: 1200 columns × 6 bytes: height, height, 0, bass, mids, highs (0-127).
    band_ref = [np.percentile(x, 99.5) or 1.0 for x in (lo, mid, hi)]
    pwv4 = np.zeros((1200, 6), np.uint8)
    for i, s in enumerate(columns(1200)):
        hh = int(round(h[s].max() * 127))
        pwv4[i, 0] = pwv4[i, 1] = hh
        for j, x in enumerate((lo, mid, hi)):
            pwv4[i, 3 + j] = int(np.clip(round(x[s].max() / band_ref[j] * 127), 0, 127))
    return {"pwav": _b64(pwav), "pwv2": _b64(pwv2), "pwv3": _b64(pwv3), "pwv4": _b64(pwv4), "pwv5": _b64(pwv5)}


# ── Main entry ───────────────────────────────────────────────────────────────

def analyse(track_id: int, progress=lambda f, msg="": None) -> dict:
    t = db.one("SELECT * FROM tracks WHERE id=?", (track_id,))
    if not t:
        raise ValueError("unknown track")
    settings = config.load_settings()
    progress(0.05, "Loading audio")
    y, _ = librosa.load(t["path"], sr=SR, mono=True)
    duration_ms = len(y) / SR * 1000
    progress(0.25, "Waveform")
    peak, lo, mid, hi = band_envelopes(y)
    progress(0.4, "Beat grid")
    a = beat_grid(y, lo, settings["bpm_min"], settings["bpm_max"])
    a["duration_ms"] = duration_ms
    progress(0.65, "Key")
    key, camelot = detect_key(y)
    progress(0.8, "Phrases and energy")
    onsets = librosa.onset.onset_detect(y=y, sr=SR, units="time")
    a.update(
        version=1, track_id=track_id, key=key, camelot=camelot,
        energy=energy_rating(peak, lo, mid, hi, len(onsets) / max(duration_ms / 1000, 1)),
        phrases=phrases(a, lo, mid), waveforms=waveforms(peak, lo, mid, hi), analysed_at=time.time(),
    )
    (config.ANALYSIS_DIR / f"{track_id}.json").write_text(json.dumps(a), "utf-8")

    with db.tx() as c:
        c.execute("UPDATE tracks SET bpm=?, musical_key=?, camelot=?, energy=?, duration=?, analysed_at=? WHERE id=?",
                  (a["bpm"], key, camelot, a["energy"], duration_ms / 1000, a["analysed_at"], track_id))
        # Place automatic cues only if the user hasn't set any.
        if not c.execute("SELECT 1 FROM cues WHERE track_id=? AND auto=0", (track_id,)).fetchone():
            c.execute("DELETE FROM cues WHERE track_id=? AND auto=1", (track_id,))
            ph = a["phrases"]
            auto = [(1, ph["first_downbeat_ms"], "Start")]
            if ph["drops"]:
                auto.append((2, ph["drops"][0], "Drop"))
            if ph["intro_end_ms"]:
                auto.append((0, ph["intro_end_ms"], "Intro end"))
            if ph["outro_start_ms"]:
                auto.append((0, ph["outro_start_ms"], "Outro"))
            c.executemany("INSERT INTO cues(track_id, slot, start_ms, label, auto) VALUES (?,?,?,?,1)",
                          [(track_id, s, int(ms), lab) for s, ms, lab in auto])
    progress(1.0, "Done")
    return a
