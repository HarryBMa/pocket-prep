r"""Synthetic training tracks for Pulse from the sample-pack loops on D: (SampleRadar, Riemann, Bandcamp packs):
every beat and downbeat is known exactly, at any tempo (no cap), including breakdowns without drums.

A loop is used when its tempo is in its path (e.g. "techno-125bpm", "..._95 bpm") and its length is a whole
number of bars at that tempo; it is assumed to start on a downbeat (as sample-pack loops do). Role from its
path: drums (kick / hat / perc / drum / beat / top / loop), bass (bass / sub), music (synth / chord / pad /
lead / stab / arp / keys / piano / organ / guitar / fx), vocals (vocal / vox / voz / adlib).

One track: tempo T (half from the loops' own tempos, half 70-200 BPM), 64 bars in 8-bar sections; drums,
bass, music, vocal loops each sped up / slowed to T (only within 0.8-1.25x, half / double time allowed),
switched on and off per section (one or two breakdowns without drums), mixed at random levels, soft-clipped.
Written as Beat This! log-mels + labels, the same layout as beat_rave (Rave(root=...) reads it).

    .venv\Scripts\python.exe pulse_loops.py --n 1500         -> F:\needle\beat_loops
"""
import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf
import soxr
import torch

ROOTS = [Path("D:/"), Path(r"I:\datasets\bandcamp")]
OUT = Path(r"F:\needle\beat_loops")
SR = 22050
ROLE = [("vocals", r"vocal|vox|voz|adlib|acapella"),
        ("bass", r"bass|sub\b|sub[ _-]"),
        ("drums", r"kick|hat|perc|drum|beat|top|groove|clap|snare|ride|shaker|loop"),
        ("music", r"synth|chord|pad|lead|stab|arp|keys|piano|organ|guitar|fx|melod|seq|riff|pluck")]
BPM = re.compile(r"(\d{2,3}(?:\.\d+)?)\s*[-_ ]?bpm", re.I)
SKIP = re.compile(r"Downloads|FitGirl|Repack|\$RECYCLE|System Volume", re.I)


def scan():
    loops = defaultdict(list)
    for root in ROOTS:
        for p in root.rglob("*"):
            if p.suffix.lower() not in (".wav", ".flac", ".aif", ".aiff") or SKIP.search(str(p)):
                continue
            m = BPM.findall(str(p))
            if not m:
                continue
            bpm = float(m[-1])                                          # the most specific (file name over folder)
            if not 60 <= bpm <= 200:
                continue
            low = str(p).lower()
            role = next((r for r, pat in ROLE if re.search(pat, p.name.lower())), None) \
                or next((r for r, pat in ROLE if re.search(pat, low)), None)
            if role is None:
                continue
            try:
                info = sf.info(str(p))
            except Exception:                                           # noqa: BLE001
                continue
            beats = info.duration * bpm / 60
            bars = round(beats / 4)
            if bars < 1 or bars > 16 or abs(beats - bars * 4) > 0.08 * 4 / 4:   # whole bars, within 0.08 beat
                continue
            loops[role].append((str(p), bpm, bars))
    return loops


def load_loop(path, bpm, bars, tempo):
    """-> mono float32 at SR, exactly `bars` bars long at `tempo`, or None if the speed change is too large."""
    for mult in (1, 2, 0.5):                                            # half / double time
        s = tempo / (bpm * mult)
        if 0.8 <= s <= 1.25:
            break
    else:
        return None
    x, sr = sf.read(path, dtype="float32", always_2d=True)
    x = x.mean(1)
    n_src = int(round(bars * 4 * 60 / bpm * sr))                       # trim to whole bars at the native tempo
    x = x[:n_src] if len(x) >= n_src else np.pad(x, (0, n_src - len(x)))
    y = soxr.resample(x, sr, SR / s)                                    # played at SR: s times faster
    n = int(round(len(x) / sr / s * SR))                               # bars x 4 x (beats at tempo / mult)
    return y[:n] if len(y) >= n else np.pad(y, (0, n - len(y)))


def make_track(loops, rng, tempo):
    bar = 4 * 60 / tempo
    n_bars = 64
    total = int(round(n_bars * bar * SR)) + SR
    mix = np.zeros(total, np.float32)
    sections = n_bars // 8
    plan = {"drums": np.ones(sections, bool), "bass": rng.random(sections) < 0.75,
            "music": rng.random(sections) < 0.7, "vocals": rng.random(sections) < 0.3}
    plan["drums"][rng.choice(np.arange(1, sections - 1), size=rng.integers(1, 3), replace=False)] = False   # breakdowns
    if not plan["vocals"].any() or not loops.get("vocals"):
        plan["vocals"][:] = False
    for role, on in plan.items():
        if not on.any() or not loops.get(role):
            continue
        for _ in range(2 if role == "drums" else 1):                   # drums: two layers (e.g. kick loop + top loop)
            seg = None
            for _try in range(8):
                p, bpm, bars = loops[role][rng.integers(len(loops[role]))]
                seg = load_loop(p, bpm, bars, tempo)
                if seg is not None:
                    break
            if seg is None:
                continue
            seg = seg / (np.abs(seg).max() + 1e-6) * 10 ** (rng.uniform(-14, -3) / 20)
            for s in range(sections):
                if not on[s]:
                    continue
                for b in range(s * 8, s * 8 + 8):
                    if (b - s * 8) % max(1, round(len(seg) / SR / bar)) != 0:
                        continue                                        # place the loop at its own bar period
                    a = int(round(b * bar * SR))
                    e = min(total, a + len(seg))
                    mix[a:e] += seg[: e - a]
    mix = np.tanh(mix * rng.uniform(1.0, 3.0)) * 0.7                   # crude glue / master
    beats = np.arange(0, n_bars * 4) * 60 / tempo
    return mix[: int(round(n_bars * bar * SR))], beats, beats[::4]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1500)
    a = ap.parse_args()
    from beat_this.preprocessing import LogMelSpect
    loops = scan()
    print({k: len(v) for k, v in loops.items()}, flush=True)
    if len(loops.get("drums", [])) < 20:
        raise SystemExit("too few drum loops with a tempo in the name")
    (OUT / "mels").mkdir(parents=True, exist_ok=True)
    (OUT / "loops.json").write_text(json.dumps(loops, indent=1))
    spect = LogMelSpect(device="cuda")
    rng = np.random.default_rng(0)
    native = [bpm for v in loops.values() for _, bpm, _ in v]
    index = {}
    for i in range(a.n):
        tempo = float(rng.choice(native)) if rng.random() < 0.5 else float(rng.uniform(70, 200))
        tempo = round(tempo, 2)
        x, beats, downs = make_track(loops, rng, tempo)
        with torch.inference_mode():
            m = spect(torch.from_numpy(x).cuda()).T.half().cpu().numpy()
        tid = f"loop{i:05d}"
        np.save(OUT / "mels" / f"{tid}.npy", m)
        np.savez(OUT / f"{tid}.npz", beats=beats.astype(np.float32), downbeats=downs.astype(np.float32))
        index[tid] = {"tempo": tempo, "frames": int(m.shape[1])}
        if i % 100 == 0:
            print(f"[{i + 1}/{a.n}] {tempo} BPM", flush=True)
    (OUT / "index.json").write_text(json.dumps(index, indent=1))
    (OUT / "test.txt").write_text("")
    print(len(index), "synthetic tracks")


if __name__ == "__main__":
    main()
