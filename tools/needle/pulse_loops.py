r"""Synthetic training tracks for Pulse from the sample-pack loops on D: (SampleRadar, Riemann, Bandcamp packs):
every beat and downbeat is known exactly, at any tempo (no cap), including breakdowns without drums.

A loop is used when its tempo is in its path (e.g. "techno-125bpm", "..._95 bpm") and its length is a whole
number of bars at that tempo; it is assumed to start on a downbeat (as sample-pack loops do). Role from its
path: drums (kick / hat / perc / drum / beat / top / loop), bass (bass / sub), music (synth / chord / pad /
lead / stab / arp / keys / piano / organ / guitar / fx), vocals (vocal / vox / voz / adlib).

One track: tempo T (half from the loops' own tempos, half 70-200 BPM), 64 bars in 8-bar sections; drums,
bass, music, vocal loops each sped up / slowed to T, switched on and off per section (one or two breakdowns
without drums), mixed at random levels, soft-clipped. Written as Beat This! log-mels + labels, the same layout
as beat_rave (Rave(root=...) reads it).

One tempo per track (v2, 12 Nov 2026). A loop is used only at its own tempo, changed by at most 0.8-1.25x:
the track's beats are the beats its loops were made with. That is the rule the hand-annotated Raveform labels
follow: the tempo the producer set and a DJ's software shows (Raveform's beats were drafted from rekordbox's
grid and corrected by hand; its drum & bass is counted at 170-175 BPM, not at half of it). A sample pack's
"170 bpm" or "90 bpm" in the loop's path is that same tempo.
v1 also used a loop at half or double time when its own tempo did not fit. Then a 95-BPM hip-hop loop sat in
a "190 BPM" track with a beat marked on every half beat, and a 140-BPM techno loop in a "70 BPM" track with a
beat on every other kick. In the tracks labelled under 100 or from 160 BPM (35 % of the 1,500) 54 % of the
drum layers played at double or half the label tempo (21 % over all tracks; 458 tracks had such a layer), and
Pulse learned to double slow music. Every track now has drums (a tempo no drum loop fits is drawn again), and
index.json lists the loops each track was made from. v1 is kept as F:\needle\beat_loops_v1.

    .venv\Scripts\python.exe pulse_loops.py --n 1500         -> F:\needle\beat_loops
    ... --loops-json F:\needle\beat_loops_v1\loops.json       the same loops as v1, without scanning D: again
"""
import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf
import soxr
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pulse import DATA, pick_device          # noqa: E402

ROOTS = [Path("D:/"), Path(r"I:\datasets\bandcamp")]
OUT = DATA / "beat_loops"
SR = 22050
SPEED = (0.8, 1.25)                           # how far a loop may be sped up / slowed down from its own tempo
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


def fits(bpm, tempo):
    """A loop goes into a track only at its own tempo (never at half or double time): see the module docstring."""
    return SPEED[0] <= tempo / bpm <= SPEED[1]


def load_loop(path, bpm, bars, tempo):
    """-> mono float32 at SR, exactly `bars` bars long at `tempo`, or None if the speed change is too large."""
    if not fits(bpm, tempo):
        return None
    s = tempo / bpm
    x, sr = sf.read(path, dtype="float32", always_2d=True)
    x = x.mean(1)
    n_src = int(round(bars * 4 * 60 / bpm * sr))                       # trim to whole bars at the native tempo
    x = x[:n_src] if len(x) >= n_src else np.pad(x, (0, n_src - len(x)))
    y = soxr.resample(x, sr, SR / s)                                    # played at SR: s times faster
    n = int(round(len(x) / sr / s * SR))                               # bars x 4 beats at `tempo`
    return y[:n] if len(y) >= n else np.pad(y, (0, n - len(y)))


def make_track(loops, rng, tempo):
    """-> (audio, beat times, downbeat times, [(role, path, loop bpm)]), or None when no drum loop fits `tempo`."""
    usable = {role: [lp for lp in v if fits(lp[1], tempo)] for role, v in loops.items()}
    if not usable.get("drums"):
        return None
    used = []
    bar = 4 * 60 / tempo
    n_bars = 64
    total = int(round(n_bars * bar * SR)) + SR
    mix = np.zeros(total, np.float32)
    sections = n_bars // 8
    plan = {"drums": np.ones(sections, bool), "bass": rng.random(sections) < 0.75,
            "music": rng.random(sections) < 0.7, "vocals": rng.random(sections) < 0.3}
    plan["drums"][rng.choice(np.arange(1, sections - 1), size=rng.integers(1, 3), replace=False)] = False   # breakdowns
    if not plan["vocals"].any() or not usable.get("vocals"):
        plan["vocals"][:] = False
    for role, on in plan.items():
        if not on.any() or not usable.get(role):
            continue
        for _ in range(2 if role == "drums" else 1):                   # drums: two layers (e.g. kick loop + top loop)
            p, bpm, bars = usable[role][rng.integers(len(usable[role]))]
            seg = load_loop(p, bpm, bars, tempo)
            used.append((role, p, bpm))
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
    return mix[: int(round(n_bars * bar * SR))], beats, beats[::4], used


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1500)
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--device", help="for the log-mels: cuda, mps or cpu (default: the first of those that is there)")
    ap.add_argument("--loops-json", help="reuse this scan of the sample packs (loops.json) instead of scanning again")
    a = ap.parse_args()
    from beat_this.preprocessing import LogMelSpect
    out = Path(a.out)
    dev = pick_device(a.device)
    if a.loops_json:
        loops = {k: [tuple(lp) for lp in v] for k, v in json.loads(Path(a.loops_json).read_text()).items()}
    else:
        loops = scan()
    print({k: len(v) for k, v in loops.items()}, f"mels on {dev}", flush=True)
    if len(loops.get("drums", [])) < 20:
        raise SystemExit("too few drum loops with a tempo in the name")
    (out / "mels").mkdir(parents=True, exist_ok=True)
    (out / "loops.json").write_text(json.dumps(loops, indent=1))
    spect = LogMelSpect(device=dev)
    rng = np.random.default_rng(0)
    native = [bpm for v in loops.values() for _, bpm, _ in v]
    index = {}
    for i in range(a.n):
        track = None
        while track is None:                                            # a tempo no drum loop fits: draw again
            tempo = float(rng.choice(native)) if rng.random() < 0.5 else float(rng.uniform(70, 200))
            tempo = round(tempo, 2)
            track = make_track(loops, rng, tempo)
        x, beats, downs, used = track
        with torch.inference_mode():
            m = spect(torch.from_numpy(x).to(dev)).T.half().cpu().numpy()
        tid = f"loop{i:05d}"
        np.save(out / "mels" / f"{tid}.npy", m)
        np.savez(out / f"{tid}.npz", beats=beats.astype(np.float32), downbeats=downs.astype(np.float32))
        index[tid] = {"tempo": tempo, "frames": int(m.shape[1]),
                      "loops": [{"role": r, "bpm": b, "speed": round(tempo / b, 4), "file": p} for r, p, b in used]}
        if i % 100 == 0:
            print(f"[{i + 1}/{a.n}] {tempo} BPM", flush=True)
    (out / "index.json").write_text(json.dumps(index, indent=1))
    (out / "test.txt").write_text("")
    speeds = [lp["speed"] for v in index.values() for lp in v["loops"]]
    print(len(index), f"synthetic tracks; loop speed changes {min(speeds):.3f}-{max(speeds):.3f}x "
          f"(allowed {SPEED[0]}-{SPEED[1]}); every track has drums: "
          f"{all(any(lp['role'] == 'drums' for lp in v['loops']) for v in index.values())}")


if __name__ == "__main__":
    main()
