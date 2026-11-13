r"""One tempo per synthetic track without the sample packs: v1's tracks with the labels put right.

pulse_loops.py v1 used a loop at half or double time when its own tempo did not fit the track's (see that
file). The audio of those tracks is fine; the labels sit at the wrong level for the drums. v1's random draws do
not depend on the audio, so this tool replays them and knows, for every v1 track, which loops it was made from
and at which level each one was used. It writes a new set next to the old one:

  all drum layers at the label tempo           kept as it is
  all drum layers at half / double the label   relabelled to the tempo the drums play at (the beats in
                                               between added, or every other beat dropped; bar lines every
                                               4 beats from the start, where the loops start)
  drum layers at two different levels,         left out: no single tempo is right
  or no drum layer at all

The rule is pulse_loops.py's: a track's tempo is the tempo its drum loops were made at. Bass, music and vocal
loops may still sit at half or double time against the drums here (in a freshly made set they do not).

The mels are the same files as v1's (hard links, no copy). The replay is checked against v1's index.json:
every track's tempo has to come out the same, and it only does when every draw before it was the same.

Make a fresh set with pulse_loops.py when the sample packs (D:) are connected; this is for when they are not,
and it changes nothing but the labels: a model trained on it against one trained on v1 shows what the labels
alone do.

    .venv\Scripts\python.exe pulse_loops_relabel.py        F:\needle\beat_loops_v1 -> F:\needle\beat_loops_1t
"""
import argparse
import json
import os
import shutil
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pulse import DATA          # noqa: E402

N_BARS = 64                     # v1: every track is 64 bars of 4 beats at its label tempo


def v1_level(bpm, tempo):
    """v1's load_loop: the first of (own tempo, double time, half time) that needs a speed change within
    0.8-1.25x. The loop then plays at tempo / level: 2 = half the label tempo, 0.5 = double it."""
    for mult in (1, 2, 0.5):
        if 0.8 <= tempo / (bpm * mult) <= 1.25:
            return mult
    return None


def replay(loops, n):
    """v1's main() and make_track() with the audio left out: -> [(tempo, [(role, file, bpm, bars, level)])].
    Every rng call is in v1's order, also the ones whose value is not needed here (levels, master gain)."""
    rng = np.random.default_rng(0)
    native = [bpm for v in loops.values() for _, bpm, _ in v]
    sections = N_BARS // 8
    out = []
    for _ in range(n):
        tempo = float(rng.choice(native)) if rng.random() < 0.5 else float(rng.uniform(70, 200))
        tempo = round(tempo, 2)
        plan = {"drums": np.ones(sections, bool), "bass": rng.random(sections) < 0.75,
                "music": rng.random(sections) < 0.7, "vocals": rng.random(sections) < 0.3}
        plan["drums"][rng.choice(np.arange(1, sections - 1), size=rng.integers(1, 3), replace=False)] = False
        if not plan["vocals"].any() or not loops.get("vocals"):
            plan["vocals"][:] = False
        layers = []
        for role, on in plan.items():
            if not on.any() or not loops.get(role):
                continue
            for _layer in range(2 if role == "drums" else 1):
                got = None
                for _try in range(8):
                    p, bpm, bars = loops[role][rng.integers(len(loops[role]))]
                    level = v1_level(bpm, tempo)
                    if level is not None:
                        got = (role, p, bpm, bars, level)
                        break
                if got is None:
                    continue
                rng.uniform(-14, -3)                                    # the layer's level in the mix
                layers.append(got)
        rng.uniform(1.0, 3.0)                                           # the master gain
        out.append((tempo, layers))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(DATA / "beat_loops_v1"), help="the v1 set")
    ap.add_argument("--out", default=str(DATA / "beat_loops_1t"))
    ap.add_argument("--dry", action="store_true", help="only replay, check and count")
    a = ap.parse_args()
    src, out = Path(a.src), Path(a.out)
    loops = json.loads((src / "loops.json").read_text())
    old = json.loads((src / "index.json").read_text())
    ids = sorted(old)
    tracks = replay(loops, len(ids))
    wrong = [t for t, (tempo, _) in zip(ids, tracks) if tempo != old[t]["tempo"]]
    if wrong:
        raise SystemExit(f"the replay does not match v1's index.json: {len(wrong)} of {len(ids)} tempos differ "
                         f"(first: {wrong[0]}). Another numpy, or another loops.json than v1 was made with.")
    print(f"replay matches v1: all {len(ids)} tempos equal", flush=True)

    index, why = {}, Counter()
    for t, (tempo, layers) in zip(ids, tracks):
        drum = sorted({lv for role, _, _, _, lv in layers if role == "drums"})
        if not drum:
            why["left out: no drum layer"] += 1
            continue
        if len(drum) > 1:
            why["left out: drum layers at two levels"] += 1
            continue
        level = drum[0]
        why[{1: "kept", 2: "relabelled to half the v1 tempo", 0.5: "relabelled to double the v1 tempo"}[level]] += 1
        new = tempo / level
        index[t] = {"tempo": round(new, 2), "frames": old[t]["frames"], "v1_tempo": tempo,
                    "loops": [{"role": r, "bpm": b, "bars": n, "speed": round(new / (b * lv / level), 4),
                               "vs_drums": lv / level, "file": p} for r, p, b, n, lv in layers]}
    for k, v in sorted(why.items()):
        print(f"  {v:5d}  {k}")
    te = np.array([v["tempo"] for v in index.values()])
    other = [lp["vs_drums"] != 1 for v in index.values() for lp in v["loops"] if lp["role"] != "drums"]
    drum_speed = [lp["speed"] for v in index.values() for lp in v["loops"] if lp["role"] == "drums"]
    print(f"{len(index)} of {len(ids)} tracks; drum loops play at {min(drum_speed):.3f}-{max(drum_speed):.3f}x their own "
          f"tempo; {np.mean(other):.1%} of the bass / music / vocal layers at half or double time against the drums")
    hist = Counter(int(x // 10) * 10 for x in te)
    print("tempo histogram (10-BPM bins):", dict(sorted(hist.items())))
    if a.dry:
        return
    (out / "mels").mkdir(parents=True, exist_ok=True)
    linked = 0
    for t, v in index.items():
        n_beats = int(round(N_BARS * 4 * v["tempo"] / v["v1_tempo"]))
        beats = np.arange(n_beats) * 60 / (v["v1_tempo"] * n_beats / (N_BARS * 4))     # exact: the same 64 bars
        np.savez(out / f"{t}.npz", beats=beats.astype(np.float32), downbeats=beats[::4].astype(np.float32))
        dst = out / "mels" / f"{t}.npy"
        if not dst.exists():
            try:
                os.link(src / "mels" / f"{t}.npy", dst)                 # the same file under a second name
                linked += 1
            except OSError:
                shutil.copy2(src / "mels" / f"{t}.npy", dst)
    (out / "index.json").write_text(json.dumps(index, indent=1))
    (out / "test.txt").write_text("")
    (out / "README.txt").write_text(
        f"v1's synthetic tracks ({src}) with one tempo label each: pulse_loops_relabel.py.\n"
        "mels\\ are hard links to v1's mels. index.json: tempo = the tempo the drum loops play at, v1_tempo = v1's label.\n")
    print(f"{out}: {len(index)} tracks, {linked} mels linked", flush=True)


if __name__ == "__main__":
    main()
