r"""Pulse pass / fail: one checkpoint through the firmware's C++ (pulse_c\pulse_beat.cpp) on the two test sets.

  mixed   459 mixed-genre tracks: up to 40 per 10-BPM bin of the Beat This! teacher set (the draw of
          F:\needle\fw\h6_eval\dump_more.py), stored log-mels -> C++ network -> C++ grid fit. Reference tempo:
          60 / the median interval of the teacher's beats. A grid is "double" / "half" when its tempo is within
          2.8 % (0.04 octave) of 2x / 0.5x the reference. "unseen" = not in the training split of the ensemble
          teacher set that p3 and later were trained on.
  rave    the 200 Raveform test tracks (never trained on, human beats and downbeats), twice: from the stored
          log-mels like the mixed set, and end to end from the audio (resampler, log-mel, network, grid; 3 threads)
            tempo right        within 2 % of the human tempo, of the tracks with a steady human grid
            bar line on beat 1 share of the tempo-right tracks where > 90 % of the human downbeats sit on the
                               anchor's bar grid (+-70 ms)
            drum & bass        grid within 2 % of the human tempo (60 / the median beat interval), among all
                               tracks over 150 BPM (23 of the 200, steady human grid or not)
            beat F             grid beats vs human beats, +-70 ms, over the annotated span, mean over all tracks
                               (pulse_grid.py gives the Python grid's beat F, the number quoted before)

Every number is given with the firmware's double-time rule off and on (Options::doubleTimeRule: a grid whose
every other beat is weak is halved).

    .venv\Scripts\python.exe pulse_passfail.py --model p4_w24_l6            export, build, run, score
    .venv\Scripts\python.exe pulse_passfail.py --model p6_w24_l6 --no-audio  without the end-to-end run
    .venv\Scripts\python.exe pulse_passfail.py --compare p4_w24_l6 p6_w24_l6
        both models' numbers, the mixed tracks that changed class, and per Raveform score the tracks that one model
        gets right and the other wrong, with the sign test's p (how likely such a split is if neither is better)

Everything lands in F:\needle\pulse_passfail\<model> (weights header, executables, logits, result.json).
CPU only. Needs g++ (w64devkit) on the PATH.

The PC's CPU is busy while Needle trains, and the Mac mini has the stored mels too, so the part without audio
can run there (the audio stays on the PC):
    PC   pulse_passfail.py --model M --export-only        weights header + fixtures (needs the checkpoint, beat_this)
    PC   pulse_mac.py code ; pulse_mac.py push pulse_passfail
    Mac  cd ~/train/pulse && PULSE_DATA=~/train/data ~/train/.venv/bin/python pulse_passfail.py --model M --no-export --no-audio
    PC   pulse_mac.py fetch pulse_passfail ; pulse_passfail.py --model M --audio-only
"""
import argparse
import json
import os
import random
import struct
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from pulse import DATA, f_measure, split_ids          # noqa: E402

ROOT = DATA / "pulse_passfail"
RAVE = DATA / "beat_rave"
TEACHER = DATA / "beat_teacher"
SRC = HERE / "pulse_c"
FIELDS = ("ok", "steady", "bpm", "raw", "first", "bar", "on", "peaks", "integer", "parity", "halved", "act", "margin")
RATIOS = ((1, "right"), (2, "double"), (0.5, "half"), (1.5, "3:2"), (2 / 3, "2:3"), (4 / 3, "4:3"), (0.75, "3:4"))


# ── the sets ─────────────────────────────────────────────────────────────────

def mixed_ids():
    """The 459: h6_eval\\dump_more.py's draw (random.Random(7), up to 40 per 10-BPM bin), kept in a file."""
    cache = ROOT / "mixed459_ids.json"
    if cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))
    rng = random.Random(7)
    tempos = {}
    for p in sorted(TEACHER.glob("*.npz")):
        b = np.load(p)["beats"].astype(np.float64)
        if len(b) < 32 or not (TEACHER / "mels" / (p.stem + ".npy")).exists():
            continue
        tempos[p.stem] = 60.0 / float(np.median(np.diff(b)))
    bins = {}
    for t, bpm in tempos.items():
        bins.setdefault(int(bpm // 10), []).append(t)
    ids = []
    for k in sorted(bins):
        some = sorted(bins[k])
        rng.shuffle(some)
        ids += some[:40]
    ids = sorted(ids)
    seen = set(split_ids(DATA / "beat_teacher_ens")[0]["train"])
    out = [{"key": f"t{n:04d}", "id": t, "tempo": tempos[t], "seen": t in seen} for n, t in enumerate(ids)]
    ROOT.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(out, indent=1), encoding="utf-8")
    return out


def rave_ids():
    index = json.loads((RAVE / "index.json").read_text())
    return sorted(t for t in set((RAVE / "test.txt").read_text().split()) if t in index), index


# ── export, build, run ───────────────────────────────────────────────────────

def run(cmd, **kw):
    r = subprocess.run([str(c) for c in cmd], capture_output=True, text=True, **kw)
    if r.returncode != 0:
        raise SystemExit(f"failed ({r.returncode}): {' '.join(str(c) for c in cmd)}\n{r.stdout[-2000:]}\n{r.stderr[-2000:]}")
    return r.stdout


def build(model, out, export=True, compile_=True):
    """pulse_cexport.py -> pulse_weights.h + fixtures; the three executables; the C++ vs PyTorch fixture test."""
    out.mkdir(parents=True, exist_ok=True)
    if not compile_:                                                    # --no-build: the executables that are there
        exe = ".exe" if os.name == "nt" else ""
        return out / ("pulse_logits" + exe), out / ("pulse_cli" + exe)
    if export:
        print(run([sys.executable, HERE / "pulse_cexport.py", "--model", model, "--out", out]).strip(), flush=True)
    elif not (out / "pulse_weights.h").exists():
        raise SystemExit(f"--no-export, but there is no {out / 'pulse_weights.h'}")
    exe = ".exe" if os.name == "nt" else ""
    for name in ("pulse_logits", "pulse_cli", "test_pulse_beat"):
        run(["g++", "-O2", "-std=c++17", f"-I{out}", SRC / f"{name}.cpp", SRC / "pulse_beat.cpp", "-o", out / (name + exe),
             "-pthread"] + (["-static"] if os.name == "nt" else []))
    test = run([out / ("test_pulse_beat" + exe), out])
    print("  " + "\n  ".join(ln for ln in test.strip().splitlines() if not ln.startswith("stages")), flush=True)
    return out / ("pulse_logits" + exe), out / ("pulse_cli" + exe)


def logits(exe, out, name, tracks):
    """tracks: [(key, mel file)] -> {key: (beat, down)} from the C++ network, kept in logits_<name>.npz (float16)."""
    path = out / f"logits_{name}.npz"
    if path.exists():
        z = np.load(path)
        if all(f"{k}__beat" in z.files for k, _ in tracks):
            return {k: (z[f"{k}__beat"].astype(np.float32), z[f"{k}__down"].astype(np.float32)) for k, _ in tracks}
    tmp, res, got, t0 = out / "_mel.f32", out / "_logits.f32", {}, time.time()
    for n, (k, mel) in enumerate(tracks):
        m = np.ascontiguousarray(np.load(mel).astype(np.float32))
        assert m.shape[0] == 128, m.shape
        m.tofile(tmp)
        run([exe, "net", tmp, m.shape[1], res, 3], timeout=900)
        y = np.fromfile(res, dtype=np.float32).reshape(2, -1)
        got[f"{k}__beat"], got[f"{k}__down"] = y[0].astype(np.float16), y[1].astype(np.float16)
        if n % 50 == 0:
            print(f"  {name}: network {n + 1}/{len(tracks)}, {time.time() - t0:.0f} s", flush=True)
    np.savez_compressed(path, **got)
    for f in (tmp, res):
        f.unlink(missing_ok=True)
    return {k: (got[f"{k}__beat"].astype(np.float32), got[f"{k}__down"].astype(np.float32)) for k, _ in tracks}


def grids(exe, out, name, lg, rule):
    """The C++ grid fit on stored logits -> {key: {field: value}}."""
    binf = out / f"_{name}.bin"
    with open(binf, "wb") as f:
        for k, (b, d) in lg.items():
            e = k.encode()
            f.write(struct.pack("<I", len(e)) + e + struct.pack("<I", len(b)) + b.astype(np.float32).tobytes() + d.astype(np.float32).tobytes())
    res = {}
    for line in run([exe, "grid", binf, int(rule)]).splitlines():
        v = line.split()
        res[v[0]] = dict(zip(FIELDS, (float(x) for x in v[1:])))
    binf.unlink(missing_ok=True)
    return res


def audio_grids(cli, out, ids, index, model, rules=(False, True)):
    """End to end from the audio files -> {rule: {id: fields}}. A track that pulse_phase.py prep has already run
    through this model's pulse_cli (F:\\needle\\pulse_phase\\rave_test) is taken from there."""
    def fields(v):
        return dict(ok=float(v[0]), steady=float(v[1]), bpm=float(v[2]), first=float(v[3]), bar=float(v[4]), on=float(v[5]),
                    peaks=float(v[6]), ms=float(v[7]), parity=float(v[9]), halved=float(v[10]), act=float(v[11]),
                    margin=float(v[12]), raw=float(v[13]))
    tmp, res, t0, ran = out / "_audio.f32", {r: {} for r in rules}, time.time(), 0
    for n, t in enumerate(ids):
        ready = DATA / "pulse_phase" / "rave_test" / f"{t}.npz"
        g = None
        if ready.exists():
            z = np.load(ready)
            g = z[f"grid_{model}"] if f"grid_{model}" in z.files else None
        if g is not None:
            for rule in rules:
                res[rule][t] = fields(g[int(rule)])
            continue
        import soundfile as sf
        try:
            x, sr = sf.read(index[t]["file"], dtype="float32", always_2d=True)
        except Exception as ex:                                             # noqa: BLE001
            print("  no audio:", t, ex, flush=True)
            continue
        x = x[:, :2] if x.shape[1] >= 2 else np.repeat(x, 2, 1)
        np.ascontiguousarray(x).tofile(tmp)
        for rule in rules:
            res[rule][t] = fields(run([cli, tmp, sr, 3, 0, int(rule)], timeout=900).split())
        ran += 1
        if ran % 20 == 1:
            print(f"  rave audio {n + 1}/{len(ids)}, {time.time() - t0:.0f} s", flush=True)
    tmp.unlink(missing_ok=True)
    print(f"  rave audio: {len(ids) - ran} tracks from pulse_phase.py prep, {ran} run now", flush=True)
    return res


# ── scores ───────────────────────────────────────────────────────────────────

def human_line(b):
    k = np.arange(len(b))
    period, phase = np.polyfit(k, b, 1)
    return 60 / period, phase, float(np.std(b - (phase + k * period)) * 1000)


def ratio_class(q):
    d, name = min((abs(np.log2(q / c)), n) for c, n in RATIOS)
    return name if d < 0.04 else "other"


def score_mixed(ids, g):
    rows = []
    for v in ids:
        r = g[v["key"]]
        cls = ratio_class(r["bpm"] / v["tempo"]) if r["ok"] and r["steady"] else "no grid"
        rows.append({"key": v["key"], "id": v["id"], "ref": v["tempo"], "bpm": r["bpm"], "class": cls, "seen": v["seen"],
                     "halved": bool(r["halved"]), "parity": r["parity"], "on": r["on"], "act": r["act"], "margin": r["margin"]})
    return rows


def score_rave(ids, g):
    rows = []
    for t in ids:
        z = np.load(RAVE / f"{t}.npz")
        b, d = z["beats"].astype(np.float64), z["downbeats"].astype(np.float64)
        hb, hphase, jit = human_line(b)
        r = g.get(t)
        row = {"id": t, "human_bpm": hb, "steady_human": jit < 15, "grid": bool(r and r["ok"] and r["steady"]), "tempo_right": False,
               "beat_f": 0.0, "human_bpm_median": 60 / float(np.median(np.diff(b)))}
        if r:
            row.update(bpm=r["bpm"], halved=bool(r["halved"]), parity=r["parity"], on=r["on"], act=r["act"], margin=r["margin"],
                       bar_vote=r["bar"], anchor=r["first"])
        if row["grid"]:
            row["tempo_right"] = bool(abs(r["bpm"] / hb - 1) < 0.02)
            p = 60 / r["bpm"]
            k0, k1 = np.ceil((b[0] - p / 2 - r["first"]) / p), np.floor((b[-1] + p / 2 - r["first"]) / p)
            row["beat_f"] = float(f_measure(b, r["first"] + np.arange(k0, k1 + 1) * p))
            if row["tempo_right"]:
                ph = 60 / hb
                row["offset_ms"] = float(((r["first"] - hphase + ph / 2) % ph - ph / 2) * 1000)      # grid minus human, signed
                bar = 4 * p
                row["bar_share"] = float(np.mean(np.abs((d - r["first"] + bar / 2) % bar - bar / 2) < 0.07)) if len(d) else 0.0
        rows.append(row)
    return rows


def sum_mixed(rows):
    out = {}
    for name, sel in (("all", rows), ("unseen", [r for r in rows if not r["seen"]]), ("seen", [r for r in rows if r["seen"]])):
        c = {k: 0 for k in ("right", "double", "half", "other", "no grid")}
        for r in sel:
            c[r["class"] if r["class"] in c else "other"] += 1
        out[name] = dict(n=len(sel), **c)
    out["double_by_ref"] = {f"{lo}-{hi}": [sum(r["class"] == "double" for r in rows if lo <= r["ref"] < hi), sum(lo <= r["ref"] < hi for r in rows)]
                            for lo, hi in ((60, 80), (80, 105), (105, 150), (150, 260))}
    out["half_tracks"] = sorted((round(r["ref"], 1), round(r["bpm"], 1), r["id"]) for r in rows if r["class"] == "half")
    return out


def sum_rave(rows):
    st = [r for r in rows if r["steady_human"]]
    tr = [r for r in st if r["tempo_right"]]
    off = np.array([r["offset_ms"] for r in tr])
    bs = np.array([r["bar_share"] for r in tr])
    dnb = [r for r in rows if r["human_bpm_median"] > 150]
    dnb_right = [r for r in dnb if r["grid"] and abs(r["bpm"] / r["human_bpm_median"] - 1) < 0.02]
    return {"tracks": len(rows), "steady_human": len(st), "tempo_right": len(tr), "no_grid": sum(not r["grid"] for r in st),
            "wrong": sorted((round(r["human_bpm"], 1), round(r.get("bpm", 0), 1), r["id"]) for r in st if r["grid"] and not r["tempo_right"]),
            "offset_median_abs_ms": float(np.median(np.abs(off))), "offset_median_signed_ms": float(np.median(off)),
            "offset_over_20ms": float(np.mean(np.abs(off) > 20)), "bar_line_over_90": float(np.mean(bs > 0.9)),
            "bar_share_mean": float(bs.mean()), "dnb": [len(dnb_right), len(dnb)],
            "dnb_wrong": sorted((round(r["human_bpm_median"], 1), round(r.get("bpm", 0), 1) if r["grid"] else 0.0, r["id"]) for r in dnb if r not in dnb_right),
            "beat_f": float(np.mean([r["beat_f"] for r in rows])), "halved": sum(bool(r.get("halved")) for r in rows)}


def show(model, res):
    print(f"\n{model}")
    for rule in ("off", "on"):
        m = res["mixed"][rule]["summary"]
        a = m["all"]
        print(f"  mixed 459, rule {rule:3s}: right {a['right']}, double {a['double']}, half {a['half']}, other {a['other']}, no grid {a['no grid']}"
              f" | unseen ({m['unseen']['n']}): double {m['unseen']['double']}, half {m['unseen']['half']}"
              f" | double by reference tempo: " + ", ".join(f"{k} BPM {v[0]}/{v[1]}" for k, v in m["double_by_ref"].items()))
        if m["half_tracks"]:
            print("      halved (reference, grid):", ", ".join(f"{a_} -> {b_}" for a_, b_, _ in m["half_tracks"]))
    for kind, title in (("rave", "Raveform 200, stored mels"), ("rave_audio", "Raveform 200, from audio ")):
        for rule in ("off", "on"):
            if kind not in res or rule not in res[kind]:
                continue
            s = res[kind][rule]["summary"]
            print(f"  {title}, rule {rule:3s}: tempo right {s['tempo_right']}/{s['steady_human']}, beat F {s['beat_f']:.3f}, "
                  f"bar line on beat 1 {s['bar_line_over_90'] * 100:.0f} %, drum & bass {s['dnb'][0]}/{s['dnb'][1]}, "
                  f"|offset| median {s['offset_median_abs_ms']:.1f} ms (> 20 ms: {s['offset_over_20ms'] * 100:.0f} %), "
                  f"no grid {s['no_grid']}, halved {s['halved']}")
            if s["wrong"]:
                print("      wrong tempo (human, grid):", ", ".join(f"{a_} -> {b_}" for a_, b_, _ in s["wrong"]))
            if s.get("dnb_wrong"):
                print("      drum & bass misses (human, grid; 0 = no grid):", ", ".join(f"{a_} -> {b_}" for a_, b_, _ in s["dnb_wrong"]))


def sign_test(a, b):
    """Two-sided exact sign test: a tracks changed one way, b the other; p of a split at least this uneven if
    either way is equally likely."""
    from math import comb
    n, k = a + b, min(a, b)
    return 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def compare_rave(names, both):
    """Per Raveform score: the tracks one model gets right and the other wrong (the same 200 tracks for both)."""
    for kind, title in (("rave", "stored mels"), ("rave_audio", "from audio")):
        for rule in ("off", "on"):
            if not all(kind in r and rule in r[kind] for r in both):
                continue
            x, y = ({r["id"]: r for r in b[kind][rule]["rows"]} for b in both)
            print(f"\nRaveform 200, {title}, rule {rule}: {names[0]} -> {names[1]}")
            tests = (("steady grid", lambda r: True, lambda r: r["grid"]),
                     ("tempo right", lambda r: r["steady_human"], lambda r: r["tempo_right"]),
                     ("bar line on beat 1", lambda r: r["steady_human"], lambda r: r.get("bar_share", 0.0) > 0.9),
                     ("grid within 10 ms of the human one", lambda r: r["steady_human"], lambda r: abs(r.get("offset_ms", 1e9)) <= 10))
            for name, counts, good in tests:
                ids = [t for t in x if t in y and counts(x[t])]
                lost = sum(good(x[t]) and not good(y[t]) for t in ids)
                won = sum(good(y[t]) and not good(x[t]) for t in ids)
                print(f"  {name:36s} {sum(good(x[t]) for t in ids):3d} -> {sum(good(y[t]) for t in ids):3d} of {len(ids)}: "
                      f"{lost} lost, {won} gained (sign test p = {sign_test(lost, won):.2f})")
            fx, fy = (np.array([r[t]["beat_f"] for t in x if t in y]) for r in (x, y))
            d = fy - fx
            boot = np.random.default_rng(0).choice(d, (4000, len(d))).mean(1)
            print(f"  {'beat F (mean over the tracks)':36s} {fx.mean():.3f} -> {fy.mean():.3f}: difference {d.mean():+.4f} "
                  f"(95 % bootstrap {np.percentile(boot, 2.5):+.4f} .. {np.percentile(boot, 97.5):+.4f})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", help="a runs_pulse name")
    ap.add_argument("--no-audio", action="store_true", help="skip the end-to-end run on the Raveform audio")
    ap.add_argument("--export-only", action="store_true", help="only the weights header, the fixtures, the build and its test")
    ap.add_argument("--no-export", action="store_true", help="build from the pulse_weights.h already in the output folder")
    ap.add_argument("--audio-only", action="store_true", help="add the end-to-end run to the result.json that is there")
    ap.add_argument("--no-build", action="store_true", help="use the executables that are in the output folder (one may be in "
                    "use: Windows cannot replace a running .exe)")
    ap.add_argument("--compare", nargs=2, metavar=("BEFORE", "AFTER"))
    a = ap.parse_args()
    if a.compare:
        both = [json.loads((ROOT / m / "result.json").read_text()) for m in a.compare]
        for m, r in zip(a.compare, both):
            show(m, r)
        for rule in ("off", "on"):
            x, y = ({r["key"]: r for r in b["mixed"][rule]["rows"]} for b in both)
            moved = {}
            for k in x:
                if x[k]["class"] != y[k]["class"]:
                    moved.setdefault(f"{x[k]['class']} -> {y[k]['class']}", []).append(round(x[k]["ref"], 1))
            print(f"\nmixed 459, rule {rule}: tracks that changed class, {a.compare[0]} -> {a.compare[1]}")
            for k, v in sorted(moved.items(), key=lambda kv: -len(kv[1])):
                print(f"  {k}: {len(v)} (reference tempos {sorted(v)[:12]}{' ...' if len(v) > 12 else ''})")
        compare_rave(a.compare, both)
        return
    out = ROOT / a.model
    exe, cli = build(a.model, out, export=not a.no_export, compile_=not a.no_build)
    if a.export_only:
        mixed_ids()                                                     # so the list travels with the folder
        return
    rids, index = rave_ids()
    if a.audio_only:
        res = json.loads((out / "result.json").read_text())
        res["rave_audio"] = {}
        for rule, g in audio_grids(cli, out, rids, index, a.model).items():
            rows = score_rave(rids, g)
            res["rave_audio"]["on" if rule else "off"] = {"summary": sum_rave(rows), "rows": rows}
        (out / "result.json").write_text(json.dumps(res, indent=1))
        show(a.model, res)
        print("PASSFAIL DONE", flush=True)
        return
    res = {"model": a.model, "mixed": {}, "rave": {}}
    mids = mixed_ids()
    lg_m = logits(exe, out, "mixed", [(v["key"], TEACHER / "mels" / f"{v['id']}.npy") for v in mids])
    lg_r = logits(exe, out, "rave", [(t, RAVE / "mels" / f"{t}.npy") for t in rids])
    for rule, name in ((False, "off"), (True, "on")):
        rows = score_mixed(mids, grids(exe, out, "mixed", lg_m, rule))
        res["mixed"][name] = {"summary": sum_mixed(rows), "rows": rows}
        rows = score_rave(rids, grids(exe, out, "rave", lg_r, rule))
        res["rave"][name] = {"summary": sum_rave(rows), "rows": rows}
    (out / "result.json").write_text(json.dumps(res, indent=1))
    show(a.model, res)
    if not a.no_audio:
        res["rave_audio"] = {}
        for rule, g in audio_grids(cli, out, rids, index, a.model).items():
            rows = score_rave(rids, g)
            res["rave_audio"]["on" if rule else "off"] = {"summary": sum_rave(rows), "rows": rows}
        (out / "result.json").write_text(json.dumps(res, indent=1))
        show(a.model, res)
    print("PASSFAIL DONE", flush=True)


if __name__ == "__main__":
    main()
