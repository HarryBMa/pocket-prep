r"""Pulse on music no Pulse model has trained on: a slow-track (and every-tempo) test set that is unseen for every
model alike, unlike the mixed set (nearly all trained on) or holdout_slow_quarter.txt (unseen for p10 only).

Sources: MUSDB18-HQ train + test (mixture.wav), MoisesDB and NI (sum of the four stems), Luke Traxx (bought DJ
tracks). None was in a Pulse teacher set; a title that matches a teacher-set track (pulse_raveform.norm,
difflib >= 0.85) is left out anyway. Reference, as for the mixed set: Beat This! final0-2 averaged (the ensemble
teacher), tempo = 60 / the median beat interval. A reference is "steady" when >= 80 % of its beat intervals are
within 4 % of that median (live-played songs drift; Pulse's grid is one straight line, so the steady ones are the
fair test of tempo).

    .venv\Scripts\python.exe pulse_unseen.py prep                 -> F:\needle\beat_unseen (pulse.Tracks layout)
    .venv\Scripts\python.exe pulse_unseen.py score --model p4_w24_l6 [--model ...]
        C++ network + grid (the executables pulse_passfail.py built in F:\needle\pulse_passfail\<model>), rule off
        and on; -> F:\needle\beat_unseen\score_<model>.json and a table per tempo bin.
    .venv\Scripts\python.exe pulse_unseen.py table p4_w24_l6 p10_w24_l6_bal ...   the tables again, side by side
    .venv\Scripts\python.exe pulse_unseen.py confidence p4_w24_l6
        does a Pulse number predict a wrong tempo on these unseen tracks? AUC per number (pulse_confidence.py), and a
        calibration fitted on the mixed set (pulse_passfail result.json, mostly trained-on tracks) tested here.
        Double-time rule on (as on the deck); fault = a steady grid whose tempo is not the reference's.

CPU only (PULSE_DEVICE is ignored; 3 torch threads), so it can run beside a Needle job. prep resumes.
"""
import argparse
import difflib
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from pulse import DATA                                          # noqa: E402

OUT = DATA / "beat_unseen"
SOURCES = {
    "musdb": [DATA / "data" / "musdb18hq" / "train", DATA / "data" / "musdb18hq" / "test"],
    "moisesdb": [DATA / "data" / "stems" / "moisesdb", DATA / "data" / "stems" / "moisesdb_test"],
    "ni": [DATA / "data" / "stems" / "ni", DATA / "data" / "stems" / "ni_test"],
    "luke": [Path(r"E:\needle-music\Luke Traxx")],
}
AUDIO = (".flac", ".wav", ".mp3", ".m4a", ".aiff", ".aif")
STEMS = ("drums", "bass", "other", "vocals")
BINS = ((0, 80), (80, 105), (105, 150), (150, 999))


def track_id(src: str, name: str, path: str) -> str:
    return f"{src}_" + "".join(c if c.isalnum() else "_" for c in name)[:50] + "-" + hashlib.sha1(path.encode()).hexdigest()[:8]


def find_tracks():
    """-> [(id, source set, title, [audio files to sum])]"""
    out = []
    for src, roots in SOURCES.items():
        for root in roots:
            if not root.exists():
                print("missing:", root, flush=True)
                continue
            if src == "luke":
                for p in sorted(root.rglob("*")):
                    if p.suffix.lower() in AUDIO:
                        out.append((track_id(src, p.stem, str(p)), src, p.stem, [str(p)]))
                continue
            for d in sorted(x for x in root.iterdir() if x.is_dir()):
                mix = d / "mixture.wav"
                if mix.exists():
                    files = [str(mix)]
                else:
                    files = [str(f) for s in STEMS for f in [d / f"{s}.flac", d / f"{s}.wav"] if f.exists()]
                    if len(files) != 4:
                        continue
                out.append((track_id(src, d.name, str(d)), src, d.name, files))
    return out


def teacher_titles():
    from pulse_raveform import norm
    t = []
    for folder in ("beat_teacher_ens", "beat_teacher"):
        idx = DATA / folder / "index.json"
        if idx.exists():
            for v in json.loads(idx.read_text()).values():
                stem = Path(str(v.get("source", ""))).stem
                stem = stem.split(". ", 1)[-1]
                parts = stem.split(" - ")
                t.append(norm(parts[-1] if len(parts) > 1 else stem))
    return norm, sorted(set(x for x in t if len(x) > 3))


def load_mix(files):
    import soundfile as sf
    import soxr
    x, sr = None, None
    for f in files:
        y, r = sf.read(f, dtype="float32", always_2d=True)               # mp3 through libsndfile; m4a fails
        y = y.mean(1)
        if x is None:
            x, sr = y, r
        else:
            n = min(len(x), len(y))
            x = x[:n] + y[:n]
    return (soxr.resample(x, sr, 22050) if sr != 22050 else x).astype(np.float32)


def prep(a):
    import torch
    torch.set_num_threads(3)
    from beat_this.inference import Spect2Frames
    from beat_this.model.postprocessor import Postprocessor
    from beat_this.preprocessing import LogMelSpect
    (OUT / "mels").mkdir(parents=True, exist_ok=True)
    tracks = find_tracks()
    norm, titles = teacher_titles()
    index = json.loads((OUT / "index.json").read_text()) if (OUT / "index.json").exists() else {}
    keep, dup = [], 0
    for t in tracks:
        name = t[2].split(" - ")[-1]
        if t[0] not in index and difflib.get_close_matches(norm(name), titles, n=1, cutoff=0.85):
            dup += 1
            index[t[0]] = {"source": t[3][0], "set": t[1], "title": t[2], "error": "title matches a teacher-set track"}
            continue
        keep.append(t)
    todo = [t for t in keep if t[0] not in index]
    print(f"{len(tracks)} tracks ({', '.join(f'{s} {sum(1 for t in tracks if t[1] == s)}' for s in SOURCES)}); "
          f"{dup} left out as teacher-set titles; {len(todo)} to do", flush=True)
    nets = [Spect2Frames(checkpoint_path=str(DATA / "beat_this" / f"final{i}.ckpt"), device="cpu", float16=False)
            for i in range(3)]
    post, spect, t0 = Postprocessor(type="minimal"), LogMelSpect(device="cpu"), time.time()
    for i, (tid, src, title, files) in enumerate(todo):
        try:
            melp = OUT / "mels" / f"{tid}.npy"
            if not melp.exists():
                with torch.inference_mode():
                    np.save(melp, spect(torch.from_numpy(load_mix(files))).T.half().numpy())
            mel = torch.from_numpy(np.load(melp).astype(np.float32).T)
            with torch.inference_mode():
                pb, pd = 0, 0
                for n in nets:
                    b, d = n(mel)
                    pb, pd = pb + torch.sigmoid(b) / len(nets), pd + torch.sigmoid(d) / len(nets)
                lb, ld = torch.logit(pb.clamp(1e-4, 1 - 1e-4)), torch.logit(pd.clamp(1e-4, 1 - 1e-4))
                beats, downs = post(lb, ld)
            beats = np.asarray(beats, np.float64)
            np.savez_compressed(OUT / f"{tid}.npz", beat=lb.numpy().astype(np.float16), down=ld.numpy().astype(np.float16),
                                beats=beats.astype(np.float32), downbeats=np.asarray(downs, np.float32))
            v = {"source": files[0], "set": src, "title": title, "mel": str(melp), "beats": int(len(beats))}
            if len(beats) >= 32:
                ibi = np.diff(beats)
                med = float(np.median(ibi))
                v["tempo"] = 60.0 / med
                v["steady_share"] = float(np.mean(np.abs(ibi / med - 1) < 0.04))
            index[tid] = v
        except Exception as ex:                                                     # noqa: BLE001
            index[tid] = {"source": files[0], "set": src, "title": title, "error": str(ex)[:200]}
        if i % 20 == 0 or i == len(todo) - 1:
            (OUT / "index.json").write_text(json.dumps(index, indent=1))
            print(f"[{i + 1}/{len(todo)}] {time.time() - t0:.0f} s  {tid}", flush=True)
    (OUT / "index.json").write_text(json.dumps(index, indent=1))
    print("PREP DONE", flush=True)


def ref_tracks():
    index = json.loads((OUT / "index.json").read_text())
    ok = [(k, v) for k, v in sorted(index.items()) if "tempo" in v and (OUT / "mels" / f"{k}.npy").exists()]
    return [{"key": f"u{n:04d}", "id": k, "tempo": v["tempo"], "steady": v["steady_share"] >= 0.8, "set": v["set"]}
            for n, (k, v) in enumerate(ok)]


def score(a):
    import pulse_passfail as pf
    ids = ref_tracks()
    for model in a.model:
        out = DATA / "pulse_passfail" / model
        exe = out / ("pulse_logits.exe" if sys.platform == "win32" else "pulse_logits")
        if not exe.exists():
            sys.exit(f"{exe} missing: run pulse_passfail.py --model {model} --export-only first")
        lg = pf.logits(exe, out, "unseen", [(v["key"], OUT / "mels" / f"{v['id']}.npy") for v in ids])
        res = {"model": model}
        for rule in (False, True):
            g = pf.grids(exe, out, "unseen", lg, rule)
            rows = []
            for v in ids:
                r = g[v["key"]]
                cls = pf.ratio_class(r["bpm"] / v["tempo"]) if r["ok"] and r["steady"] else "no grid"
                rows.append({**v, "bpm": r["bpm"], "class": cls, "halved": bool(r["halved"]), "on": r["on"],
                             "act": r["act"], "margin": r["margin"], "parity": r["parity"]})
            res["on" if rule else "off"] = rows
        (OUT / f"score_{model}.json").write_text(json.dumps(res, indent=1))
    table(a.model)


def table(models):
    def count(rows):
        c = {}
        for r in rows:
            c[r["class"]] = c.get(r["class"], 0) + 1
        return c
    res = {m: json.loads((OUT / f"score_{m}.json").read_text()) for m in models}
    any_rows = next(iter(res.values()))["off"]
    sets = sorted(set(r["set"] for r in any_rows))
    per_set = ", ".join(f"{s} {sum(r['set'] == s for r in any_rows)}" for s in sets)
    print(f"\nunseen set: {len(any_rows)} tracks with a reference ({per_set}); "
          f"steady reference: {sum(r['steady'] for r in any_rows)}")
    for steady_only in (True, False):
        print(f"\n{'steady references only' if steady_only else 'all references'}: right / double / half / other / no grid")
        for lo, hi in BINS:
            sel = [r for r in any_rows if lo <= r["tempo"] < hi and (r["steady"] or not steady_only)]
            print(f"  {lo:3d}-{hi if hi < 999 else '':<3} BPM ({len(sel)} tracks)")
            keys = {r["key"] for r in sel}
            for m, x in res.items():
                line = []
                for rule in ("off", "on"):
                    c = count([r for r in x[rule] if r["key"] in keys])
                    line.append(f"{c.get('right', 0)} / {c.get('double', 0)} / {c.get('half', 0)} / "
                                f"{sum(v for k, v in c.items() if k not in ('right', 'double', 'half', 'no grid'))} / {c.get('no grid', 0)}")
                print(f"    {m:16s} rule off {line[0]:22s} rule on {line[1]}")
    for m, x in res.items():
        hv = [f"{r['tempo']:.1f}->{r['bpm']:.1f}" for r in x["off"] if r["class"] == "half" and r["tempo"] >= 105]
        print(f"  {m}: halved without the rule (reference >= 105 BPM): {len(hv)} {hv[:8]}")


def confidence(models):
    import pulse_confidence as pc
    numbers = ("on", "act", "margin", "parity")
    for m in models:
        x = json.loads((OUT / f"score_{m}.json").read_text())

        def rows_of(rs, steady_only):
            return [{**{k: r[k] for k in numbers}, "spread": None, "tempo": r["class"] != "right"}
                    for r in rs if r["class"] != "no grid" and (r.get("steady", True) or not steady_only)]
        test = rows_of(x["on"], True)
        mixed = rows_of(json.loads((DATA / "pulse_passfail" / m / "result.json").read_text())["mixed"]["on"]["rows"], False)
        print(f"\n==== {m}, rule on")
        pc.auc_table(f"unseen set, steady references (fault: wrong tempo)", test, ["tempo"], numbers)
        pc.auc_table(f"unseen set, every reference", rows_of(x["on"], False), ["tempo"], numbers)
        for nums in (("act",), ("act", "parity")):
            pc.calibrate(f"wrong tempo from {' + '.join(nums)}: fitted on the mixed set, tested on the unseen set",
                         mixed, test, "tempo", nums)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["prep", "score", "table", "confidence"])
    ap.add_argument("models", nargs="*")
    ap.add_argument("--model", action="append", default=[])
    a = ap.parse_args()
    if a.cmd == "prep":
        prep(a)
    elif a.cmd == "score":
        score(a)
    elif a.cmd == "confidence":
        confidence(a.models or a.model)
    else:
        table(a.models or a.model)


if __name__ == "__main__":
    main()
