r"""Beat-grid lab: the DECK001 firmware's beat-grid analysis (beat_grid.cpp v2.2.118, unchanged, built as
bg_cli.exe) fed three different inputs, scored against a precise reference grid.

Inputs to the firmware analysis, per track:
  mix      the full track (what the deck does today)
  needle   Needle's drum stem (the proposal)
  teacher  the teacher's drum stem (what perfect-ish separation would give)

Reference grid: kick onsets of the teacher drums (< 150 Hz, 1.45-ms hops), a straight line fitted through
them (onset time vs beat number) with outliers dropped: tempo to ~0.001 BPM and phase to ~1 ms on house.

Scores per input: BPM error, half/double/3:2 tempo errors, phase error (ms, folded to +-half a beat),
and drift: how far this track slides against a correctly gridded partner after 60 s of sync
(|BPM error| / BPM x 60 s). The phase error is a constant offset on top (a flam from the first beat).

    .venv\Scripts\python.exe beatgrid_lab.py      -> F:\needle\beatgrid\results.json + summary
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import scipy.signal as ss
import soundfile as sf
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from chunked import separate_chunked, torch_predictor      # noqa: E402
from needle import SOURCES, Needle, NeedleConfig           # noqa: E402

OUT = Path(r"F:\needle\beatgrid")
BG = OUT / "bg_cli.exe"
SETS = {"holdout": Path(r"E:\needle-teacher\holdout_house\jamendo_v2"), "ibiza": Path(r"E:\needle-teacher\ibiza_test_v2")}
CKPT = HERE / "runs" / "b20_jam2_frontier" / "best.pt"
SR = 44100
D = SOURCES.index("drums")


def firmware_grid(x, sr):
    """x (N, 2) float32 -> (bpm, rawBpm, firstBeat s, confidence, integer, estimated) from the firmware code."""
    tmp = OUT / "_tmp.f32"
    np.ascontiguousarray(x, dtype=np.float32).tofile(tmp)
    r = subprocess.run([str(BG), str(tmp), str(sr)], capture_output=True, text=True, timeout=600)
    v = r.stdout.split()
    return dict(zip(["bpm", "raw", "phase", "conf", "integer", "ok"], [float(a) for a in v])) if len(v) == 6 else None


OFFSET_S = 0.005   # calibrated on synthetic kicks (120-132 BPM): the detector fires 5.0 ms before the first sample


def kick_onsets(drums, sr, hop=64):
    """Kick onset times (s) of a drum stem: rise of the < 150 Hz energy envelope, peak-picked."""
    lo = ss.sosfiltfilt(ss.butter(4, 150, fs=sr, output="sos"), drums.mean(1))
    pw = np.convolve(lo ** 2, np.ones(512) / 512, "same")             # 11.6-ms power: smooths the 40-150 Hz cycles
    n = len(pw) // hop
    e = np.log10(pw[: n * hop: hop][:n] + 1e-10)
    rise = np.maximum(0, e[3:] - e[:-3])                                # 3-hop (4.4 ms) rise
    peaks, _ = ss.find_peaks(rise, height=0.5, distance=int(0.2 * sr / hop))   # > 5 dB rise, < 300 BPM
    loud = e[peaks + 3] > np.percentile(e, 99) - 2.5                    # within 25 dB of the loud kicks
    p = peaks[loud]
    # sub-hop timing: where the rise is steepest within the 3 hops before the peak
    return (p + 1.5) * hop / sr + OFFSET_S


def reference_grid(onsets, bpm_hint):
    """Straight-line grid through kick onsets. For the hint tempo and its 1/2, 2, 2/3, 3/2 relatives: a fine
    search (+-2 %, 0.005 BPM) for the tempo whose grid the onsets line up on best, then a line fit through
    the onsets within 15 ms of it. The relative with the best F1 (onsets on the grid x grid beats with an
    onset) wins, so a double-time grid that leaves every other beat empty loses."""
    if len(onsets) < 16:
        return None
    best = None
    for rel in (1, 0.5, 2, 2 / 3, 1.5):
        b0 = bpm_hint * rel
        if not 70 <= b0 <= 190:
            continue
        cand = np.arange(b0 * 0.98, b0 * 1.02, 0.005)
        z = np.exp(2j * np.pi * onsets[None, :] * (cand[:, None] / 60))
        r = np.abs(z.mean(1))
        bpm = cand[np.argmax(r)]
        period = 60 / bpm
        phase = np.angle(z[np.argmax(r)].mean()) / (2 * np.pi) * period
        for _ in range(3):
            k = np.round((onsets - phase) / period)
            keep = np.abs(onsets - (phase + k * period)) < 0.015
            if keep.sum() < 16:
                break
            period, phase = np.polyfit(k[keep], onsets[keep], 1)
        k = np.round((onsets - phase) / period)
        res = onsets - (phase + k * period)
        keep = np.abs(res) < 0.015
        if keep.sum() < 16:
            continue
        precision = keep.mean()
        recall = len(np.unique(k[keep])) / (k.max() - k.min() + 1)
        f1 = 2 * precision * recall / (precision + recall)
        if best is None or f1 > best["f1"]:
            best = {"bpm": 60 / period, "phase": float(phase % period), "n": int(keep.sum()), "of": int(len(onsets)),
                    "f1": float(f1), "recall": float(recall), "jitter_ms": float(np.std(res[keep]) * 1000)}
    return best


def score(grid, ref):
    if not grid or not grid["ok"] or grid["bpm"] <= 0:
        return {"failed": True}
    ratio = grid["bpm"] / ref["bpm"]
    rel = min((1, 0.5, 2, 2 / 3, 1.5), key=lambda r: abs(np.log(ratio / r)))
    if abs(np.log(ratio / rel)) > 0.02:                                # not a simple relative: just wrong
        return {"failed": False, "relation": "wrong"}
    name = {1: "same", 0.5: "half", 2: "double", 2 / 3: "2:3", 1.5: "3:2"}[rel]
    if rel != 1:
        return {"failed": False, "relation": name}
    period = 60 / ref["bpm"]
    err = (grid["phase"] - ref["phase"] + period / 2) % period - period / 2   # grid offset, folded to +-half a beat
    bpm_err = grid["bpm"] - ref["bpm"]
    return {"failed": False, "relation": name, "bpm_err": float(bpm_err), "phase_ms": float(err * 1000),
            "drift_ms_60s": float(abs(bpm_err) / ref["bpm"] * 60 * 1000)}


def main():
    OUT.mkdir(exist_ok=True)
    res = json.loads((OUT / "results.json").read_text()) if (OUT / "results.json").exists() else {}
    torch.set_num_threads(8)
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    cfg = NeedleConfig(**ck["config"])
    model = Needle(cfg)
    model.load_state_dict(ck["model"])
    model.eval()
    pred = torch_predictor(model, 16)
    for set_name, root in SETS.items():
        for d in sorted(p.parent for p in root.glob("*/done")):
            key = f"{set_name}/{d.name}"
            if key in res and res[key].get("fw_needle") and res[key].get("ref_v") == 3:
                continue
            if key in res and res[key].get("fw_needle"):                      # firmware grids done: redo the reference only
                row = res[key]
                teacher, tsr = sf.read(str(d / "drums.flac"), dtype="float32", always_2d=True)
                hint = row["fw_teacher"]["bpm"] if row["fw_teacher"] and row["fw_teacher"]["bpm"] > 0 else 124.0
                row["ref"] = reference_grid(kick_onsets(teacher, tsr), hint)
                row["ref_v"] = 3
                for k in ("fw_mix", "fw_needle", "fw_teacher"):
                    row.pop("s_" + k[3:], None)
                    if row["ref"]:
                        row["s_" + k[3:]] = score(row[k], row["ref"])
                (OUT / "results.json").write_text(json.dumps(res, indent=1))
                continue
            src = Path((d / "source.txt").read_text(encoding="utf-8").splitlines()[0])
            try:
                mix, sr = sf.read(str(src), dtype="float32", always_2d=True)
            except Exception as ex:                                         # noqa: BLE001
                res[key] = {"error": f"source: {ex}"}
                continue
            mix = mix[:, :2] if mix.shape[1] >= 2 else np.repeat(mix, 2, 1)
            teacher, tsr = sf.read(str(d / "drums.flac"), dtype="float32", always_2d=True)
            row = {"source": str(src)}
            row["fw_mix"] = firmware_grid(mix, sr)
            row["fw_teacher"] = firmware_grid(teacher, tsr)
            if sr != SR:
                import torchaudio
                mix44 = torchaudio.functional.resample(torch.from_numpy(mix.T.copy()), sr, SR).numpy()
            else:
                mix44 = mix.T.copy()
            needle = separate_chunked(pred, torch.from_numpy(mix44), cfg, 8, batch=16).numpy()[D].T
            row["fw_needle"] = firmware_grid(needle, SR)
            hint = row["fw_teacher"]["bpm"] if row["fw_teacher"] and row["fw_teacher"]["bpm"] > 0 else 124.0
            ref = reference_grid(kick_onsets(teacher, tsr), hint)
            row["ref"] = ref
            row["ref_v"] = 3
            if ref:
                for k in ("fw_mix", "fw_needle", "fw_teacher"):
                    row["s_" + k[3:]] = score(row[k], ref)
            res[key] = row
            (OUT / "results.json").write_text(json.dumps(res, indent=1))
            print(key[:60], json.dumps({k: row.get("s_" + k) for k in ("mix", "needle", "teacher")}), flush=True)
    summarize(res)


def summarize(res):
    rows = [r for r in res.values() if r.get("ref") and r["ref"]["f1"] > 0.6 and r["ref"]["jitter_ms"] < 5]
    print(f"\n{len(rows)} tracks with a clean reference grid (kicks on one straight line: F1 > 0.6, jitter < 5 ms)")
    for k in ("mix", "needle", "teacher"):
        s = [r["s_" + k] for r in rows if "s_" + k in r]
        ok = [x for x in s if not x["failed"]]
        same = [x for x in ok if x["relation"] == "same"]
        be = np.array([abs(x["bpm_err"]) for x in same])
        ph = np.array([abs(x["phase_ms"]) for x in same])
        dr = np.array([x["drift_ms_60s"] for x in same])
        if not same:
            print(k, "no track with the right tempo"); continue
        print(f"{k:8s} tempo right {len(same)}/{len(s)} (half/double/3:2 {len(ok) - len(same)}, failed {len(s) - len(ok)}) | "
              f"|BPM err| median {np.median(be):.3f}, >0.02 {np.mean(be > 0.02) * 100:.0f}% | |phase| median {np.median(ph):.1f} ms, "
              f">10 ms {np.mean(ph > 10) * 100:.0f}% | drift after 60 s median {np.median(dr):.1f} ms, >20 ms {np.mean(dr > 20) * 100:.0f}%")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "summary":
        summarize(json.loads((OUT / "results.json").read_text()))
    else:
        main()
