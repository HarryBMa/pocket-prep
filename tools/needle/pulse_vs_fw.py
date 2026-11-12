r"""Pulse + grid vs the DECK001 firmware's beat grid (beat_grid.cpp v2.2.118 as bg_cli.exe), on the Raveform test
set, scored against the human annotations.

Human reference: a straight line through the hand-marked beats (tracks whose beats sit on one line within
15 ms; the rest are left out of the tempo / drift numbers). Per method:
  tempo right       BPM within 2 % of the human BPM (not half / double / 3:2)
  |BPM error|       on tracks with the tempo right
  drift after 60 s  how far the track slides against a perfectly gridded partner in sync: |dBPM|/BPM x 60 s
  phase             grid offset vs the human beats, folded to +- half a beat
  beat F            the method's grid beats vs the human beats, +- 70 ms

    .venv\Scripts\python.exe pulse_vs_fw.py [--model p2_w24]
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from pulse import Pulse, f_measure, predict_track      # noqa: E402
from pulse_grid import fit_grid                        # noqa: E402
from pulse_rave import RAVE, Rave                      # noqa: E402

BG = Path(r"F:\needle\beatgrid\bg_cli.exe")
TMP = Path(r"F:\needle\beatgrid\_tmp_pvf.f32")


def firmware(path):
    x, sr = sf.read(str(path), dtype="float32", always_2d=True)
    x = x[:, :2] if x.shape[1] >= 2 else np.repeat(x, 2, 1)
    np.ascontiguousarray(x).tofile(TMP)
    v = subprocess.run([str(BG), str(TMP), str(sr)], capture_output=True, text=True, timeout=600).stdout.split()
    if len(v) != 6 or float(v[0]) <= 0:
        return None
    bpm, first = float(v[0]), float(v[2])
    return bpm, first, len(x) / sr


def human_line(b):
    k = np.arange(len(b))
    period, phase = np.polyfit(k, b, 1)
    res = b - (phase + k * period)
    return 60 / period, phase, float(np.std(res) * 1000)


def metrics(bpm, phase, grid, b, hb, hphase):
    ratio = bpm / hb
    out = {"tempo_right": abs(ratio - 1) < 0.02, "beat_F": f_measure(b, grid)}
    if out["tempo_right"]:
        p = 60 / hb
        out["bpm_err"] = abs(bpm - hb)
        out["drift_ms_60s"] = abs(bpm - hb) / hb * 60_000
        out["phase_ms"] = abs((phase - hphase + p / 2) % p - p / 2) * 1000
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="p2_w24")
    a = ap.parse_args()
    run = HERE / "runs_pulse" / a.model
    ck = torch.load(run / ("best.pt" if (run / "best.pt").exists() else "last.pt"), map_location="cpu", weights_only=False)
    net = Pulse(ck["width"])
    net.load_state_dict(ck["model"])
    net = net.cuda().eval()
    index = json.loads((RAVE / "index.json").read_text())
    te = Rave("test")
    rows = []
    for t in te.ids:
        b, _ = te.lab[t]
        hb, hphase, jit = human_line(b)
        steady = jit < 15
        mel = np.asarray(te.mels[t], dtype=np.float32)
        with torch.inference_mode():
            y = predict_track(net, mel, "cuda")
        grid, pbpm, _ = fit_grid(y[0], mel.shape[1])
        r = {"id": t, "steady": steady, "human_bpm": hb}
        r["pulse"] = metrics(pbpm, grid[0] if len(grid) else 0.0, grid, b, hb, hphase) if pbpm > 0 else {"tempo_right": False, "beat_F": 0.0}
        fw = firmware(index[t]["file"])
        if fw:
            bpm, first, dur = fw
            g = first + np.arange(0, int((dur - first) * bpm / 60) + 1) * 60 / bpm
            r["firmware"] = metrics(bpm, first, g, b, hb, hphase)
        else:
            r["firmware"] = {"tempo_right": False, "beat_F": 0.0, "failed": True}
        rows.append(r)
    (RAVE / "pulse_vs_fw.json").write_text(json.dumps(rows, indent=1, default=float))
    st = [r for r in rows if r["steady"]]
    print(f"Raveform test set: {len(rows)} tracks ({len(st)} with a steady human grid for the tempo numbers)")
    for k in ("firmware", "pulse"):
        tr = [r[k] for r in st if r[k]["tempo_right"]]
        be = np.array([x["bpm_err"] for x in tr]); dr = np.array([x["drift_ms_60s"] for x in tr])
        ph = np.array([x["phase_ms"] for x in tr])
        print(f"{k:9s} tempo right {len(tr)}/{len(st)} | |BPM err| median {np.median(be):.3f}, >0.05 BPM {np.mean(be > 0.05) * 100:.0f}% | "
              f"drift after 60 s median {np.median(dr):.1f} ms, >20 ms {np.mean(dr > 20) * 100:.0f}% | phase median {np.median(ph):.1f} ms, "
              f">20 ms {np.mean(ph > 20) * 100:.0f}% | beat F (all {len(rows)}) {np.mean([r[k]['beat_F'] for r in rows]):.3f}")


if __name__ == "__main__":
    main()
