r"""The firmware's C++ Pulse (pulse_cli.exe, built from pulse_beat.cpp + exported weights) vs the Python Pulse
and vs the v2.2.118 firmware analyser, on the Raveform test set, scored against the human annotations.

Every method gives what the deck stores: a BPM and one anchor (firstBeatSeconds). Pulse's anchor is the bar
line (beat 1); the firmware's is a beat (it makes no downbeat claim).
  tempo right    within 2 % of the human BPM
  phase          anchor vs the human beat grid, folded to +- half a beat (ms)
  bar line right the anchor lands on a human downbeat (+- 70 ms), checked against the human downbeats
                 bar by bar: share of human downbeats that fall on the anchor's bar grid
  C++ vs Python  BPM difference and anchor difference per track (the port must not change the answer)

    .venv\Scripts\python.exe pulse_cvalidate.py --model p3_w24_l6
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
from pulse import Pulse, predict_track            # noqa: E402
from pulse_grid import fit_grid                   # noqa: E402
from pulse_rave import RAVE, Rave, bar_fit        # noqa: E402
from pulse_vs_fw import firmware, human_line      # noqa: E402

CLI = Path(r"F:\needle\pulse_c\pulse_cli.exe")
TMP = Path(r"F:\needle\pulse_c\_tmp.f32")


def py_anchor(net, mel):
    """Python reference of what pulse_beat.cpp returns: grid + majority bar position, anchor moved to the start."""
    with torch.inference_mode():
        y = predict_track(net, mel, "cuda")
    grid, bpm, steady = fit_grid(y[0], mel.shape[1])
    if not steady or len(grid) < 8:
        return bpm, None
    d = bar_fit(grid, y[1], bars=32)
    idx = np.searchsorted(grid, d)
    votes = np.bincount(idx % 4, minlength=4)
    anchor = grid[int(np.argmax(votes))]
    bar = 4 * 60 / bpm
    return bpm, anchor - np.floor(anchor / bar) * bar


def score(bpm, anchor, b, d, hb, hphase):
    if not bpm or anchor is None:
        return {"tempo_right": False}
    p = 60 / hb
    out = {"tempo_right": abs(bpm / hb - 1) < 0.02}
    if out["tempo_right"]:
        out["phase_ms"] = abs((anchor - hphase + p / 2) % p - p / 2) * 1000
        bar = 4 * 60 / bpm
        r = (d - anchor + bar / 2) % bar - bar / 2                  # each human downbeat vs the anchor's bar grid
        out["bar_share"] = float(np.mean(np.abs(r) < 0.07))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="p3_w24_l6")
    a = ap.parse_args()
    run = HERE / "runs_pulse" / a.model
    ck = torch.load(run / ("best.pt" if (run / "best.pt").exists() else "last.pt"), map_location="cpu", weights_only=False)
    net = Pulse(ck["width"], ck.get("levels", 5))
    net.load_state_dict(ck["model"])
    net = net.cuda().eval()
    index = json.loads((RAVE / "index.json").read_text())
    te = Rave("test")
    rows, times, audio_s = [], [], 0.0
    for t in te.ids:
        b, d = te.lab[t]
        hb, hphase, jit = human_line(b)
        x, sr = sf.read(index[t]["file"], dtype="float32", always_2d=True)
        x = x[:, :2] if x.shape[1] >= 2 else np.repeat(x, 2, 1)
        np.ascontiguousarray(x).tofile(TMP)
        v = subprocess.run([str(CLI), str(TMP), str(sr)], capture_output=True, text=True, timeout=600).stdout.split()
        ok, steady, cbpm, canchor, ms = int(v[0]), int(v[1]), float(v[2]), float(v[3]), float(v[7])
        times.append(ms)
        audio_s += len(x) / sr
        pbpm, panchor = py_anchor(net, np.asarray(te.mels[t], dtype=np.float32))
        fw = firmware(index[t]["file"])
        r = {"id": t, "steady_human": jit < 15,
             "cpp": score(cbpm if ok else 0, canchor if ok else None, b, d, hb, hphase),
             "python": score(pbpm, panchor, b, d, hb, hphase),
             "firmware": score(fw[0], fw[1], b, d, hb, hphase) if fw else {"tempo_right": False},
             "cpp_vs_py_bpm": abs(cbpm - pbpm) if ok and pbpm else None,
             "cpp_vs_py_anchor_ms": (abs(((canchor - panchor) + 30 / cbpm) % (60 / cbpm) - 30 / cbpm) * 1000)
             if ok and panchor is not None else None,
             "cpp_vs_py_same_bar": bool(ok and panchor is not None and abs(((canchor - panchor) + 120 / cbpm) % (240 / cbpm) - 120 / cbpm) < 0.035)}
        rows.append(r)
    (RAVE / "pulse_cvalidate.json").write_text(json.dumps(rows, indent=1, default=float))
    st = [r for r in rows if r["steady_human"]]
    print(f"Raveform test set: {len(rows)} tracks ({len(st)} with a steady human grid); model {a.model}")
    for k in ("firmware", "python", "cpp"):
        tr = [r[k] for r in st if r[k]["tempo_right"]]
        ph = np.array([x["phase_ms"] for x in tr])
        bs = np.array([x["bar_share"] for x in tr])
        print(f"{k:8s} tempo right {len(tr)}/{len(st)} | phase median {np.median(ph):.1f} ms, >20 ms {np.mean(ph > 20) * 100:.0f}% | "
              f"bar line on human downbeats: mean {bs.mean():.3f}, tracks >90 % {np.mean(bs > 0.9) * 100:.0f}%")
    db = np.array([r["cpp_vs_py_bpm"] for r in rows if r["cpp_vs_py_bpm"] is not None])
    da = np.array([r["cpp_vs_py_anchor_ms"] for r in rows if r["cpp_vs_py_anchor_ms"] is not None])
    sb = np.mean([r["cpp_vs_py_same_bar"] for r in rows])
    print(f"C++ vs Python: |BPM diff| median {np.median(db):.4f}, max {db.max():.4f}; anchor diff median {np.median(da):.2f} ms, "
          f"max {da.max():.1f} ms; same bar position {sb * 100:.0f}%")
    print(f"C++ speed on this PC (one thread): {np.sum(times) / 1000:.1f} s for {audio_s / 60:.0f} min of audio = "
          f"{np.sum(times) / audio_s:.1f} ms per second of audio, median {np.median(times) / 1000:.2f} s per track")


if __name__ == "__main__":
    main()
