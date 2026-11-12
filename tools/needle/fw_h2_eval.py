r"""End-to-end check of firmware patch 2.2.118h2: the patched beat_grid.cpp (Pulse first, legacy fallback),
built as bg_cli_h2.exe, vs the unpatched v2.2.118 analyser (bg_cli.exe), on the Raveform test set against the
human annotations. Same scoring as pulse_cvalidate.py (tempo right, phase, bar line on human downbeats).

    .venv\Scripts\python.exe fw_h2_eval.py
"""
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from pulse_cvalidate import score                  # noqa: E402
from pulse_rave import RAVE, Rave                  # noqa: E402
from pulse_vs_fw import human_line                 # noqa: E402

BINS = {"v2.2.118": Path(r"F:\needle\beatgrid\bg_cli.exe"), "v2.2.118h2": Path(r"F:\needle\beatgrid\bg_cli_h2.exe")}
TMP = Path(r"F:\needle\beatgrid\_tmp_h2.f32")


def main():
    index = json.loads((RAVE / "index.json").read_text())
    te = Rave("test")
    rows, secs, audio = [], {k: 0.0 for k in BINS}, 0.0
    for t in te.ids:
        b, d = te.lab[t]
        hb, hphase, jit = human_line(b)
        x, sr = sf.read(index[t]["file"], dtype="float32", always_2d=True)
        x = x[:, :2] if x.shape[1] >= 2 else np.repeat(x, 2, 1)
        np.ascontiguousarray(x).tofile(TMP)
        audio += len(x) / sr
        row = {"id": t, "steady_human": jit < 15}
        for k, exe in BINS.items():
            t0 = time.time()
            v = subprocess.run([str(exe), str(TMP), str(sr)], capture_output=True, text=True, timeout=600).stdout.split()
            secs[k] += time.time() - t0
            ok = len(v) == 6 and float(v[0]) > 0
            row[k] = score(float(v[0]) if ok else 0, float(v[2]) if ok else None, b, d, hb, hphase)
            row[k]["conf"] = float(v[3]) if ok else 0.0
        rows.append(row)
    (RAVE / "fw_h2_eval.json").write_text(json.dumps(rows, indent=1, default=float))
    st = [r for r in rows if r["steady_human"]]
    print(f"Raveform test set: {len(rows)} tracks ({len(st)} with a steady human grid)")
    for k in BINS:
        tr = [r[k] for r in st if r[k]["tempo_right"]]
        ph = np.array([x["phase_ms"] for x in tr])
        bs = np.array([x["bar_share"] for x in tr])
        rel = np.mean([r[k]["conf"] >= 0.5 for r in rows]) * 100
        print(f"{k:11s} tempo right {len(tr)}/{len(st)} | phase median {np.median(ph):.1f} ms, >20 ms {np.mean(ph > 20) * 100:.0f}% | "
              f"bar line on human downbeats: mean {bs.mean():.3f}, tracks >90 % {np.mean(bs > 0.9) * 100:.0f}% | "
              f"'reliable' (conf >= 0.5) {rel:.0f}% | {secs[k] / audio * 1000:.1f} ms per s of audio (incl. file load)")


if __name__ == "__main__":
    main()
