r"""Pulse C++ speed vs accuracy on the Raveform test set: threads and analysis windows (pulse_cli.exe args).
For each config: tempo right, phase, bar line on the human downbeats (pulse_cvalidate.score), time per track.

    .venv\Scripts\python.exe pulse_speed_eval.py
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from pulse_cvalidate import score            # noqa: E402
from pulse_rave import RAVE, Rave            # noqa: E402
from pulse_vs_fw import human_line           # noqa: E402

CLI = Path(r"F:\needle\pulse_c\pulse_cli.exe")
TMP = Path(r"F:\needle\pulse_c\_tmp_speed.f32")
CONFIGS = [("1 thread, whole track", 1, 0), ("3 threads, whole track", 3, 0), ("3 threads, 3 x 60 s", 3, 180),
           ("3 threads, 3 x 40 s", 3, 120), ("3 threads, 3 x 30 s", 3, 90), ("1 thread, 3 x 40 s", 1, 120)]


def main():
    index = json.loads((RAVE / "index.json").read_text())
    te = Rave("test")
    res = {c[0]: [] for c in CONFIGS}
    audio = 0.0
    for t in te.ids:
        b, d = te.lab[t]
        hb, hphase, jit = human_line(b)
        x, sr = sf.read(index[t]["file"], dtype="float32", always_2d=True)
        x = x[:, :2] if x.shape[1] >= 2 else np.repeat(x, 2, 1)
        np.ascontiguousarray(x).tofile(TMP)
        audio += len(x) / sr
        for name, th, mx in CONFIGS:
            v = subprocess.run([str(CLI), str(TMP), str(sr), str(th), str(mx)], capture_output=True, text=True,
                               timeout=600).stdout.split()
            ok, bpm, anchor, ms = int(v[0]) and int(v[1]), float(v[2]), float(v[3]), float(v[7])
            s = score(bpm if ok else 0, anchor if ok else None, b, d, hb, hphase)
            s.update(ms=ms, steady_human=jit < 15, seconds=len(x) / sr)
            res[name].append(s)
    print(f"Raveform test set, {len(te.ids)} tracks, {audio / 60:.0f} min of audio; PC (Ryzen) times")
    for name, rows in res.items():
        st = [r for r in rows if r["steady_human"]]
        tr = [r for r in st if r["tempo_right"]]
        ph = np.array([r["phase_ms"] for r in tr])
        bs = np.array([r["bar_share"] for r in tr])
        ms = np.array([r["ms"] for r in rows])
        per6 = np.sum(ms) / audio * 390 / 1000                      # a 6.5-min track
        print(f"{name:24s} tempo right {len(tr)}/{len(st)} | phase median {np.median(ph):.1f} ms, >20 ms {np.mean(ph > 20) * 100:.0f}% | "
              f"bar line {bs.mean():.3f}, >90 % {np.mean(bs > 0.9) * 100:.0f}% | {per6:.2f} s per 6.5-min track")
    (RAVE / "pulse_speed_eval.json").write_text(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
