r"""Per-hit drum checks on the hold-out house set, against the teacher's drums.

For every drum hit (onsets of the teacher's drum stem) and every model:
  level   model/teacher energy in the 50 ms after the onset, dB (0 = same; very negative = a dropped hit)
  attack  the same for 6-16 kHz in the first 20 ms (cymbal and hat attacks: a smeared crash loses this)
  phase   where the hit falls in the device's chunk grid: 0 = chunk centre, 1 = inside the 8-frame crossfade
Summaries per model: mean level, hit-to-hit spread (std of level: "inconsistent snare"), share of hits
>10 dB down, mean attack, and level/spread for hits inside vs outside the crossfades.

    .venv\Scripts\python.exe hit_scan.py runs\b20_jam_plain\best.pt runs\b20_jam_frontier\best.pt ...
writes F:\needle\stem_loops\hit_scan.json (per track) and prints the summary.
"""
import json
import sys
from pathlib import Path

import numpy as np
import scipy.signal as ss
import torch

from chunked import separate_chunked, torch_predictor
from data import read_stems
from needle import SOURCES, Needle, NeedleConfig

SR = 44100
D = SOURCES.index("drums")
HOLD = Path(r"E:\needle-teacher\holdout_house\jamendo_v2")
OUT = Path(r"F:\needle\stem_loops\hit_scan.json")
torch.set_num_threads(8)


def onsets(x, hop=256):
    """Onset times (samples) of a mono drum signal: peaks of the half-wave rectified log-energy flux."""
    n = len(x) // hop
    e = np.log10((x[: n * hop].reshape(n, hop) ** 2).mean(1) + 1e-9)
    flux = np.maximum(0, np.diff(e, prepend=e[0]))
    thr = np.convolve(flux, np.ones(43) / 43, "same") + 0.3                 # local mean + 3 dB
    peaks, _ = ss.find_peaks(flux, height=thr, distance=int(0.08 * SR / hop))  # at most one hit per 80 ms
    loud = e[peaks] > e.max() - 4                                           # within 40 dB of the loudest frame
    return peaks[loud] * hop


def window_db(x, starts, length):
    return np.array([10 * np.log10((x[s:s + length] ** 2).mean() + 1e-12) for s in starts])


def main():
    models = []
    for p in sys.argv[1:]:
        ck = torch.load(p, map_location="cpu", weights_only=False)
        cfg = NeedleConfig(**ck["config"])
        m = Needle(cfg)
        m.load_state_dict(ck["model"])
        m.eval()
        models.append((Path(p).parent.name, torch_predictor(m, 16), cfg))
    hf = ss.butter(4, [6000, 16000], btype="band", fs=SR, output="sos")
    res = json.loads(OUT.read_text()) if OUT.exists() else {}
    dirs = sorted(p.parent for p in HOLD.glob("*/done"))
    for i, d in enumerate(dirs):
        if d.name in res and all(n in res[d.name] for n, _, _ in models):
            continue
        st = read_stems(d, "flac", dtype="float32")
        mix, t = st.sum(0), st[D]
        tm = t.mean(0)
        hits = onsets(tm)
        hits = hits[(hits > 0) & (hits < len(tm) - SR // 10)]
        t_lvl, t_att = window_db(tm, hits, int(0.05 * SR)), window_db(ss.sosfilt(hf, tm), hits, int(0.02 * SR))
        row = res.get(d.name, {})
        for name, pred, cfg in models:
            if name in row:
                continue
            e = separate_chunked(pred, torch.from_numpy(mix), cfg, 8, batch=16).numpy()[D].mean(0)
            lvl = window_db(e, hits, int(0.05 * SR)) - t_lvl
            att = window_db(ss.sosfilt(hf, e), hits, int(0.02 * SR)) - t_att
            step = (cfg.frames - 8) * cfg.hop
            pos = (hits // cfg.hop) % (cfg.frames - 8)                       # frame of the hit within its chunk step
            xfade = (pos < 8).astype(int)                                    # first 8 frames of a step = the crossfade
            hf_live = t_att > t_att.max() - 30                               # hits that have real high-frequency content
            row[name] = {"hits": int(len(hits)), "level": lvl.round(2).tolist(), "attack": np.where(hf_live, att, np.nan).round(2).tolist(),
                         "xfade": xfade.tolist(), "step": int(step)}
        res[d.name] = row
        OUT.write_text(json.dumps(res))
        print(i, d.name[:40], len(hits), "hits", flush=True)
    summarize(res, [n for n, _, _ in models])


def summarize(res, names):
    for n in names:
        lv = np.concatenate([np.array(r[n]["level"]) for r in res.values() if n in r])
        at = np.concatenate([np.array(r[n]["attack"], dtype=float) for r in res.values() if n in r])
        xf = np.concatenate([np.array(r[n]["xfade"]) for r in res.values() if n in r])
        spread = np.median([np.std(np.clip(r[n]["level"], -30, 10)) for r in res.values() if n in r and len(r[n]["level"]) > 8])
        print(f"{n}: {len(lv)} hits | level {np.median(lv):+.2f} dB median | >10 dB down {np.mean(lv < -10) * 100:.1f}% | "
              f"spread {spread:.2f} dB | attack 6-16k {np.nanmedian(at):+.2f} dB, >6 dB down {np.nanmean(at < -6) * 100:.1f}% | "
              f">10 dB down in crossfade {np.mean(lv[xf == 1] < -10) * 100:.1f}% vs elsewhere {np.mean(lv[xf == 0] < -10) * 100:.1f}%")


if __name__ == "__main__":
    main()
