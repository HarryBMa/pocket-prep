"""Drum dropouts and leakage on the 50 hold-out house tracks: plain vs frontier, against the teacher drums.
dropout = share of 100-ms windows (where the teacher's drums are audible) in which the model's drums are >20 dB quieter.
leak = energy of the drum estimate's error that projects onto the teacher's non-drum part, dB below the teacher drums."""
import json, sys, time
from pathlib import Path
import numpy as np, torch
from chunked import separate_chunked, torch_predictor
from data import read_stems
from needle import SOURCES, Needle, NeedleConfig
torch.set_num_threads(10)
cks = {n: f"runs/{n}/best.pt" for n in ["b20_jam_plain", "b20_jam_frontier"]}
models = {}
for n, p in cks.items():
    ck = torch.load(p, map_location="cpu", weights_only=False); cfg = NeedleConfig(**ck["config"]); m = Needle(cfg); m.load_state_dict(ck["model"]); m.eval()
    models[n] = (torch_predictor(m, 16), cfg)
D = SOURCES.index("drums")
out = {}
dirs = sorted(p.parent for p in Path(r"E:\needle-teacher\holdout_house\jamendo_v2").glob("*/done"))
for i, d in enumerate(dirs):
    st = read_stems(d, "flac", dtype="float32")          # (4, 2, N)
    mix = st.sum(0); t = st[D]; other = mix - t
    w = lambda x: np.convolve(x.mean(0) ** 2, np.ones(4410) / 4410, "same")[::2205]
    tw = w(t); live = tw > tw.max() * 1e-3
    row = {}
    for n, (pred, cfg) in models.items():
        e = separate_chunked(pred, torch.from_numpy(mix), cfg, 8, batch=16).numpy()[D]
        r = 10 * np.log10((w(e)[live] + 1e-12) / (tw[live] + 1e-12))
        err = e - t; a = (err * other).sum() / ((other ** 2).sum() + 1e-12)
        row[n] = {"dropout20": float(np.mean(r < -20)), "dropout10": float(np.mean(r < -10)),
                  "leak_db": float(10 * np.log10(((a * other) ** 2).sum() / ((t ** 2).sum() + 1e-12) + 1e-12)),
                  "sdr": float(10 * np.log10((t ** 2).sum() / ((err ** 2).sum() + 1e-12)))}
    out[d.name] = row
    print(i, d.name[:40], json.dumps({k: {kk: round(vv, 3) for kk, vv in v.items()} for k, v in row.items()}), flush=True)
    Path(r"F:\needle\stem_loops\dropout_scan.json").write_text(json.dumps(out, indent=1))
