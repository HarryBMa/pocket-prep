r"""hit_scan.py's per-hit drum checks on any reference set, e.g. real stems instead of the teacher's.

    .venv\Scripts\python.exe hit_scan2.py --refs F:\needle\data\stems\moisesdb_test --out results\hits_moisesdb_test.json ^
        runs\b20_jam3_frontier_env\best.pt runs\n2_a\best.pt
    .venv\Scripts\python.exe hit_scan2.py --musdb F:\needle\data\musdb18hq --out results\hits_musdb_test.json <checkpoints>

Same measures as hit_scan.py, with the drum stem of the reference set as the truth:
  level   model/reference energy in the 50 ms after each drum onset, dB (very negative = a dropped hit)
  attack  the same for 6-16 kHz in the first 20 ms (hat and cymbal attacks)
The mixture is the sum of the reference stems. Tracks without drum hits are skipped. Results are kept per track and
model in --out, so a second call with one more checkpoint only separates with the new one. Runs on the CPU.

    .venv\Scripts\python.exe hit_scan2.py --compare results\hits_moisesdb_test.json b20_jam3_frontier_env n2_a
prints where the second model loses attacks that the first keeps (pooled, at the same hits, per track).
"""
import argparse
import json
from pathlib import Path

import numpy as np
import scipy.signal as ss
import torch

from chunked import separate_chunked, torch_predictor
from data import musdb_split, read_stems
from hit_scan import D, SR, onsets, summarize, window_db
from needle import Needle, NeedleConfig


def scan(a):
    models = []
    for p in a.checkpoints:
        ck = torch.load(p, map_location="cpu", weights_only=False)
        cfg = NeedleConfig(**ck["config"])
        m = Needle(cfg)
        m.load_state_dict(ck["model"])
        m.eval()
        models.append((Path(p).parent.name, torch_predictor(m, 16), cfg))
    if a.refs:
        tracks = [(d.name, d, "flac") for d in sorted(p.parent for p in Path(a.refs).glob("*/done"))]
    else:
        tracks = [(n, Path(a.musdb) / "test" / n, "wav") for n in musdb_split(a.musdb)["test"]]
    out = Path(a.out)
    res = json.loads(out.read_text()) if out.exists() else {}
    hf = ss.butter(4, [6000, 16000], btype="band", fs=SR, output="sos")
    for i, (name, d, ext) in enumerate(tracks):
        if name in res and all(n in res[name] for n, _, _ in models):
            continue
        st = read_stems(d, ext, dtype="float32")
        mix, tm = st.sum(0), st[D].mean(0)
        if np.abs(tm).max() < 1e-4:
            print(i, name[:40], "no drums, skipped", flush=True)
            continue
        hits = onsets(tm)
        hits = hits[(hits > 0) & (hits < len(tm) - SR // 10)]
        if not len(hits):
            continue
        t_lvl, t_att = window_db(tm, hits, int(0.05 * SR)), window_db(ss.sosfilt(hf, tm), hits, int(0.02 * SR))
        hf_live = t_att > t_att.max() - 30                                   # hits that have real high-frequency content
        row = res.get(name, {})
        for mname, pred, cfg in models:
            if mname in row:
                continue
            e = separate_chunked(pred, torch.from_numpy(mix), cfg, 8, batch=16).numpy()[D].mean(0)
            lvl = window_db(e, hits, int(0.05 * SR)) - t_lvl
            att = window_db(ss.sosfilt(hf, e), hits, int(0.02 * SR)) - t_att
            pos = (hits // cfg.hop) % (cfg.frames - 8)
            row[mname] = {"hits": int(len(hits)), "level": lvl.round(2).tolist(),
                          "attack": np.where(hf_live, att, np.nan).round(2).tolist(),
                          "xfade": (pos < 8).astype(int).tolist(), "step": int((cfg.frames - 8) * cfg.hop)}
        res[name] = row
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(res))
        print(i, name[:40], len(hits), "hits", flush=True)
    summarize(res, [n for n, _, _ in models])


def compare(path, A, B):
    """Where does model B lose the 6-16 kHz attack of hits that model A keeps?"""
    res = json.loads(Path(path).read_text())
    rows = [(t, r[A], r[B]) for t, r in res.items() if A in r and B in r]
    cat = lambda key, i: np.concatenate([np.array(row[i][key], dtype=float) for row in rows])
    att, lvl = {A: cat("attack", 1), B: cat("attack", 2)}, {A: cat("level", 1), B: cat("level", 2)}
    live = ~np.isnan(att[A]) & ~np.isnan(att[B])
    print(f"{len(rows)} tracks, {live.sum()} hits with high-frequency content of {len(live)}")
    for m in (A, B):
        x, v = att[m][live], lvl[m]
        print(f"{m:>24}: attack median {np.median(x):+.2f} dB, 5th pct {np.percentile(x, 5):+.2f}, 1st pct {np.percentile(x, 1):+.2f} | "
              f">3 dB down {np.mean(x < -3) * 100:.1f}% | >6 dB down {np.mean(x < -6) * 100:.1f}% | >10 dB down {np.mean(x < -10) * 100:.1f}% | "
              f"level: median {np.median(v):+.2f} dB, >10 dB down {np.mean(v < -10) * 100:.1f}%")
    bad = live & (att[B] < -6)
    if bad.any():
        print(f"hits where {B} loses > 6 dB of attack: {bad.sum()}; {A} at the same hits: median {np.median(att[A][bad]):+.2f} dB, "
              f"also > 6 dB down on {np.mean(att[A][bad] < -6) * 100:.0f}%; {B}'s full-band level there: median {np.median(lvl[B][bad]):+.2f} dB")
    per = []
    for t, ra, rb in rows:
        x, y = np.array(ra["attack"], dtype=float), np.array(rb["attack"], dtype=float)
        ok = ~np.isnan(x) & ~np.isnan(y)
        if ok.sum() >= 20:
            per.append((np.mean(y[ok] < -6) * 100, np.mean(x[ok] < -6) * 100, int(ok.sum()), int((y[ok] < -6).sum()), t))
    per.sort(reverse=True)
    tot = max(sum(p[3] for p in per), 1)
    print(f"per track ({len(per)} tracks with >= 20 such hits): {B} dulls more hits than {A} on {sum(p[0] > p[1] for p in per)}; "
          f"median per track {np.median([p[0] for p in per]):.1f}% vs {np.median([p[1] for p in per]):.1f}%; "
          f"the worst 5 tracks hold {sum(p[3] for p in per[:5]) / tot * 100:.0f}% of {B}'s dulled hits")
    for p in per[:6]:
        print(f"  {p[0]:5.1f}% ({B}) vs {p[1]:5.1f}% ({A})  {p[2]:5d} hits  {p[4][:60]}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoints", nargs="*")
    ap.add_argument("--refs", help="folder of real stems (import_stems.py layout)")
    ap.add_argument("--musdb", help="MUSDB18-HQ root: its test tracks")
    ap.add_argument("--out", help="json with the per-hit results")
    ap.add_argument("--compare", nargs=3, metavar=("JSON", "A", "B"), help="compare two models of a finished scan")
    a = ap.parse_args()
    if a.compare:
        compare(*a.compare)
    else:
        assert (a.refs or a.musdb) and a.out and a.checkpoints, "give --refs or --musdb, --out and checkpoints"
        torch.set_num_threads(8)
        scan(a)
