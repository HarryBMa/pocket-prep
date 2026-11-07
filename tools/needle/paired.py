"""Paired, song-level comparison of two eval.py results: is B really better than A?

Both models were scored on the same songs, so compare them song by song: for each song take
B's SDR minus A's SDR. Songs differ a lot in difficulty (one song is 1 dB for every model,
another 9 dB), and pairing removes that song-to-song spread, so a real difference shows up
with far fewer songs than comparing the two averages would need.

Reports, per stem and for the average over stems:
- mean difference (dB) with a 95 % bootstrap confidence interval (resample songs 10,000 times),
- how many songs got better.
If the interval does not include 0, the difference is unlikely to be chance on this test set.

    python paired.py results/b16_f32_test.json results/b16_f32_stems_test.json
"""
import argparse
import json

import numpy as np

STEMS = ["drums", "bass", "other", "vocals"]


def per_track(path):
    d = json.load(open(path))
    return d["name"], {t: [v["SDR"].get(s, np.nan) for s in STEMS] for t, v in d["per_track"].items()}


def bootstrap_ci(x, rng, n=10_000):
    idx = rng.integers(0, len(x), size=(n, len(x)))
    means = x[idx].mean(1)
    return np.percentile(means, [2.5, 97.5])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a", help="baseline results json")
    ap.add_argument("b", help="new results json")
    a = ap.parse_args()
    name_a, ta = per_track(a.a)
    name_b, tb = per_track(a.b)
    songs = sorted(set(ta) & set(tb))
    A = np.array([ta[s] for s in songs], dtype=float)
    B = np.array([tb[s] for s in songs], dtype=float)
    rng = np.random.default_rng(0)
    print(f"{name_b} minus {name_a}, {len(songs)} songs in both (positive = {name_b} better)")
    cols = STEMS + ["avg"]
    A = np.column_stack([A, np.nanmean(A, 1)])
    B = np.column_stack([B, np.nanmean(B, 1)])
    for i, c in enumerate(cols):
        d = B[:, i] - A[:, i]
        d = d[~np.isnan(d)]                     # silent stems (NaN) are left out
        if len(d) == 0:
            continue
        lo, hi = bootstrap_ci(d, rng)
        sure = "  *" if lo > 0 or hi < 0 else ""
        print(f"  {c:7s} {d.mean():+6.2f} dB  95% CI [{lo:+.2f}, {hi:+.2f}]  better on {(d > 0).sum()}/{len(d)}{sure}")
    print("  * = interval excludes 0")


if __name__ == "__main__":
    main()
