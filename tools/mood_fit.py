#!/usr/bin/env python3
"""Fit the mood prototypes (src/mood.rs MOODS) on measured tracks, and test them honestly.

Input: the JSON lines printed by the `mood_calibration` test
  POCKET_PREP_MOOD_LIST=<tsv of "mood id<TAB>path"> cargo test --release mood_calibration -- --ignored --nocapture > run.jsonl
Each line has "want" (the mood you labelled the track with), "file" and "features" (nine numbers).

Usage:
  python tools/mood_fit.py run1.jsonl run2.jsonl [--keep keep.txt] [--folds 5] [--weights] [--rust]

  --keep FILE   only use files whose name is listed in FILE (one per line): the tracks you approved
  --without N   leave the number N (e.g. lufs) out of the distance: use it when the sources differ in mastering
  --weights     also try changing the feature weights, keeping a change only if it helps in the cross-validation
  --rust        print the fitted MOODS lines to paste into src/mood.rs

What it does (standard library only):
  1. typical value of a number = the median over the mood's tracks; spread = IQR / 1.35 (the standard deviation a
     normal distribution with that IQR would have), never below a floor so one tight cluster can't make it too strict.
  2. the same distance and confidence as `rank()` in src/mood.rs (tau 0.35).
  3. stratified k-fold cross-validation: every mood is split evenly over the folds; fit on k-1 folds, test on the one
     left out, k times. The accuracy is only ever counted on tracks the values were not fitted on.
"""
import argparse, json, math, random, re, statistics, sys, collections
from pathlib import Path

RS = Path(__file__).resolve().parent.parent / "src" / "mood.rs"
TAU = 0.35
FLOOR = {"bpm": 2.0, "lufs": 1.0, "bass": 0.03, "bright": 0.03, "drums": 0.03, "melody": 0.03, "vocals": 0.03, "breaks": 0.03, "minor": 0.3}


def read_mood_rs():
    """FEATURES (name, weight) and MOODS (id, [(typical, spread) x 9]) as written in src/mood.rs."""
    src = RS.read_text(encoding="utf-8")
    feats = [(n, float(w)) for n, w in re.findall(r'\("(\w+)",\s*([\d.]+),\s*"[^"]*"\)', src.split("pub const N")[0])]
    moods = []
    for m in re.finditer(r'Mood \{ id: "(\w+)".*?proto: \[(.*?)\] \},', src, re.S):
        nums = re.findall(r"\((-?[\d.]+),\s*(-?[\d.]+)\)", m.group(2))
        moods.append((m.group(1), [(float(a), float(b)) for a, b in nums]))
    assert len(feats) == 9 and all(len(p) == 9 for _, p in moods), "could not read src/mood.rs"
    return feats, moods


def distance(x, proto, w):
    s = ws = 0.0
    for j, v in enumerate(x):
        if v is not None:
            z = (v - proto[j][0]) / proto[j][1]
            s += w[j] * z * z
            ws += w[j]
    return math.sqrt(s / ws) if ws else math.inf


def predict(x, protos, w):
    d = [distance(x, p, w) for p in protos]
    return min(range(len(d)), key=lambda i: (d[i], i))


def confidence(x, protos, w):
    d = [distance(x, p, w) for p in protos]
    like = [math.exp(-v * v / (2 * TAU * TAU)) for v in d]
    return max(like) / max(sum(like), 1e-300)


def fit(data, names, builtin, min_tracks=5):
    """Prototype per mood from (mood index, features); a mood with fewer than min_tracks keeps the built-in values."""
    protos = []
    for m, base in enumerate(builtin):
        rows = [x for k, x in data if k == m]
        p = []
        for j, name in enumerate(names):
            v = sorted(r[j] for r in rows if r[j] is not None)
            if len(rows) < min_tracks or len(v) < min_tracks:
                p.append(base[j])
                continue
            q = statistics.quantiles(v, n=4, method="inclusive")
            p.append((statistics.median(v), max((q[2] - q[0]) / 1.35, FLOOR[name])))
        protos.append(p)
    return protos


def folds_of(data, k, seed=1):
    by = collections.defaultdict(list)
    for i, (m, _) in enumerate(data):
        by[m].append(i)
    rng, fold = random.Random(seed), [0] * len(data)
    for idx in by.values():
        rng.shuffle(idx)
        for n, i in enumerate(idx):
            fold[i] = n % k
    return fold


def cross_validate(data, names, builtin, w, k, fitted=True):
    """(accuracy, per-fold accuracies, confusion matrix) on held-out folds; the built-in values need no fitting."""
    fold, conf, per = folds_of(data, k), collections.Counter(), []
    for f in range(k):
        train = [d for d, g in zip(data, fold) if g != f]
        test = [d for d, g in zip(data, fold) if g == f]
        protos = fit(train, names, builtin) if fitted else builtin
        ok = 0
        for m, x in test:
            p = predict(x, protos, w)
            conf[(m, p)] += 1
            ok += p == m
        per.append(ok / len(test) if test else float("nan"))
    n = sum(conf.values())
    return sum(v for (a, b), v in conf.items() if a == b) / n, per, conf


def show(title, acc, per, conf, ids, present):
    print(f"\n{title}: {100 * acc:.1f} % correct  (folds: {', '.join(f'{100 * p:.0f}' for p in per)} %)")
    head = "labelled \\ predicted"
    print(head.ljust(22) + "".join(ids[i][:9].rjust(10) for i in present) + "    recall")
    for a in present:
        row = [conf[(a, b)] for b in present]
        print(ids[a].ljust(22) + "".join(str(v).rjust(10) for v in row) + f"    {100 * conf[(a, a)] / max(sum(row), 1):.0f} %")
    pairs = sorted(((v, ids[a], ids[b]) for (a, b), v in conf.items() if a != b), reverse=True)[:5]
    print("most confused (labelled -> predicted):", ", ".join(f"{a} -> {b} ({v})" for v, a, b in pairs) or "none")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("jsonl", nargs="+")
    ap.add_argument("--keep")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--without", action="append", default=[])
    ap.add_argument("--weights", action="store_true")
    ap.add_argument("--rust", action="store_true")
    a = ap.parse_args()

    feats, moods = read_mood_rs()
    names, w0 = [n for n, _ in feats], [0.0 if n in a.without else x for n, x in feats]
    ids, builtin = [m for m, _ in moods], [p for _, p in moods]
    keep = {l.strip() for l in Path(a.keep).read_text(encoding="utf-8").splitlines() if l.strip()} if a.keep else None
    data = []
    for path in a.jsonl:
        for line in open(path, encoding="utf-8"):
            if not line.startswith("{"):
                continue
            r = json.loads(line)
            if r["want"] not in ids or (keep is not None and r["file"] not in keep):
                continue
            data.append((ids.index(r["want"]), [r["features"].get(n) for n in names]))
    count = collections.Counter(m for m, _ in data)
    present = [i for i in range(len(ids)) if count[i] >= a.folds]
    data = [d for d in data if d[0] in present]
    print("tracks per mood:", ", ".join(f"{ids[i]} {count[i]}" for i in range(len(ids)) if count[i]),
          "| moods with fewer than", a.folds, "tracks are left out:", ", ".join(ids[i] for i in range(len(ids)) if 0 < count[i] < a.folds) or "none")
    if not data:
        sys.exit("no usable tracks")

    # Only the moods that have data take part, so the built-in values are not blamed for moods nobody labelled.
    sub = [i for i in present]
    remap = {old: new for new, old in enumerate(sub)}
    d2 = [(remap[m], x) for m, x in data]
    base2 = [builtin[i] for i in sub]
    ids2 = [ids[i] for i in sub]
    pres2 = list(range(len(sub)))

    acc0, per0, conf0 = cross_validate(d2, names, base2, w0, a.folds, fitted=False)
    show("built-in values (never fitted on these tracks)", acc0, per0, conf0, ids2, pres2)
    acc1, per1, conf1 = cross_validate(d2, names, base2, w0, a.folds)
    show(f"fitted values, {a.folds}-fold cross-validation", acc1, per1, conf1, ids2, pres2)
    print(f"\nchange: {100 * (acc1 - acc0):+.1f} percentage points; better in {sum(x > y for x, y in zip(per1, per0))} of {a.folds} folds")

    w = list(w0)
    if a.weights:
        # coordinate search: halve or double one weight at a time, keep a change only if the cross-validated accuracy rises
        # by more than 3 points and is no worse in most folds; anything less is noise at this sample size.
        best, bper = acc1, per1
        for _ in range(2):
            for j in range(len(names)):
                for f in (0.5, 2.0):
                    t = list(w)
                    t[j] = w[j] * f
                    acc, per, _ = cross_validate(d2, names, base2, t, a.folds)
                    if acc > best + 0.03 and sum(x >= y for x, y in zip(per, bper)) >= a.folds - 1:
                        w, best, bper = t, acc, per
        print("\nweights:", ", ".join(f"{n} {x:g}" for n, x in zip(names, w)),
              "(unchanged)" if w == w0 else f"-> {100 * best:.1f} % cross-validated (was {100 * acc1:.1f} %)")
        acc1, per1, conf1 = cross_validate(d2, names, base2, w, a.folds)

    full = fit(d2, names, base2)
    # confidence of the fitted values on the tracks they were fitted on is NOT an accuracy; shown for the mean only
    print("\nmean confidence on held-in tracks (not an accuracy):", f"{100 * statistics.mean(confidence(x, full, w) for _, x in d2):.0f} %")
    if a.rust:
        print("\nfitted values (typical, spread) per mood, for src/mood.rs; moods without enough tracks keep the built-in line:")
        for i, m in enumerate(ids):
            p = full[remap[i]] if i in remap else builtin[i]
            nums = ", ".join(f"({t:.4g}, {s:.3g})" for t, s in p)
            print(f'    {m}: [{nums}]')


if __name__ == "__main__":
    main()
