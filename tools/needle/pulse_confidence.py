r"""Does anything Pulse knows about its own grid tell that the grid is wrong? (step 3: confidence, only with evidence)

What the analysis knows about a grid (all printed by pulse_cli; pulse_phase.py prep --refined keeps them):
  on       share of the beat peaks within 35 ms of the grid (the deck's "confidence" is made of this today:
           0.5 + 0.5 * (on - 0.6) / 0.3)
  act      mean beat activation at the grid beats
  margin   bar-line vote: (votes for the chosen bar position - votes for the runner-up) / all votes
  spread   phase stage (pulse_phase.py): how far the drum onsets scatter around the grid, ms
  parity   activation of the weaker half of the grid beats / the stronger half (the double-time rule's number;
           not asked for, kept as a fifth)
What can be wrong with a grid (only tracks where Pulse gives a steady grid count; without one the deck falls
back to the old analyser):
  tempo    the tempo is not the reference's (2 %; on the mixed set 0.04 octave, as pulse_passfail.py)
  off10    the tempo is right but the grid sits more than 10 ms from the reference: `before` the phase stage
           (the grid the deck uses today) and `after` it
  bar      the tempo is right but the bar line is not on beat 1 (under 90 % of the human downbeats on the
           anchor's bar grid; hand-labelled sets only; not asked for, but it is what `margin` is about)

For every number and every fault: AUC = the chance that a faulty grid looks worse on that number than a good
one (0.5 = the number says nothing, 1 = it sorts them perfectly), with a 95 % bootstrap interval (2000 draws).
A number "predicts" when the interval stays above 0.5 on the test tracks. Then: logistic calibration fitted on
the fit tracks, and on the test tracks a reliability table (predicted chance of a fault against how often there
was one) and the Brier score against always saying the base rate.

  fit / test   tempo: the mixed set split in two halves (random.Random(1)); every track of it was in the
               network's training set, so this flatters the network; Raveform's test tracks have too few wrong
               tempos to count (3 or 4 of 195) and are shown for reference
               off10, bar: fit on rave_cal (98 Raveform training tracks), test on rave_test (never trained on);
               stems / lab (reference: the line through a drum stem's kicks) as a second pair

    .venv\Scripts\python.exe pulse_confidence.py --model p4_w24_l6
    .venv\Scripts\python.exe pulse_confidence.py --model p4_w24_l6 --python     spread from refine() here (sets with bands)
"""
import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pulse_phase as pp          # noqa: E402

NUMBERS = ("on", "act", "margin", "spread", "parity")
WORSE = {"on": -1, "act": -1, "margin": -1, "spread": +1, "parity": -1}     # which way a number points at a fault
SETS = ("mixed", "rave_cal", "rave_test", "stems", "lab")


# ── per track ────────────────────────────────────────────────────────────────

def track_rows(folder, model, name, python=False):
    """One row per track with a steady Pulse grid: the numbers, and which faults the grid has (None = cannot be
    told for this track: no steady reference, or the tempo is wrong so there is no phase to compare)."""
    rows, no_grid, skipped = [], 0, 0
    for t, z in pp.load_set(folder, model):
        beats = z["beats"]
        if len(beats) < 16:
            continue
        if python:
            g = z[f"grid_{model}"][1]
        elif f"refined_{model}" in z:
            g = z[f"refined_{model}"]
        else:
            skipped += 1
            continue
        if not (g[0] and g[1]):
            no_grid += 1
            continue
        bpm = float(g[2])
        if name == "mixed":                                             # the teacher's beats: a tempo, no precise phase
            ref_bpm, ref_ok, precise = 60.0 / float(np.median(np.diff(beats))), True, False
            right = abs(np.log2(bpm / ref_bpm)) < 0.04
        elif "ref" in z:                                                # the line through a drum stem's kicks (fitted with
            ref_bpm, precise = float(z["ref"][0]), True                 # Pulse's tempo as the hint: no test of the tempo)
            ref_ok = bool(z["ref"][5] < 3 and z["ref"][2] >= 64)
            right = abs(bpm / ref_bpm - 1) < 0.02 if ref_bpm > 0 else False
        else:                                                           # hand labels, or beatgrid_lab's stem line
            ref_bpm, _, jitter = pp.human_line(beats)
            ref_ok, precise = bool(jitter < 15), True
            right = abs(bpm / ref_bpm - 1) < 0.02
        row = {"id": t, "set": name, "bpm": bpm, "ref_bpm": float(ref_bpm), "on": float(g[5]), "act": float(g[11]),
               "margin": float(g[12]), "parity": float(g[9]), "halved": bool(g[10]), "spread": None,
               "tempo": None, "before": None, "after": None, "bar": None}
        if python:
            if "env_high" in z:
                r = pp.refine(z["low"].astype(np.float64), bpm, float(g[3]), z["env_mid"], z["env_high"])
                first, shift, row["spread"] = float(g[3]), r.delta, r.spread * 1000 if r.used >= 8 else None
            else:
                first, shift = float(g[3]), None
        else:
            first, shift = float(g[3]) - float(g[14]), float(g[14])
            row["spread"] = float(g[16]) * 1000 if g[17] >= 8 else None
        if ref_ok and name not in ("stems", "lab"):
            row["tempo"] = not right
        if ref_ok and right and precise:
            off = pp.track_offset(beats, bpm, first)
            if off is not None:
                row["before"] = bool(abs(off) > 0.010)
                row["offset_before_ms"] = off * 1000
            if shift is not None:
                off = pp.track_offset(beats, bpm, first + shift)
                if off is not None:
                    row["after"] = bool(abs(off) > 0.010)
                    row["offset_after_ms"] = off * 1000
            d = z["downbeats"] if "downbeats" in z else np.zeros(0)
            if len(d) >= 4 and name.startswith("rave"):
                bar = 240.0 / bpm
                row["bar"] = bool(np.mean(np.abs((d - first + bar / 2) % bar - bar / 2) < 0.07) <= 0.9)
        rows.append(row)
    return rows, no_grid, skipped


# ── statistics ───────────────────────────────────────────────────────────────

def pick(rows, number, fault):
    """-> (score, faulty) over the rows where both are known; score: higher = looks worse."""
    sel = [r for r in rows if r[fault] is not None and r[number] is not None]
    return np.array([WORSE[number] * r[number] for r in sel], float), np.array([r[fault] for r in sel], bool)


def auc(score, bad):
    """The chance that a faulty track's score is above a good track's (ties count half)."""
    n1, n0 = int(bad.sum()), int((~bad).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score))
    ranks[order] = np.arange(1, len(score) + 1)
    s = score[order]
    i = 0
    while i < len(s):                                                   # tied scores share their mean rank
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return float((ranks[bad].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def auc_interval(score, bad, draws=2000, seed=0):
    rng = np.random.default_rng(seed)
    a, bs = auc(score, bad), []
    for _ in range(draws):
        i = rng.integers(0, len(score), len(score))
        v = auc(score[i], bad[i])
        if np.isfinite(v):
            bs.append(v)
    lo, hi = np.percentile(bs, [2.5, 97.5]) if len(bs) > 100 else (float("nan"), float("nan"))
    return a, float(lo), float(hi)


def logistic_fit(x, y, l2=1e-3):
    """Logistic regression by Newton steps on standardised columns -> (weights with the constant first, mean, sd)."""
    x, y = np.asarray(x, float).reshape(len(y), -1), np.asarray(y, float)
    mu, sd = x.mean(0), x.std(0) + 1e-9
    z = np.c_[np.ones(len(x)), (x - mu) / sd]
    w = np.zeros(z.shape[1])
    for _ in range(100):
        p = 1 / (1 + np.exp(-z @ w))
        grad = z.T @ (p - y) + l2 * np.r_[0, w[1:]]
        hess = (z * (p * (1 - p))[:, None]).T @ z + l2 * np.diag(np.r_[0, np.ones(len(w) - 1)])
        step = np.linalg.solve(hess, grad)
        w -= step
        if np.abs(step).max() < 1e-9:
            break
    return w, mu, sd


def logistic_chance(model, x):
    w, mu, sd = model
    x = np.asarray(x, float).reshape(-1, len(mu))
    return 1 / (1 + np.exp(-np.c_[np.ones(len(x)), (x - mu) / sd] @ w))


def reliability(p, y, edges=(0, 0.05, 0.1, 0.2, 0.35, 0.5, 0.7, 1.0001)):
    out = []
    for lo, hi in zip(edges, edges[1:]):
        m = (p >= lo) & (p < hi)
        if m.sum():
            out.append((lo, min(hi, 1.0), int(m.sum()), float(p[m].mean()), float(y[m].mean()), int(y[m].sum())))
    return out


# ── report ───────────────────────────────────────────────────────────────────

def auc_table(title, rows, faults, numbers=NUMBERS):
    print(f"\n{title}")
    out = {}
    for fault in faults:
        known = [r for r in rows if r[fault] is not None]
        bad = sum(r[fault] for r in known)
        print(f"  {fault:6s}: {bad} faulty of {len(known)} tracks" + ("" if bad >= 10 else "   (too few faulty tracks for a firm AUC)"))
        if bad < 3 or bad > len(known) - 3:
            continue
        for number in numbers:
            s, y = pick(rows, number, fault)
            if y.sum() < 3 or (~y).sum() < 3:
                continue
            a, lo, hi = auc_interval(s, y)
            verdict = "predicts" if lo > 0.5 else ("the wrong way" if hi < 0.5 else "no")
            print(f"      {number:7s} AUC {a:.2f}  (95 % {lo:.2f} .. {hi:.2f})  n {len(y)}  {verdict}")
            out[(fault, number)] = (a, lo, hi, int(y.sum()), len(y))
    return out


def calibrate(title, fit, test, fault, numbers):
    """Logistic fit on `fit`, checked on `test`: AUC, Brier score against the base rate, reliability table."""
    def table(rows):
        sel = [r for r in rows if r[fault] is not None and all(r[k] is not None for k in numbers)]
        return np.array([[r[k] for k in numbers] for r in sel], float), np.array([r[fault] for r in sel], bool)
    xf, yf = table(fit)
    xt, yt = table(test)
    if yt.sum() < 10:
        print(f"\n{title}: too few faulty test tracks to check a calibration ({int(yt.sum())})")
        return None
    if yf.sum() >= 10:
        model = logistic_fit(xf, yf)
        p, base, how = logistic_chance(model, xt), float(yf.mean()), f"fitted on {len(yf)} tracks ({int(yf.sum())} faulty)"
    else:
        # too few faulty fit tracks: every test track's chance from a fit on the other four fifths of the test set
        order, p = np.random.default_rng(2).permutation(len(yt)), np.empty(len(yt))
        for k in range(5):
            part = order[k::5]
            rest = np.setdiff1d(order, part)
            p[part] = logistic_chance(logistic_fit(xt[rest], yt[rest]), xt[part])
        model, base = logistic_fit(xt, yt), float(yt.mean())            # the formula shown: fitted on all of them
        how = (f"the fit tracks have only {int(yf.sum())} faulty ones, so: 5-fold on the test tracks, each track's chance from a "
               "fit on the other four fifths")
    brier, brier_base = float(np.mean((p - yt) ** 2)), float(np.mean((base - yt) ** 2))
    a, lo, hi = auc_interval(p, yt)
    w, mu, sd = model
    terms = " ".join(f"{w[i + 1] / sd[i]:+.3f}*{k}" for i, k in enumerate(numbers))
    const = w[0] - float(np.sum(w[1:] * mu / sd))
    print(f"\n{title}: chance of `{fault}` = 1 / (1 + exp(-({const:+.3f} {terms})))")
    print(f"  {how}; tested on {len(yt)} ({int(yt.sum())} faulty, {yt.mean():.0%})")
    print(f"  on the test tracks: AUC {a:.2f} (95 % {lo:.2f} .. {hi:.2f}); Brier {brier:.4f} against {brier_base:.4f} for always "
          f"saying {base:.0%} ({'better' if brier < brier_base else 'no better'}: {(1 - brier / brier_base):+.0%})")
    print("  reliability on the test tracks (predicted chance -> how often faulty):")
    for lo_, hi_, n, mp, obs, k in reliability(p, yt):
        print(f"      {lo_:4.0%} .. {hi_:4.0%}: {n:4d} tracks, predicted {mp:5.1%}, faulty {obs:5.1%} ({k})")
    order = np.argsort(-p)
    for share in (0.1, 0.2, 0.3):                                        # what flagging the worst-looking tracks would catch
        n = max(1, int(round(share * len(p))))
        print(f"      flag the {share:.0%} that look worst ({n}): {int(yt[order[:n]].sum())} of the {int(yt.sum())} faulty ones among them "
              f"({yt[order[:n]].mean():.0%} of the flagged are faulty)")
    return {"numbers": list(numbers), "const": const, "weights": [float(w[i + 1] / sd[i]) for i in range(len(numbers))],
            "auc": a, "auc_lo": lo, "auc_hi": hi, "brier": brier, "brier_base": brier_base, "fit": len(yf), "test": len(yt),
            "cross_fitted": bool(yf.sum() < 10)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="p4_w24_l6")
    ap.add_argument("--dir", help="the folder with the prepared sets, if it is not F:\\needle\\pulse_phase")
    ap.add_argument("--python", action="store_true", help="the grid without the phase switch and spread from refine() here "
                    "(for sets prepared with --bands but not yet with --refined)")
    ap.add_argument("--sets", nargs="+", default=list(SETS))
    a = ap.parse_args()
    if a.dir:
        root = Path(a.dir)
    else:
        from pulse import DATA                                          # needs torch; --dir does not
        root = DATA / "pulse_phase"
    rows = {}
    for name in a.sets:
        if not (root / name).exists():
            continue
        rows[name], no_grid, skipped = track_rows(root / name, a.model, name, a.python)
        print(f"{name}: {len(rows[name])} tracks with a steady grid, {no_grid} without" +
              (f", {skipped} not yet through prep --refined" if skipped else ""), flush=True)
    result = {"model": a.model, "auc": {}, "calibration": {}}

    def keep(title, table):
        result["auc"][title] = {f"{f} / {n}": v for (f, n), v in table.items()}

    if "mixed" in rows:
        mixed = sorted(rows["mixed"], key=lambda r: r["id"])
        random.Random(1).shuffle(mixed)
        half = len(mixed) // 2
        keep("mixed", auc_table("mixed set, all tracks (reference: the teacher's tempo)", mixed, ("tempo",)))
        t = auc_table("mixed set, the test half", mixed[half:], ("tempo",))
        keep("mixed, test half", t)
        good = [n for n in NUMBERS if ("tempo", n) in t and t[("tempo", n)][1] > 0.5]
        for n in good:
            result["calibration"][f"tempo / {n}"] = calibrate(f"wrong tempo from `{n}`", mixed[:half], mixed[half:], "tempo", (n,))
        if len(good) > 1:
            result["calibration"]["tempo / all that predict"] = calibrate("wrong tempo from " + " + ".join(good), mixed[:half], mixed[half:], "tempo", tuple(good))
        if not good:
            print("\nwrong tempo: none of the numbers predicts it on the test half")
    for fit_name, test_name, what in (("rave_cal", "rave_test", "hand labels"), ("stems", "lab", "the line through a drum stem's kicks")):
        if test_name not in rows:
            continue
        faults = ("tempo", "before", "after", "bar") if test_name == "rave_test" else ("before", "after")
        t = auc_table(f"{test_name} (reference: {what})", rows[test_name], faults)
        keep(test_name, t)
        if fit_name in rows:
            keep(fit_name, auc_table(f"{fit_name} (the fit tracks; reference: {what})", rows[fit_name], faults))
            for fault in faults:
                good = [n for n in NUMBERS if (fault, n) in t and t[(fault, n)][1] > 0.5]
                for n in good:
                    result["calibration"][f"{test_name}: {fault} / {n}"] = calibrate(
                        f"{test_name}: `{fault}` from `{n}`", rows[fit_name], rows[test_name], fault, (n,))
                if len(good) > 1:
                    result["calibration"][f"{test_name}: {fault} / all that predict"] = calibrate(
                        f"{test_name}: `{fault}` from " + " + ".join(good), rows[fit_name], rows[test_name], fault, tuple(good))
                if not good and fault in ("before", "after", "tempo"):
                    print(f"\n{test_name}: none of the numbers predicts `{fault}`")
    out = root / f"confidence_{a.model}{'_python' if a.python else ''}.json"
    result["rows"] = rows
    out.write_text(json.dumps(result, indent=1))
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
