r"""Pulse, round 2: fine-tune on Raveform's HUMAN beat annotations, and fit downbeats as a bar grid.

    python pulse_rave.py prep                         match audio to annotations, log-mels, fixed split
    python pulse_rave.py train --init runs_pulse\p1_w24\best.pt --name p2_w24
    python pulse_rave.py eval                         test set: Beat This!, p1, p2; peak vs bar-fit downbeats

Data: I:\datasets\raveform\downloads (the YouTube versions the annotations were made on), matched by title.
Split: 200 tracks are test (never trained on), the rest train. Training batches are half Raveform (human
targets) and half the Beat This! teacher set (its soft targets), so the network keeps what it learned on the
3,000 teacher tracks while it corrects towards human downbeats.

Bar-fit downbeats: house / techno bars are 4 beats almost everywhere. Instead of picking downbeat peaks one by
one, every beat gets a score = downbeat activation summed over the same bar position for 8 bars either side;
a beat is a downbeat when its score beats the other three positions around it. One wrong peak can't move the
bar line, but a real bar shift (a 2-beat fill, a 3/4 break) still shows up within a few bars.
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from pulse import FPS, T_CHUNK, Pulse, Tracks, count_params, f_measure, peaks, predict_track   # noqa: E402
from pulse_raveform import AUDIO, load_annotations, match, tempo_ok               # noqa: E402

RAVE = Path(r"F:\needle\beat_rave")
TEACHER = Path(r"F:\needle\beat_teacher")
N_TEST = 200


# ── Data ─────────────────────────────────────────────────────────────────────

def match_by_duration(files, ann):
    r"""Every file came from the annotated YouTube list (I:\datasets\raveform\downloaded.txt holds the ids), but
    the files are named after the YouTube title, not the Raveform title. The video's duration (in tracks.jsonl)
    is the key: candidates = downloaded annotated ids within 1.5 s of the file's length; several -> best title."""
    import difflib
    import io
    import soundfile as sf
    import zipfile
    from pulse_raveform import ZIP, norm
    z = zipfile.ZipFile(ZIP)
    dur = {}
    for line in io.TextIOWrapper(z.open("raveform/tracks.jsonl"), encoding="utf-8"):
        t = json.loads(line)
        dur[t["id"]] = t["duration"]
    got = {ln.split()[1] for ln in (AUDIO.parent / "downloaded.txt").read_text().splitlines() if len(ln.split()) >= 2}
    allowed = [t for t in ann if t in got and t in dur]
    ad = np.array([dur[t] for t in allowed])
    out, used = {}, {}
    for f in files:
        try:
            d = sf.info(str(f)).duration
        except Exception:                                               # noqa: BLE001
            continue
        c = [allowed[i] for i in np.where(np.abs(ad - d) < 1.5)[0]]
        if not c:
            continue
        best = max(c, key=lambda t: difflib.SequenceMatcher(None, norm(f.stem), norm(ann[t][0])).ratio())
        if best in used:                                                # two files claim one id: keep neither
            out.pop(used[best], None)
            continue
        out[f], used[best] = best, f
    return out


def prep():
    import soundfile as sf
    import soxr
    from beat_this.preprocessing import LogMelSpect
    (RAVE / "mels").mkdir(parents=True, exist_ok=True)
    ann = load_annotations()
    files = sorted(p for p in AUDIO.rglob("*") if p.suffix.lower() in {".flac", ".wav", ".mp3", ".m4a", ".opus"})
    pairs = match_by_duration(files, ann)
    old = match(files, ann)                                             # the title-only match used for round 2
    agree = sum(pairs.get(f) == t for f, t in old.items())
    print(f"{len(files)} files, {len(pairs)} matched by duration (title-only match: {len(old)}, "
          f"{agree} of those agree)", flush=True)
    index = json.loads((RAVE / "index.json").read_text()) if (RAVE / "index.json").exists() else {}
    fixed = {t for t, v in index.items() if Path(v["file"]) in pairs and pairs[Path(v["file"])] != t}
    for t in fixed:                                                     # an old title match the duration disagrees with
        print("dropping old match", index[t]["file"], "->", t)
        index.pop(t)
    spect = LogMelSpect(device="cuda")
    for i, (f, tid) in enumerate(pairs.items()):
        if tid in index:
            continue
        try:
            x, sr = sf.read(str(f), dtype="float32", always_2d=True)
        except Exception as ex:                                             # noqa: BLE001
            print("skip", f.name, ex)
            continue
        x = x.mean(1)
        x = soxr.resample(x, sr, 22050) if sr != 22050 else x
        with torch.inference_mode():
            m = spect(torch.from_numpy(x).cuda()).T.half().cpu().numpy()
        np.save(RAVE / "mels" / f"{tid}.npy", m)
        _, b, d = ann[tid]
        np.savez(RAVE / f"{tid}.npz", beats=b.astype(np.float32), downbeats=d.astype(np.float32))
        index[tid] = {"file": str(f), "title": ann[tid][0], "frames": int(m.shape[1])}
        if i % 100 == 0:
            print(f"[{i + 1}/{len(pairs)}]", flush=True)
            (RAVE / "index.json").write_text(json.dumps(index, indent=1))
    (RAVE / "index.json").write_text(json.dumps(index, indent=1))
    if not (RAVE / "test.txt").exists():                                    # fixed once, never reshuffled
        ids = sorted(index)
        random.Random(0).shuffle(ids)
        (RAVE / "test.txt").write_text("\n".join(sorted(ids[:N_TEST])))
    print(len(index), "tracks prepared")


def targets(times, n, width=1):
    """Human beat times -> frame targets: 1 at the beat frame, 0.5 one frame either side (annotation jitter)."""
    y = np.zeros(n, np.float32)
    f = np.round(np.asarray(times) * FPS).astype(int)
    f = f[(f >= 0) & (f < n)]
    for k in range(1, width + 1):
        for s in (-k, k):
            g = np.clip(f + s, 0, n - 1)
            y[g] = np.maximum(y[g], 0.5)
    y[f] = 1.0
    return y


class Rave:
    """Tracks with exact beat labels: Raveform (human) by default, or root=beat_loops (synthetic)."""
    def __init__(self, split, root=RAVE):
        index = json.loads((root / "index.json").read_text())
        test = set((root / "test.txt").read_text().split())
        self.ids = sorted(t for t in index if (t in test) == (split == "test"))
        self.mels = {t: np.load(root / "mels" / f"{t}.npy", mmap_mode="r") for t in self.ids}
        self.lab = {}
        for t in self.ids:
            z = np.load(root / f"{t}.npz")
            self.lab[t] = (z["beats"], z["downbeats"])

    def batch(self, n, rng):
        """Random 512-frame crops with the same augmentation as the teacher set (tempo +-8 %, pitch, gain, tilt)."""
        xs, ys = [], []
        while len(xs) < n:
            t = rng.choice(self.ids)
            m = self.mels[t]
            b, d = self.lab[t]
            r = rng.uniform(0.92, 1.08)
            need = int(np.ceil(T_CHUNK * r)) + 2
            if m.shape[1] < need + 1:
                continue
            a = rng.randrange(0, m.shape[1] - need)
            seg = torch.from_numpy(np.asarray(m[:, a:a + need], dtype=np.float32))
            pos = torch.arange(T_CHUNK) * r
            i0 = pos.floor().long().clamp(max=need - 2)
            fr = (pos - i0).unsqueeze(0)
            seg = seg[:, i0] * (1 - fr) + seg[:, i0 + 1] * fr
            t0 = a / FPS                                                    # beat times in the stretched crop's frames
            tb = (b[(b >= t0) & (b < t0 + need / FPS)] - t0) / r
            td = (d[(d >= t0) & (d < t0 + need / FPS)] - t0) / r
            tgt = torch.from_numpy(np.stack([targets(tb, T_CHUNK), targets(td, T_CHUNK)]))
            sh = rng.randint(-2, 2)
            seg = torch.roll(seg, sh, 0)
            seg = seg + rng.uniform(-1.2, 0.6) + torch.linspace(-1, 1, seg.shape[0])[:, None] * rng.uniform(-0.4, 0.4)
            xs.append(seg)
            ys.append(tgt)
        return torch.stack(xs)[:, None], torch.stack(ys)


# ── Downbeats as a bar grid ──────────────────────────────────────────────────

def bar_fit(beats, down_logit, bars=8):
    """beats (s), downbeat logit (frames) -> downbeat times (s). See the module docstring."""
    if len(beats) < 8:
        return beats[:0]
    act = torch.sigmoid(down_logit).numpy()
    a = act[np.clip(np.round(beats * FPS).astype(int), 0, len(act) - 1)]
    comb = np.zeros(8 * bars + 1)
    comb[::4] = 1                                                           # the same bar position, 8 bars either side
    s = np.convolve(a, comb, "same")
    keep = []
    for i in range(len(beats)):
        nb = [s[j] for j in (i - 3, i - 2, i - 1, i + 1, i + 2, i + 3) if 0 <= j < len(s)]
        if all(s[i] > v for v in nb):
            keep.append(i)
    return beats[keep]


# ── Train / eval ─────────────────────────────────────────────────────────────

def train(a):
    dev = "cuda"
    torch.cuda.set_per_process_memory_fraction(a.gpu_mem)
    if a.init:
        ck = torch.load(a.init, map_location="cpu", weights_only=False)
        model = Pulse(ck["width"], ck.get("levels", 5))
        model.load_state_dict(ck["model"])
    else:                                                               # from scratch (e.g. a deeper network)
        ck = {"width": a.width, "levels": a.levels}
        model = Pulse(a.width, a.levels)
    model.to(dev)
    rv, tt = Rave("train"), Tracks(Path(a.teacher), "train")
    lp = Rave("train", root=Path(a.loops)) if a.loop_frac > 0 else None
    print(f"init {a.init} (width {ck['width']}, {count_params(model):,} params); {len(rv.ids)} Raveform + "
          f"{len(tt.ids)} teacher train tracks", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.steps, pct_start=0.05)
    run = HERE / "runs_pulse" / a.name
    run.mkdir(parents=True, exist_ok=True)
    rng = random.Random(2)
    val = Rave("train")                                                     # quick check: 40 train tracks (test stays untouched)
    val.ids = sorted(val.ids)[:40]
    t0 = time.time()
    for step in range(1, a.steps + 1):
        n_loop = int(round(a.batch * a.loop_frac)) if lp else 0
        n_hum = (a.batch - n_loop) // 2
        xh, yh = rv.batch(n_hum, rng)
        if n_loop:                                                      # synthetic tracks: exact labels, like human ones
            xl, yl = lp.batch(n_loop, rng)
            xh, yh = torch.cat([xh, xl]), torch.cat([yh, yl])
        xt, yt = tt.batch(a.batch - n_hum - n_loop, rng)
        x = torch.cat([xh, xt]).to(dev)
        p = torch.cat([yh, torch.sigmoid(yt)]).to(dev)
        w = 1 + 4 * F.max_pool1d(p, 5, 1, 2)
        w[: len(xh), 1] *= a.down_weight                                    # human downbeats: the part p1 got wrong
        out = model(x)
        loss = (F.binary_cross_entropy_with_logits(out, p, reduction="none") * w).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % 200 == 0:
            print(f"step {step:6d} loss {loss.item():.4f} {step / (time.time() - t0):.1f} steps/s", flush=True)
        if step % 5000 == 0 or step == a.steps:
            model.eval()
            fb, fd = [], []
            for t in val.ids:
                y = predict_track(model, val.mels[t], dev)
                eb = peaks(y[0])
                fb.append(f_measure(val.lab[t][0], eb))
                fd.append(f_measure(val.lab[t][1], bar_fit(eb, y[1])))
            model.train()
            print(f"check at step {step}: beat F {np.mean(fb):.4f}, downbeat F (bar fit) {np.mean(fd):.4f}", flush=True)
            torch.save({"model": model.state_dict(), "width": ck["width"], "levels": ck.get("levels", 5), "step": step},
                       run / "last.pt")
    print("done")


def evaluate(a):
    from beat_this.inference import Spect2Frames
    from beat_this.model.postprocessor import Postprocessor
    te = Rave("test")
    s2f = Spect2Frames(checkpoint_path=str(Path(r"F:\needle\beat_this\final0.ckpt")), device="cuda", float16=False)
    post = Postprocessor(type="minimal")
    models = {}
    for name in a.models:
        ck = torch.load(HERE / "runs_pulse" / name / ("best.pt" if (HERE / "runs_pulse" / name / "best.pt").exists() else "last.pt"),
                        map_location="cpu", weights_only=False)
        m = Pulse(ck["width"], ck.get("levels", 5))
        m.load_state_dict(ck["model"])
        models[name] = m.to("cuda").eval()
    rows = {k: [] for k in ["beat_this"] + [f"{n} {v}" for n in models for v in ("peaks", "bar fit")]}
    for t in te.ids:
        b, d = te.lab[t]
        mel = np.asarray(te.mels[t], dtype=np.float32)
        with torch.inference_mode():
            bl, dl = s2f(torch.from_numpy(mel.T).cuda())
            bb, dd = post(bl, dl)
        bb, dd = np.asarray(bb), np.asarray(dd)
        rows["beat_this"].append((f_measure(b, bb), f_measure(d, dd), tempo_ok(b, bb)))
        for n, m in models.items():
            y = predict_track(m, mel, "cuda")
            eb = peaks(y[0])
            rows[f"{n} peaks"].append((f_measure(b, eb), f_measure(d, peaks(y[1], min_gap=0.3)), tempo_ok(b, eb)))
            rows[f"{n} bar fit"].append((f_measure(b, eb), f_measure(d, bar_fit(eb, y[1])), tempo_ok(b, eb)))
    print(f"\nRaveform test set: {len(te.ids)} tracks never trained on, human annotations, +-70 ms")
    out = {}
    for k, v in rows.items():
        v = np.array(v, dtype=float)
        out[k] = v.mean(0).tolist()
        print(f"{k:22s} beat F {v[:, 0].mean():.3f}  downbeat F {v[:, 1].mean():.3f}  tempo family right {v[:, 2].mean() * 100:.0f}%")
    (RAVE / "eval.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["prep", "train", "eval"])
    ap.add_argument("--init", default=str(HERE / "runs_pulse" / "p1_w24" / "best.pt"), help="'' = train from scratch")
    ap.add_argument("--width", type=int, default=24, help="from scratch only")
    ap.add_argument("--levels", type=int, default=5, help="from scratch only; 6 = ~12 s context instead of ~6 s")
    ap.add_argument("--loops", default=r"F:\needle\beat_loops", help="synthetic loop tracks (pulse_loops.py)")
    ap.add_argument("--loop-frac", type=float, default=0.0, help="share of each batch from the loop tracks")
    ap.add_argument("--teacher", default=str(TEACHER), help="teacher label folder (beat_teacher or beat_teacher_ens)")
    ap.add_argument("--name", default="p2_w24")
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--down-weight", type=float, default=2.0)
    ap.add_argument("--gpu-mem", type=float, default=0.3)
    ap.add_argument("--models", nargs="+", default=["p1_w24", "p2_w24"])
    a = ap.parse_args()
    {"prep": prep, "train": lambda: train(a), "eval": lambda: evaluate(a)}[a.cmd]()
