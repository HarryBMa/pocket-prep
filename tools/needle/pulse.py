r"""Pulse: a tiny beat / downbeat tracker for the RA8P1's Ethos-U55, distilled from Beat This! (Needle's recipe).

Input: the same log-mel spectrogram Beat This! uses (22.05 kHz, n_fft 1024, hop 441 = 50 frames/s, 128 mel
bands 30 Hz - 11 kHz). Output: beat and downbeat logits per frame. The beat grid (tempo + phase) is then fitted
to the beat activation on the M85, so there is no tempo range to cap: the network says where the beats are.

Network (only Conv2D, ReLU, ADD and nearest-neighbour 2x upsampling, all Ethos-U55 operators):
  frequency stem   (1, 128, T) -> conv 3x3 stride (2, 1) six times -> (C, 2, T) -> conv (2 x 1) -> (C, 1, T)
  time U-Net       5 levels, each conv (1 x 5) stride 2 in time; back up with 2x nearest upsampling + skip add
  head             conv 1x1 -> (2, 1, T): beat, downbeat logits
T = 512 frames (10.24 s) per chunk; the coarsest level sees about 6 s around each frame.

    python pulse.py mels  --teacher F:\needle\beat_teacher          precompute log-mels (once)
    python pulse.py train --teacher F:\needle\beat_teacher --name p1 [--width 32] [--steps 60000]
    python pulse.py eval  --teacher F:\needle\beat_teacher --checkpoint runs_pulse\p1\best.pt

Device: cuda, then mps (Apple GPU), then cpu; --device or PULSE_DEVICE overrides (cpu while the PC's GPU is busy).
Data: F:\needle on the PC. Elsewhere PULSE_DATA names the folder with the same layout (the Mac mini:
~/train/data); the absolute Windows paths stored in index files are mapped to it, the files are not rewritten.
"""
import argparse
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

FPS = 50
T_CHUNK = 512
MELS = 128
PC_DATA = "F:\\needle"
DATA = Path(os.environ.get("PULSE_DATA", PC_DATA)).expanduser()


def local_path(p) -> Path:
    """A path as stored on the PC (F:\\needle\\...) -> where that file is on this machine (PULSE_DATA)."""
    s = str(p)
    if str(DATA) != PC_DATA and s[:len(PC_DATA) + 1].lower() == PC_DATA.lower() + "\\":
        return DATA.joinpath(*s[len(PC_DATA) + 1:].split("\\"))
    return Path(s).expanduser()


def pick_device(want=None):
    """cuda, then mps (Apple GPU), then cpu. `want` (--device) or PULSE_DEVICE overrides the choice."""
    want = want or os.environ.get("PULSE_DEVICE")
    if want:
        return want
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def limit_gpu_memory(dev, fraction):
    if dev == "cuda":                                                       # the only backend that has this call
        torch.cuda.set_per_process_memory_fraction(fraction)


def allow_open_files(n=16384):
    """Every memory-mapped mel holds a file descriptor on macOS / Linux, and a Mac allows 256 by default:
    raise the process's own limit as far as the system lets it. Windows has no such limit."""
    try:
        import resource
    except ImportError:
        return
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    for want in (n, 10240, 4096):                                           # 10240 = macOS's OPEN_MAX
        want = want if hard == resource.RLIM_INFINITY else min(want, hard)
        if soft >= want:
            return
        try:
            resource.setrlimit(resource.RLIMIT_NOFILE, (want, hard))
            return
        except (ValueError, OSError):
            continue


# ── Network ──────────────────────────────────────────────────────────────────

def cbr(cin, cout, k, s=1, p=0):
    return nn.Sequential(nn.Conv2d(cin, cout, k, s, p, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class Pulse(nn.Module):
    def __init__(self, width=32, levels=5):
        super().__init__()
        w = width
        ch = [16, 16, 24, 24, 32, w]
        stem, c = [cbr(1, ch[0], 3, 1, 1)], ch[0]
        for o in ch:                                                        # 128 -> 64 -> 32 -> 16 -> 8 -> 4 -> 2 bands
            stem.append(cbr(c, o, (3, 3), (2, 1), (1, 1)))
            c = o
        stem.append(cbr(c, w, (2, 1)))                                      # 2 bands -> 1
        self.stem = nn.Sequential(*stem)
        self.levels = levels
        self.down = nn.ModuleList([cbr(w, w, (1, 5), (1, 2), (0, 2)) for _ in range(levels)])
        self.mid = nn.ModuleList([cbr(w, w, (1, 5), 1, (0, 2)) for _ in range(levels)])
        self.up = nn.ModuleList([cbr(w, w, (1, 5), 1, (0, 2)) for _ in range(levels)])
        self.head = nn.Conv2d(w, 2, 1)

    def forward(self, x):                                                   # (B, 1, 128, T) -> (B, 2, T)
        h = self.stem(x)
        skips = []
        for d, m in zip(self.down, self.mid):
            skips.append(h)
            h = m(d(h))
        for u in self.up:
            h = F.interpolate(h, scale_factor=(1, 2), mode="nearest")
            h = u(h + skips.pop())
        return self.head(h)[:, :, 0]


def count_params(m):
    return sum(p.numel() for p in m.parameters())


# ── Data ─────────────────────────────────────────────────────────────────────

def mel_path(teacher: Path, tid: str) -> Path:
    return teacher / "mels" / f"{tid}.npy"


def make_mels(teacher: Path, dev):
    """Log-mels with Beat This!'s own front end, float16 (128, T), next to the teacher labels."""
    import soundfile as sf
    import soxr
    from beat_this.preprocessing import LogMelSpect
    index = json.loads((teacher / "index.json").read_text())
    spect = LogMelSpect(device=dev)
    (teacher / "mels").mkdir(exist_ok=True)
    t0 = time.time()
    for i, (tid, v) in enumerate(index.items()):
        if "error" in v or mel_path(teacher, tid).exists():
            continue
        x, sr = sf.read(v["source"], dtype="float32", always_2d=True)
        x = x.mean(1)
        if sr != 22050:
            x = soxr.resample(x, sr, 22050)
        with torch.inference_mode():
            m = spect(torch.from_numpy(x).to(dev)).T.half().cpu().numpy()  # (128, T)
        np.save(mel_path(teacher, tid), m)
        if i % 200 == 0:
            print(f"mels {i}/{len(index)} {time.time() - t0:.0f} s", flush=True)


class Tracks:
    def __init__(self, teacher: Path, split: str):
        index = json.loads((teacher / "index.json").read_text())
        hold = set((teacher / "holdout.txt").read_text(encoding="utf-8").split())
        # an index entry may point at its mel elsewhere ("mel", an absolute path on the PC): the ensemble teacher
        # reuses round 1's mels
        where = {t: local_path(v["mel"]) if "mel" in v else mel_path(teacher, t) for t, v in index.items() if "error" not in v}
        ids = [t for t, p in where.items() if p.exists()]
        rng = random.Random(0)
        rest = sorted(t for t in ids if t not in hold)
        rng.shuffle(rest)
        n_val = 60
        self.ids = {"train": rest[n_val:], "val": rest[:n_val], "holdout": sorted(t for t in ids if t in hold)}[split]
        self.teacher = teacher
        allow_open_files()
        self.mels = {t: np.load(where[t], mmap_mode="r") for t in self.ids}
        self.lab = {}
        for t in self.ids:
            z = np.load(teacher / f"{t}.npz")
            self.lab[t] = (z["beat"].astype(np.float32), z["down"].astype(np.float32), z["beats"], z["downbeats"])

    def batch(self, n, rng, augment=True):
        """Random 512-frame crops. Augmentation: tempo +-8 % (resample the time axis of mels and targets
        together), pitch +-2 mel bands, gain -12..+6 dB (an offset in log-mel), band-limited EQ tilt."""
        xs, ys = [], []
        while len(xs) < n:
            t = rng.choice(self.ids)
            m, (b, d, _, _) = self.mels[t], self.lab[t]
            L = min(m.shape[1], len(b))
            r = rng.uniform(0.92, 1.08) if augment else 1.0
            need = int(math.ceil(T_CHUNK * r)) + 2
            if L < need:
                continue
            a = rng.randrange(0, L - need)
            seg = torch.from_numpy(np.asarray(m[:, a:a + need], dtype=np.float32))
            tgt = torch.from_numpy(np.stack([b[a:a + need], d[a:a + need]]))
            if r != 1.0:
                pos = torch.arange(T_CHUNK) * r                             # resample: frame i of the crop = source frame i*r
                i0 = pos.floor().long().clamp(max=need - 2)
                f = (pos - i0).unsqueeze(0)
                seg = seg[:, i0] * (1 - f) + seg[:, i0 + 1] * f
                # logits of a peak must survive resampling: take the max of the two neighbours (soft targets)
                tgt = torch.maximum(tgt[:, i0], tgt[:, i0 + 1])
            else:
                seg, tgt = seg[:, :T_CHUNK], tgt[:, :T_CHUNK]
            if augment:
                sh = rng.randint(-2, 2)
                seg = torch.roll(seg, sh, 0)
                if sh > 0:
                    seg[:sh] = seg[sh]
                elif sh < 0:
                    seg[sh:] = seg[sh - 1]
                seg = seg + rng.uniform(-1.2, 0.6) + torch.linspace(-1, 1, MELS)[:, None] * rng.uniform(-0.4, 0.4)
            xs.append(seg)
            ys.append(tgt)
        return torch.stack(xs)[:, None], torch.stack(ys)


# ── Evaluation ───────────────────────────────────────────────────────────────

@torch.no_grad()
def predict_track(model, mel, device, hop=448):
    """Whole-track logits from overlapping 512-frame chunks, crossfaded (as on the deck). mel (128, N)."""
    n = mel.shape[1]
    pad = (-n) % 64 + T_CHUNK
    x = F.pad(torch.from_numpy(np.asarray(mel, dtype=np.float32)), (32, pad))
    out = torch.zeros(2, x.shape[1])
    wsum = torch.zeros(x.shape[1])
    w = torch.ones(T_CHUNK)
    ov = T_CHUNK - hop
    w[:ov] = torch.linspace(0, 1, ov + 2)[1:-1]
    w[-ov:] = torch.linspace(1, 0, ov + 2)[1:-1]
    starts = list(range(0, x.shape[1] - T_CHUNK + 1, hop))
    for i in range(0, len(starts), 32):
        g = starts[i:i + 32]
        y = model(torch.stack([x[:, a:a + T_CHUNK] for a in g])[:, None].to(device)).float().cpu()
        for yy, a in zip(y, g):
            out[:, a:a + T_CHUNK] += yy * w
            wsum[a:a + T_CHUNK] += w
    return (out / wsum.clamp(min=1e-6))[:, 32:32 + n]


def peaks(logit, thr=0.0, min_gap=0.07):
    """Beat times (s): local maxima of the logit above thr (sigmoid 0.5), at least min_gap apart."""
    l = logit.numpy()
    p = np.where((l[1:-1] > thr) & (l[1:-1] >= l[:-2]) & (l[1:-1] >= l[2:]))[0] + 1
    keep, last = [], -1e9
    for i in p:
        if (i - last) / FPS >= min_gap:
            keep.append(i)
            last = i
    return np.array(keep) / FPS


def f_measure(ref, est, tol=0.07):
    if len(ref) == 0 or len(est) == 0:
        return 0.0
    used, hit = np.zeros(len(est), bool), 0
    for r in ref:
        j = np.argmin(np.abs(est - r))
        if not used[j] and abs(est[j] - r) <= tol:
            used[j] = True
            hit += 1
    p, rc = hit / len(est), hit / len(ref)
    return 0 if hit == 0 else 2 * p * rc / (p + rc)


def evaluate(model, data: Tracks, device):
    model.eval()
    fb, fd = [], []
    for t in data.ids:
        y = predict_track(model, data.mels[t], device)
        _, _, beats, downs = data.lab[t]
        fb.append(f_measure(beats, peaks(y[0])))
        fd.append(f_measure(downs, peaks(y[1], min_gap=0.3)))
    model.train()
    return float(np.mean(fb)), float(np.mean(fd))


# ── Training ─────────────────────────────────────────────────────────────────

def train(a):
    teacher = local_path(a.teacher)
    dev = pick_device(a.device)
    limit_gpu_memory(dev, a.gpu_mem)
    tr, va = Tracks(teacher, "train"), Tracks(teacher, "val")
    model = Pulse(a.width).to(dev)
    print(f"Pulse width {a.width} on {dev}: {count_params(model):,} parameters; {len(tr.ids)} train, {len(va.ids)} val tracks", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.steps, pct_start=0.05)
    run = Path("runs_pulse") / a.name
    run.mkdir(parents=True, exist_ok=True)
    rng = random.Random(1)
    best, t0 = -1.0, time.time()
    for step in range(1, a.steps + 1):
        x, y = tr.batch(a.batch, rng)
        x, y = x.to(dev), y.to(dev)
        out = model(x)
        # distillation: BCE against the teacher's probabilities; beats are sparse, so frames near a beat
        # count more (the teacher's own probability, smoothed, as the weight)
        p = torch.sigmoid(y)
        wgt = 1 + 4 * F.max_pool1d(p, 5, 1, 2)
        loss = (F.binary_cross_entropy_with_logits(out, p, reduction="none") * wgt).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % 200 == 0:
            print(f"step {step:6d} loss {loss.item():.4f} {step / (time.time() - t0):.1f} steps/s", flush=True)
        if step % a.val_every == 0 or step == a.steps:
            fb, fd = evaluate(model, va, dev)
            score = fb + 0.5 * fd
            print(f"validation at step {step}: beat F {fb:.4f}, downbeat F {fd:.4f}", flush=True)
            ck = {"model": model.state_dict(), "width": a.width, "step": step, "beatF": fb, "downF": fd}
            torch.save(ck, run / "last.pt")
            if score > best:
                best = score
                torch.save(ck, run / "best.pt")
    print("done, best", best)


def evaluate_cli(a):
    teacher = local_path(a.teacher)
    dev = pick_device(a.device)
    ck = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
    model = Pulse(ck["width"], ck.get("levels", 5))
    model.load_state_dict(ck["model"])
    model.to(dev)
    ho = Tracks(teacher, "holdout")
    fb, fd = evaluate(model, ho, dev)
    print(f"hold-out ({len(ho.ids)} tracks): beat F {fb:.4f}, downbeat F {fd:.4f} vs Beat This! final0; "
          f"{count_params(model):,} parameters")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["mels", "train", "eval"])
    ap.add_argument("--teacher", default=str(DATA / "beat_teacher"))
    ap.add_argument("--name", default="p1")
    ap.add_argument("--width", type=int, default=32)
    ap.add_argument("--steps", type=int, default=60000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--val-every", type=int, default=5000)
    ap.add_argument("--gpu-mem", type=float, default=0.3, help="cuda only")
    ap.add_argument("--device", help="cuda, mps or cpu (default: the first of those that is there)")
    ap.add_argument("--checkpoint")
    a = ap.parse_args()
    {"mels": lambda: make_mels(local_path(a.teacher), pick_device(a.device)), "train": lambda: train(a),
     "eval": lambda: evaluate_cli(a)}[a.cmd]()
