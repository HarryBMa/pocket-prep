"""Needle Tags, part 1: an embedding for every track, taken from the Demucs the app already runs.

Like Essentia's Discogs-EffNet, this is a frozen network turned into a fixed-size vector; a small head
(mood_head.py) is then trained on top. The network is htdemucs (MIT licence, Meta): the output of its
cross-transformer, where the time and the spectrum branch meet, pooled over time (mean and standard
deviation), plus how loud and how busy each separated stem is. The app makes the stems anyway, so the
same pass could give the embedding for free.

Usage (the Needle venv, which has torch + demucs):
  python mood_embed.py <list.tsv> <out.npz> [--seconds 120]
  list.tsv: one "label<TAB>path" per line (the label is kept, the files are only read)
"""
import argparse, subprocess, sys
import numpy as np
import torch
from demucs.pretrained import get_model

SR = 44100


def load(path, seconds, duration=None):
    """The middle `seconds` of the file as float32 stereo at 44.1 kHz (read through ffmpeg; the file is not touched)."""
    if duration is None:
        out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", path],
                             capture_output=True, text=True)
        duration = float(out.stdout.strip() or 0)
    start = max(0.0, duration / 2 - seconds / 2)
    raw = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{start:.2f}", "-t", str(seconds), "-i", path,
                          "-f", "f32le", "-ac", "2", "-ar", str(SR), "-"], capture_output=True).stdout
    x = np.frombuffer(raw, dtype=np.float32)
    return x[: len(x) // 2 * 2].reshape(-1, 2).T.copy()


@torch.no_grad()
def embed(model, hook, wav, dev):
    seg = int(7.8 * SR)
    chunks = [wav[:, i:i + seg] for i in range(0, wav.shape[1] - seg + 1, seg)]
    feats = []
    for c in chunks:
        x = torch.from_numpy(c)[None].to(dev)
        ref = x.mean(0)
        x = (x - ref.mean()) / (ref.std() + 1e-8)                       # as demucs does before the network
        hook.clear()
        y = model(x)                                                      # (1, 4 stems, 2, T)
        fq, tm = hook.out                                                 # (1, C, F, T1), (1, C, T2)
        v = [fq.mean((2, 3)), fq.std((2, 3)), tm.mean(2), tm.std(2)]
        # stems: level against the mix (dB) and share of 0.5 s windows that carry sound
        e = (y ** 2).mean((0, 2))                                         # (stems, T)
        mix = (x ** 2).mean() + 1e-10
        w = e[:, : e.shape[1] // (SR // 2) * (SR // 2)].reshape(4, -1, SR // 2).mean(2)
        level = 10 * torch.log10(e.mean(1) / mix + 1e-10)
        active = (w > w.max() * 1e-3).float().mean(1)
        feats.append(torch.cat([torch.cat(v, 1)[0], level, active]).cpu().numpy())
    return np.mean(feats, axis=0)


class Hook:
    def __init__(self, module):
        self.out = None
        module.register_forward_hook(lambda m, i, o: setattr(self, "out", o))

    def clear(self):
        self.out = None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("list")
    ap.add_argument("out")
    ap.add_argument("--seconds", type=int, default=120)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = get_model("htdemucs").models[0].to(dev).eval()
    hook = Hook(model.crosstransformer)
    labels, files, vecs = [], [], []
    rows = [l.rstrip("\n").split("\t") for l in open(a.list, encoding="utf-8") if "\t" in l]
    for n, (label, path) in enumerate(rows):
        try:
            wav = load(path, a.seconds)
            if wav.shape[1] < 10 * SR:
                raise ValueError("too short")
            vecs.append(embed(model, hook, wav, dev))
            labels.append(label)
            files.append(path.replace("\\", "/").rsplit("/", 1)[-1])
        except Exception as e:
            print("skip", path, e, file=sys.stderr)
        if n % 10 == 0:
            print(n + 1, len(rows), flush=True)
    np.savez(a.out, x=np.stack(vecs), labels=np.array(labels), files=np.array(files))
    print("saved", a.out, np.stack(vecs).shape)


if __name__ == "__main__":
    main()
