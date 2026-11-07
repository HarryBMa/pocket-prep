"""Needle Tags, part 1: an embedding for every track, taken from the Demucs the app already runs.

Like Essentia's Discogs-EffNet, this is a frozen network turned into a fixed-size vector; a small head
(mood_head.py) is then trained on top. The network is htdemucs (MIT licence, Meta). Only its four convolutional
encoders are needed (the transformer is 75 % of the weights and did not help enough, see mood_layers.py): the
outputs of encoder layers 2 and 3, both branches, pooled over time as mean and standard deviation, 2,304 numbers.
The app runs the same thing from a 12 MB ONNX file (tools/export_demucs_onnx.py --embedding).

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
def embed(model, grabbed, wav, dev):
    seg = int(7.8 * SR)
    feats = []
    for i in range(0, wav.shape[1] - seg + 1, seg):
        x = torch.from_numpy(wav[:, i:i + seg])[None].to(dev)
        ref = x.mean(0)
        x = (x - ref.mean()) / (ref.std() + 1e-8)                       # as demucs does before the network
        grabbed.clear()
        model(x)
        row = []
        for k in (2, 3):
            f, t = grabbed[f"f{k}"], grabbed[f"t{k}"]                     # (1, C, F, T1), (1, C, T2)
            row += [f.mean((2, 3)), f.std((2, 3)), t.mean(2), t.std(2)]
        feats.append(torch.cat(row, 1)[0].cpu().numpy())
    return np.mean(feats, axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("list")
    ap.add_argument("out")
    ap.add_argument("--seconds", type=int, default=120)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = get_model("htdemucs").models[0].to(dev).eval()
    grabbed = {}
    for k in (2, 3):
        model.encoder[k].register_forward_hook(lambda m, i, o, k=k: grabbed.__setitem__(f"f{k}", o))
        model.tencoder[k].register_forward_hook(lambda m, i, o, k=k: grabbed.__setitem__(f"t{k}", o))
    labels, files, vecs = [], [], []
    rows = [l.rstrip("\n").split("\t") for l in open(a.list, encoding="utf-8") if "\t" in l]
    for n, (label, path) in enumerate(rows):
        try:
            wav = load(path, a.seconds)
            if wav.shape[1] < 10 * SR:
                raise ValueError("too short")
            vecs.append(embed(model, grabbed, wav, dev))
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
