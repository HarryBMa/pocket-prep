"""Mood without a neural network: ordinary audio statistics + the same small head as mood_head.py.

Per track: a log-mel spectrogram (64 bands) summarised over time by mean, standard deviation and the 10th / 90th
percentile of every band (256 numbers), the same for the spectral flux of each band group (how much the sound moves),
and optionally the nine numbers DKPRP's analysis already measures. No model file; the head is a few KB.

Usage (Needle venv):  python mood_classic.py <list.tsv> <out.npz> [--wav-dir DIR] [--nine measure.jsonl ...]
  list.tsv: "label<TAB>path"; paths are decoded through ffmpeg (middle 120 s, 22.05 kHz mono)
"""
import argparse, json, subprocess, sys
import numpy as np
import torch, torchaudio

SR = 22050


def load(path, seconds=120):
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", path], capture_output=True, text=True)
    d = float(out.stdout.strip() or 0)
    raw = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{max(0.0, d / 2 - seconds / 2):.2f}", "-t", str(seconds), "-i", path,
                          "-f", "f32le", "-ac", "1", "-ar", str(SR), "-"], capture_output=True).stdout
    return torch.from_numpy(np.frombuffer(raw, dtype=np.float32).copy())


_mel = torchaudio.transforms.MelSpectrogram(SR, n_fft=1024, hop_length=512, n_mels=64, f_min=30, f_max=11000, power=2.0)


def features(x):
    m = torch.log10(_mel(x) + 1e-8)                       # [64, T]
    q = torch.quantile(m, torch.tensor([0.1, 0.9]), dim=1)  # [2, 64]
    flux = (m[:, 1:] - m[:, :-1]).clamp(min=0)            # rising energy per band: kicks, hats, builds
    groups = flux.reshape(8, 8, -1).mean(1)               # 8 band groups
    return torch.cat([m.mean(1), m.std(1), q[0], q[1], groups.mean(1), groups.std(1)]).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("list")
    ap.add_argument("out")
    ap.add_argument("--nine", nargs="*", default=[], help="mood_calibration JSON lines to take the nine numbers from (matched by file name)")
    a = ap.parse_args()
    nine = {}
    for p in a.nine:
        for l in open(p, encoding="utf-8"):
            if l.startswith("{"):
                r = json.loads(l)
                nine[r["file"]] = [r["features"].get(k) for k in ("bpm", "lufs", "bass", "bright", "drums", "melody", "breaks", "minor")]
    labels, files, vecs = [], [], []
    for n, l in enumerate(open(a.list, encoding="utf-8")):
        if "\t" not in l:
            continue
        label, path = l.rstrip("\n").split("\t")
        name = path.replace("\\", "/").rsplit("/", 1)[-1]
        try:
            f = features(load(path))
            if a.nine:
                f = np.concatenate([f, np.array([np.nan if v is None else v for v in nine.get(name, [np.nan] * 8)], dtype=np.float32)])
            vecs.append(f); labels.append(label); files.append(name)
        except Exception as e:
            print("skip", path, e, file=sys.stderr)
        if n % 20 == 0:
            print(n, flush=True)
    x = np.stack(vecs)
    x = np.where(np.isnan(x), np.nanmean(x, axis=0), x)     # a track without the nine numbers gets the average
    np.savez(a.out, x=np.concatenate([x, np.zeros((len(x), 8), np.float32)], 1), labels=np.array(labels), files=np.array(files))
    print("saved", a.out, x.shape)


if __name__ == "__main__":
    main()
