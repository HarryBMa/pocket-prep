"""Separate one of your own tracks with Needle (float and int8) and compare with the teacher's
(htdemucs_ft) stems, as short excerpts you can listen to side by side.

    .venv-export\\Scripts\\python demo_track.py "E:\\needle-music\\...\\track.mp3" ^
        --teacher-dir <distill.py output for this track> --checkpoint runs\\b16_f32\\best.pt ^
        --tflite build\\b16_f32\\needle_b16_f32\\needle_int8.tflite --out demo

There is no ground truth for a commercial track, so this is for listening (and for an SDR of
Needle against the teacher, which says how close Needle gets to Demucs, not to the truth).
Writes <out>/<stem> <n> <version>.mp3 and prints Needle-vs-teacher SDR per stem.
"""
import argparse
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

import eval_tflite
from chunked import separate_chunked, torch_predictor
from data import SAMPLE_RATE, read_stems
from distill import decode
from listening import busiest_window
from needle import SOURCES, Needle, NeedleConfig


def mp3(path, x):
    wav = path.with_suffix(".wav")
    sf.write(wav, np.clip(x, -1, 1).T, SAMPLE_RATE, subtype="PCM_16")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(wav), "-b:a", "192k", str(path)], check=True)
    wav.unlink()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("track")
    ap.add_argument("--teacher-dir", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--tflite")
    ap.add_argument("--seconds", type=int, default=45)
    ap.add_argument("--out", default="demo")
    a = ap.parse_args()
    torch.set_num_threads(8)

    mix = decode(a.track)                                           # (2, N) float
    teacher = read_stems(a.teacher_dir, "flac", dtype="float32")[..., :mix.shape[-1]]
    ck = torch.load(a.checkpoint, map_location="cpu")
    cfg = NeedleConfig(**ck["config"])
    model = Needle(cfg)
    model.load_state_dict(ck["model"])
    x = torch.from_numpy(mix.copy())
    versions = [("2 htdemucs_ft", teacher), ("3 needle float", separate_chunked(torch_predictor(model), x, cfg).numpy())]
    if a.tflite:
        tcfg = eval_tflite.load_config(Path(a.tflite))
        eval_tflite._init(a.tflite, tcfg, 8, False)
        versions.append(("4 needle int8", separate_chunked(eval_tflite._predict, x, tcfg).numpy()))

    start = busiest_window(teacher, a.seconds)
    cut = slice(start, start + a.seconds * SAMPLE_RATE)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    mp3(out / "0 mixture.mp3", mix[:, cut])
    for label, stems in versions:
        for s, name in enumerate(SOURCES):
            mp3(out / f"{name} {label}.mp3", stems[s][:, cut])
    for label, stems in versions[1:]:
        sdr = [10 * np.log10((teacher[s] ** 2).sum() / ((teacher[s] - stems[s]) ** 2).sum() + 1e-12) for s in range(4)]
        print(f"{label} vs htdemucs_ft, whole track: " + "  ".join(f"{n} {v:.2f}" for n, v in zip(SOURCES, sdr)) +
              f"  | avg {np.mean(sdr):.2f} dB")
    print(f"excerpt {start / SAMPLE_RATE:.0f}-{start / SAMPLE_RATE + a.seconds:.0f} s -> {out}")


if __name__ == "__main__":
    main()
