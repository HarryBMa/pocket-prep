"""Listening set: 30-second excerpts of MUSDB18-HQ test tracks as stems from every model, named so
that a file browser sorts them side by side for comparing by ear.

    .venv-export\\Scripts\\python listening.py --musdb E:\\needle-data\\musdb18hq ^
        --teacher E:\\needle-data\\teacher_musdb_test ^
        --model b16=runs\\b16_f32\\best.pt,build\\b16_f32\\needle_b16_f32 ^
        --model b8=runs\\b8_f32\\best.pt,build\\b8_f32\\needle_b8_f32

Output (git-ignored), e.g. listening/2 - BKS - Too Much/:
    vocals 0 mixture.flac            the input
    vocals 1 truth.flac              MUSDB's real vocal stem
    vocals 2 teacher htdemucs_ft.flac
    vocals 3 b16 float.flac          PyTorch, as trained
    vocals 4 b16 int16.flac          TFLite, 16-bit activations
    vocals 5 b16 int8.flac           TFLite, 8-bit activations (what fits in on-chip SRAM)
    ...the same for drums, bass, other

The excerpt is the 30 s of the track where all four stems are loudest together (so every stem
has something to listen to). Needle separates the excerpt plus 3 s of context on each side,
chunked exactly as on the device, and the context is cut off afterwards.

Run with .venv-export (it has both PyTorch and TensorFlow).
"""
import argparse
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

import eval_tflite
from chunked import separate_chunked, torch_predictor
from data import SAMPLE_RATE, musdb_split, read_audio, read_stems
from eval import track_folder
from needle import SOURCES, Needle, NeedleConfig

HERE = Path(__file__).parent


def busiest_window(stems, seconds):
    """stems (S, C, N) -> start sample of the `seconds`-long window whose quietest stem is
    loudest (searched in 1-s steps), so every stem has something to listen to."""
    s, _, n = stems.shape
    secs = n // SAMPLE_RATE
    if secs <= seconds:
        return 0
    power = (stems[..., :secs * SAMPLE_RATE] ** 2).mean(1).reshape(s, secs, SAMPLE_RATE).mean(-1)   # (S, secs)
    win = np.stack([np.convolve(p, np.ones(seconds), "valid") for p in power]).min(0)
    return int(win.argmax()) * SAMPLE_RATE


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--musdb", required=True)
    ap.add_argument("--teacher", help="distill.py output for the MUSDB test set")
    ap.add_argument("--model", action="append", default=[],
                    help="label=checkpoint.pt,export_folder (folder with needle_int8.tflite / needle_int16.tflite)")
    ap.add_argument("--tracks", nargs="*", help="test track names (default: 5 spread over the list)")
    ap.add_argument("--seconds", type=int, default=30)
    ap.add_argument("--context", type=float, default=3.0)
    ap.add_argument("--out", default=str(HERE / "listening"))
    a = ap.parse_args()
    torch.set_num_threads(8)

    names = a.tracks or musdb_split(a.musdb)["test"][::10][:5]
    ctx = int(a.context * SAMPLE_RATE)
    for k, name in enumerate(names, 1):
        folder = track_folder(a.musdb, "test", name)
        truth = read_stems(folder, "wav", dtype="float32")              # (S, C, N)
        mix = read_audio(folder / "mixture.wav", dtype="float32")
        start = busiest_window(truth, a.seconds)
        end = start + a.seconds * SAMPLE_RATE
        lo, hi = max(0, start - ctx), min(mix.shape[-1], end + ctx)
        cut = slice(start - lo, start - lo + (end - start))
        versions = [("1 truth", truth[..., start:end])]
        if a.teacher:
            versions.append(("2 teacher htdemucs_ft", read_stems(Path(a.teacher) / name, "flac", dtype="float32")[..., start:end]))
        x = torch.from_numpy(mix[:, lo:hi].copy())
        for spec in a.model:
            label, paths = spec.split("=", 1)
            ckpt, export_dir = paths.split(",")
            ck = torch.load(ckpt, map_location="cpu")
            cfg = NeedleConfig(**ck["config"])
            model = Needle(cfg)
            model.load_state_dict(ck["model"])
            versions.append((f"3 {label} float", separate_chunked(torch_predictor(model), x, cfg).numpy()[..., cut]))
            for order, mode in ((4, "int16"), (5, "int8")):
                tfl = Path(export_dir) / f"needle_{mode}.tflite"
                if tfl.exists():
                    tcfg = eval_tflite.load_config(tfl)
                    eval_tflite._init(str(tfl), tcfg, 8, False)
                    versions.append((f"{order} {label} {mode}", separate_chunked(eval_tflite._predict, x, tcfg).numpy()[..., cut]))
        d = Path(a.out) / f"{k} - {name}"
        d.mkdir(parents=True, exist_ok=True)
        for s, stem in enumerate(SOURCES):
            sf.write(d / f"{stem} 0 mixture.flac", mix[:, start:end].T, SAMPLE_RATE, subtype="PCM_16")
            for label, stems in versions:
                sf.write(d / f"{stem} {label}.flac", np.clip(stems[s], -1, 1).T, SAMPLE_RATE, subtype="PCM_16")
        print(f"{d}: {len(versions)} versions, {start / SAMPLE_RATE:.0f}-{end / SAMPLE_RATE:.0f} s")


if __name__ == "__main__":
    main()
