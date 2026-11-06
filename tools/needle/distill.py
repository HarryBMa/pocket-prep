"""Teacher stems for distillation: run Demucs `htdemucs_ft` on the GPU over a music folder (or over
MUSDB18-HQ mixtures) and save its four stems as 44.1 kHz 16-bit FLAC.

    python distill.py --input "D:\\Music" --out E:\\needle-data\\teacher
    python distill.py --musdb E:\\needle-data\\musdb18hq --subset test --out E:\\needle-data\\teacher_musdb_test

Why: MUSDB18-HQ has only 86 training songs. A big model (the teacher, ~9 dB SDR) can label any
music, so the tiny model (the student) can learn from hundreds of your own tracks, from the
same kind of music the deck will play. This is called knowledge distillation.

Output, per track: <out>/<id>/{drums,bass,other,vocals}.flac, source.txt (original path, length,
peak) and an empty file `done`, written last. Tracks with `done` are skipped, so running the same
command again resumes where it stopped (an interrupted track is simply redone). Tracks that
can't be read or are too short/long are listed in <out>/failed.tsv and skipped next time too.

The originals are only read, never changed. These stems are for private training: they live
outside the repository and must never be committed or uploaded.
"""
import argparse
import hashlib
import re
import subprocess
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from tqdm import tqdm

from data import SAMPLE_RATE, keep_awake, musdb_split, read_audio
from needle import SOURCES

AUDIO = {".wav", ".flac", ".mp3", ".aif", ".aiff", ".m4a", ".aac", ".ogg", ".opus", ".alac", ".wma"}


def decode(path):
    """Any audio file -> float32 (2, N) at 44.1 kHz. libsndfile reads wav/flac/mp3/ogg/aiff;
    anything else (m4a, aac, alac, ...) is decoded by ffmpeg."""
    try:
        return read_audio(path, dtype="float32")
    except Exception:
        out = subprocess.run(["ffmpeg", "-v", "error", "-nostdin", "-i", str(path), "-f", "f32le", "-ac", "2",
                              "-ar", str(SAMPLE_RATE), "-"], capture_output=True, check=True).stdout
        return np.frombuffer(out, np.float32).reshape(-1, 2).T.copy()


def track_id(rel: str):
    """A folder name that is readable, unique and stable for the same file."""
    slug = re.sub(r"[^\w\-]+", "_", Path(rel).with_suffix("").as_posix()).strip("_")[-80:]
    return f"{slug}-{hashlib.sha1(rel.encode('utf-8')).hexdigest()[:8]}"


@torch.no_grad()
def run_teacher(model, wav, device, shifts, overlap):
    """Demucs' own recipe (demucs.separate): normalise, apply in overlapping segments, undo."""
    from demucs.apply import apply_model
    x = torch.from_numpy(wav)
    ref = x.mean(0)
    mean, std = ref.mean(), ref.std() + 1e-8
    out = apply_model(model, ((x - mean) / std)[None], device=device, shifts=shifts, split=True,
                      overlap=overlap, progress=False)[0]
    return (out * std + mean).numpy()                          # (4, 2, N)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--input", help="music folder (searched recursively)")
    src.add_argument("--musdb", help="MUSDB18-HQ root: separate <subset>'s mixture.wav files")
    ap.add_argument("--subset", default="test", choices=["train", "valid", "test"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="htdemucs_ft")
    ap.add_argument("--shifts", type=int, default=1, help="Demucs time-shift averaging (1 = Demucs' default)")
    ap.add_argument("--overlap", type=float, default=0.25)
    ap.add_argument("--min-seconds", type=float, default=20)
    ap.add_argument("--max-minutes", type=float, default=15, help="skip longer files (e.g. hour-long DJ mixes)")
    ap.add_argument("--limit", type=int, default=0, help="only the first N tracks (for a test run)")
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if a.input:
        root = Path(a.input)
        files = sorted(p for p in root.rglob("*") if p.suffix.lower() in AUDIO and p.is_file())
        jobs = [(track_id(p.relative_to(root).as_posix()), p) for p in files]
    else:
        names = musdb_split(a.musdb)[a.subset]
        folder = "test" if a.subset == "test" else "train"
        jobs = [(n, Path(a.musdb) / folder / n / "mixture.wav") for n in names]
    if a.limit:
        jobs = jobs[:a.limit]

    failed_log = out / "failed.tsv"
    failed = {line.split("\t")[0] for line in failed_log.read_text(encoding="utf-8").splitlines()} if failed_log.exists() else set()
    todo = [(i, p) for i, p in jobs if not (out / i / "done").exists() and str(p) not in failed]
    print(f"{len(jobs)} tracks, {len(jobs) - len(todo)} already done or failed before, {len(todo)} to do -> {out}")
    if not todo:
        return

    keep_awake()
    device = "cuda"
    from demucs.pretrained import get_model
    model = get_model(a.model)
    assert list(model.sources) == list(SOURCES), model.sources
    model.to(device).eval()

    t0, audio_s, n_ok = time.time(), 0.0, 0
    for tid, path in tqdm(todo, unit="track"):
        try:
            wav = decode(path)
            secs = wav.shape[-1] / SAMPLE_RATE
            if secs < a.min_seconds or secs > a.max_minutes * 60:
                raise ValueError(f"length {secs / 60:.1f} min is outside {a.min_seconds:.0f} s .. {a.max_minutes:.0f} min")
            stems = run_teacher(model, wav, device, a.shifts, a.overlap)
            d = out / tid
            d.mkdir(exist_ok=True)
            for s, x in zip(SOURCES, stems):                    # write to a temp name, rename when complete
                tmp = d / f"{s}.part"
                sf.write(tmp, np.clip(x, -1, 1).T, SAMPLE_RATE, format="FLAC", subtype="PCM_16")
                tmp.replace(d / f"{s}.flac")
            (d / "source.txt").write_text(f"{path}\nseconds\t{secs:.2f}\npeak\t{np.abs(stems).max():.3f}\n", encoding="utf-8")
            (d / "done").touch()
            audio_s += secs
            n_ok += 1
        except Exception as e:                                  # one bad file must not stop a night's run
            msg = " ".join(str(e).split())[:300]
            with open(failed_log, "a", encoding="utf-8") as f:
                f.write(f"{path}\t{type(e).__name__}: {msg}\n")
            tqdm.write(f"failed: {path}: {msg}")
        finally:
            torch.cuda.empty_cache()
    el = time.time() - t0
    print(f"done: {n_ok} of {len(todo)} tracks ({audio_s / 3600:.1f} h of audio) in {el / 3600:.2f} h, "
          f"{audio_s / max(el, 1e-9):.0f}x real time" + (f"; failures listed in {failed_log}" if failed_log.exists() else ""))


if __name__ == "__main__":
    main()
