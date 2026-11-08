"""A stronger teacher: RoFormer vocals + htdemucs_ft for the rest (the chain UVR users use).

htdemucs_ft is weakest on vocals. On the audio-separator benchmark set, the MelBand RoFormer vocal
model scores 12.6 dB on vocals against htdemucs_ft's 10.8. So per track:
  1. vocals       = MelBand RoFormer (Kim) on the mix,
  2. instrumental = mix - vocals,
  3. drums, bass  = htdemucs_ft (2 shifts) on the instrumental (no vocals left to bleed into them),
  4. other        = instrumental - drums - bass (whatever htdemucs called vocals or other).
The four stems add up to the mix exactly, like distill.py's.

It re-does the tracks of an existing distill.py cache (same folder names, read from source.txt),
so held-out splits stay the same:

    F:\\needle\\.venv-sep\\Scripts\\python teacher2.py --like E:\\needle-teacher\\luke --out E:\\needle-teacher\\luke_v2

Runs in its own environment (.venv-sep: torch cu128 + audio-separator), resumable (`done` files).
audio-separator normalises audio that peaks above a threshold; that would change the stems' level
relative to the mix, so the mix is scaled to peak 0.5 here (so no stem reaches the 1.0 threshold), and the
stems are scaled back.
"""
import argparse
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np
import soundfile as sf

SR = 44100
VOCAL_MODEL = "vocals_mel_band_roformer.ckpt"
DEMUCS_MODEL = "htdemucs_ft.yaml"
AUDIO = {".mp3", ".flac", ".wav", ".m4a", ".aiff", ".aif", ".ogg", ".opus"}


def decode(path):
    """Any audio file -> float32 (N, 2) at 44.1 kHz (ffmpeg for anything soundfile can't read)."""
    try:
        wav, sr = sf.read(str(path), dtype="float32", always_2d=True)
        if sr == SR:
            return wav[:, :2] if wav.shape[1] >= 2 else np.repeat(wav, 2, 1)
    except Exception:
        pass
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-f", "f32le", "-ac", "2", "-ar", str(SR), "-"],
                         capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.float32).reshape(-1, 2).copy()


def separator(model, out_dir, models_dir):
    from audio_separator.separator import Separator
    s = Separator(output_dir=str(out_dir), model_file_dir=str(models_dir), output_format="WAV",
                  normalization_threshold=1.0, use_soundfile=True,
                  demucs_params={"segment_size": "Default", "shifts": 2, "overlap": 0.25, "segments_enabled": True})
    s.load_model(model_filename=model)
    return s


def read_stem(path, n):
    wav, _ = sf.read(str(path), dtype="float32", always_2d=True)
    out = np.zeros((n, 2), np.float32)
    m = min(n, len(wav))
    out[:m] = wav[:m, :2]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--like", help="existing distill.py cache whose tracks to redo")
    ap.add_argument("--input", help="instead of --like: a music folder")
    ap.add_argument("--glob", default="**/*", help="with --input: which files (e.g. '**/*.flac')")
    ap.add_argument("--every", type=int, default=1, help="with --input: take every Nth file (sorted), e.g. 5 for a test subset")
    ap.add_argument("--out", required=True)
    ap.add_argument("--models", default=r"F:\needle\sep-models")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--max-seconds", type=float, help="only the first N seconds of each track (for testing)")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    jobs = []
    if a.input:
        from distill import track_id
        files = sorted(p for p in Path(a.input).glob(a.glob) if p.suffix.lower() in AUDIO)[::a.every]
        jobs = [(track_id(str(p.relative_to(a.input))), p) for p in files]
    else:
        for d in sorted(Path(a.like).glob("*/done")):
            d = d.parent
            jobs.append((d.name, Path((d / "source.txt").read_text(encoding="utf-8").splitlines()[0])))
    jobs = [(n, p) for n, p in jobs if not (out / n / "done").exists()]
    jobs = jobs[:a.limit] if a.limit else jobs
    print(f"{len(jobs)} tracks to do -> {out}", flush=True)
    tmp = Path(tempfile.mkdtemp(prefix="teacher2_"))
    voc_sep, dem_sep = separator(VOCAL_MODEL, tmp, a.models), separator(DEMUCS_MODEL, tmp, a.models)
    for i, (name, src) in enumerate(jobs):
        t = time.time()
        try:
            mix = decode(src)
            if a.max_seconds:
                mix = mix[:int(a.max_seconds * SR)]
            n = len(mix)
            scale = 0.5 / max(float(np.abs(mix).max()), 1e-9)          # peak 0.5: no stem reaches 1.0, so nothing is renormalised
            sf.write(tmp / "mix.wav", mix * scale, SR, subtype="FLOAT")
            voc_sep.output_single_stem = "Vocals"
            voc_sep.separate(str(tmp / "mix.wav"), custom_output_names={"Vocals": "vocals"})
            vocals = read_stem(tmp / "vocals.wav", n)
            inst = mix * scale - vocals
            sf.write(tmp / "inst.wav", inst, SR, subtype="FLOAT")
            dem_sep.separate(str(tmp / "inst.wav"), custom_output_names={"Drums": "drums", "Bass": "bass",
                                                                       "Other": "dm_other", "Vocals": "dm_vocals"})
            drums, bass = read_stem(tmp / "drums.wav", n), read_stem(tmp / "bass.wav", n)
            other = inst - drums - bass
            d = out / name
            d.mkdir(exist_ok=True)
            for stem, wav in (("drums", drums), ("bass", bass), ("other", other), ("vocals", vocals)):
                sf.write(d / f"{stem}.flac", np.clip(wav / scale, -1, 1 - 1 / 32768), SR, subtype="PCM_16")
            (d / "source.txt").write_text(f"{src}\nteacher\t{VOCAL_MODEL} + {DEMUCS_MODEL} on instrumental\n",
                                          encoding="utf-8")
            (d / "done").touch()
            print(f"[{i + 1}/{len(jobs)}] {name}  {n / SR:.0f} s in {time.time() - t:.0f} s", flush=True)
        except Exception as e:                                   # one bad file must not stop the run
            print(f"[{i + 1}/{len(jobs)}] FAILED {name}: {e!r}", flush=True)
            with open(out / "failed.tsv", "a", encoding="utf-8") as f:
                f.write(f"{name}\t{src}\t{e!r}\n")
        for f in tmp.glob("*.wav"):
            f.unlink()
    shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
