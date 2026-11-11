r"""Beat This! (final0, MIT) as the teacher for a tiny on-device beat tracker, the same recipe as Needle.

For every track, writes <out>/<id>.npz with the teacher's frame logits at 50 fps (beat, downbeat; float16)
and its post-processed beat and downbeat times. These are the student's training targets.

    .venv\Scripts\python.exe beat_teacher.py --out F:\needle\beat_teacher [--limit N]

Sources: all Jamendo tracks on E:, the Ibiza compilations on H:. The 50 hold-out house tracks are labelled
too (for evaluation) but listed in holdout.txt so training can skip them.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from beat_this.inference import Audio2Frames
from beat_this.model.postprocessor import Postprocessor

ROOTS = [Path(r"E:\needle-music\jamendo"), Path(r"H:\\")]
AUDIO = {".flac", ".mp3", ".wav", ".ogg"}
CKPT = Path(r"F:\needle\beat_this\final0.ckpt")
HOLDOUT = Path(r"E:\needle-teacher\holdout_house\jamendo_v2")


def track_id(p: Path) -> str:
    return p.stem[:60].replace(" ", "_") + "-" + hashlib.sha1(str(p).encode()).hexdigest()[:8]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=r"F:\needle\beat_teacher")
    ap.add_argument("--limit", type=int)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    files = sorted(p for p in ROOTS[0].rglob("*") if p.suffix.lower() in AUDIO)
    files += sorted(p for d in ROOTS[1].glob("VA-*") for p in d.rglob("*") if p.suffix.lower() in AUDIO)
    hold_src = {Path((d / "source.txt").read_text(encoding="utf-8").splitlines()[0]) for d in HOLDOUT.iterdir() if (d / "source.txt").exists()}
    (out / "holdout.txt").write_text("\n".join(track_id(p) for p in files if p in hold_src), encoding="utf-8")
    a2f = Audio2Frames(checkpoint_path=str(CKPT), device="cuda", float16=False)
    post = Postprocessor(type="minimal")
    index = json.loads((out / "index.json").read_text()) if (out / "index.json").exists() else {}
    todo = [p for p in files if track_id(p) not in index][: a.limit]
    print(len(files), "tracks,", len(todo), "to do,", len(hold_src), "hold-out", flush=True)
    t0 = time.time()
    for i, p in enumerate(todo):
        tid = track_id(p)
        try:
            x, sr = sf.read(str(p), dtype="float32", always_2d=True)
            with torch.inference_mode():
                beat, down = a2f(x.mean(1), sr)
                beats, downs = post(beat, down)
            np.savez_compressed(out / f"{tid}.npz", beat=beat.cpu().numpy().astype(np.float16),
                                down=down.cpu().numpy().astype(np.float16), beats=np.asarray(beats, np.float32),
                                downbeats=np.asarray(downs, np.float32))
            index[tid] = {"source": str(p), "seconds": round(len(x) / sr, 2), "beats": int(len(beats))}
        except Exception as ex:                                             # noqa: BLE001
            index[tid] = {"source": str(p), "error": str(ex)[:200]}
        if i % 50 == 0 or i == len(todo) - 1:
            (out / "index.json").write_text(json.dumps(index, indent=1))
            print(f"[{i + 1}/{len(todo)}] {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
