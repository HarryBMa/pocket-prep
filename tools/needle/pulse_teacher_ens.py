r"""Ensemble teacher for Pulse: Beat This! final0 + final1 + final2 (three trainings of the same model),
probabilities averaged. Averaging cancels each one's own mistakes, so the soft targets are cleaner.

Tracks: round 1's 3,138 (their mels are reused, not recomputed) plus the bought / added music in
I:\datasets\Beatport and I:\datasets\house and techno (mels computed here). Hold-out list kept.

    .venv\Scripts\python.exe pulse_teacher_ens.py       -> F:\needle\beat_teacher_ens (pulse.Tracks layout)
"""
import hashlib
import sys
import json
import shutil
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import soxr
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
from beat_this.inference import Spect2Frames
from beat_this.model.postprocessor import Postprocessor
from beat_this.preprocessing import LogMelSpect

OLD = Path(r"F:\needle\beat_teacher")
OUT = Path(r"F:\needle\beat_teacher_ens")
NEW = [Path(r"I:\datasets\Beatport"), Path(r"I:\datasets\house and techno")]
CKPTS = [Path(rf"F:\needle\beat_this\final{i}.ckpt") for i in range(3)]


def track_id(p: Path) -> str:
    return p.stem[:60].replace(" ", "_") + "-" + hashlib.sha1(str(p).encode()).hexdigest()[:8]


def main():
    (OUT / "mels").mkdir(parents=True, exist_ok=True)
    shutil.copy(OLD / "holdout.txt", OUT / "holdout.txt")
    old = json.loads((OLD / "index.json").read_text())
    jobs = [(t, v["source"], OLD / "mels" / f"{t}.npy") for t, v in old.items()
            if "error" not in v and (OLD / "mels" / f"{t}.npy").exists()]
    # the same song as a Raveform TEST track (another copy of it) stays out, so the test set stays unseen
    import difflib
    from pulse_raveform import norm
    rave = json.loads(Path(r"F:\needle\beat_rave\index.json").read_text())
    test_titles = [norm(rave[t]["title"]) for t in Path(r"F:\needle\beat_rave\test.txt").read_text().split() if t in rave]
    skipped = 0
    for root in NEW:
        for p in sorted(root.rglob("*")):
            if p.suffix.lower() in (".flac", ".wav", ".mp3", ".aiff", ".aif", ".m4a"):
                stem = norm(p.stem.split(". ", 1)[-1])                       # drop "01. " track numbers
                if difflib.get_close_matches(stem, test_titles, n=1, cutoff=0.8):
                    skipped += 1
                    continue
                t = track_id(p)
                jobs.append((t, str(p), OUT / "mels" / f"{t}.npy"))
    nets = [Spect2Frames(checkpoint_path=str(c), device="cuda", float16=False) for c in CKPTS]
    post = Postprocessor(type="minimal")
    spect = LogMelSpect(device="cuda")
    index = json.loads((OUT / "index.json").read_text()) if (OUT / "index.json").exists() else {}
    todo = [j for j in jobs if j[0] not in index]
    print(f"{len(jobs)} tracks ({len(jobs) - len(old)} new; {skipped} skipped as Raveform test songs), "
          f"{len(todo)} to do", flush=True)
    t0 = time.time()
    for i, (tid, src, melp) in enumerate(todo):
        try:
            if not melp.exists():
                x, sr = sf.read(src, dtype="float32", always_2d=True)
                x = x.mean(1)
                x = soxr.resample(x, sr, 22050) if sr != 22050 else x
                with torch.inference_mode():
                    np.save(melp, spect(torch.from_numpy(x).cuda()).T.half().cpu().numpy())
            mel = torch.from_numpy(np.load(melp).astype(np.float32).T).cuda()          # (T, 128)
            with torch.inference_mode():
                pb, pd = 0, 0
                for n in nets:
                    b, d = n(mel)
                    pb, pd = pb + torch.sigmoid(b) / len(nets), pd + torch.sigmoid(d) / len(nets)
                lb = torch.logit(pb.clamp(1e-4, 1 - 1e-4))
                ld = torch.logit(pd.clamp(1e-4, 1 - 1e-4))
                beats, downs = post(lb, ld)
            np.savez_compressed(OUT / f"{tid}.npz", beat=lb.cpu().numpy().astype(np.float16),
                                down=ld.cpu().numpy().astype(np.float16), beats=np.asarray(beats, np.float32),
                                downbeats=np.asarray(downs, np.float32))
            index[tid] = {"source": src, "mel": str(melp), "beats": int(len(beats))}
        except Exception as ex:                                                     # noqa: BLE001
            index[tid] = {"source": src, "error": str(ex)[:200]}
        if i % 200 == 0 or i == len(todo) - 1:
            (OUT / "index.json").write_text(json.dumps(index, indent=1))
            print(f"[{i + 1}/{len(todo)}] {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
