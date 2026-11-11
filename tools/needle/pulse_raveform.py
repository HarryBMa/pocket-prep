r"""Pulse vs Beat This! against HUMAN beat annotations (Raveform), not against the teacher.

Raveform annotated 1,423 tracks by hand (beat times + bar position, on the YouTube audio). The audio in
I:\datasets\raveform\downloads is those YouTube versions, so the annotations line up with it. Files were renamed
(the YouTube id dropped), so each file is matched to its annotation by title.

Scores (F-measure, +-70 ms, the usual beat-tracking tolerance):
  beat F, downbeat F                 per model, mean over tracks
  tempo family right                 median inter-beat interval within 4 % of the human one
                                     (catches half / double / 3:2 errors, which F alone hides)

    .venv\Scripts\python.exe pulse_raveform.py  -> F:\needle\pulse_raveform.json + summary
"""
import csv
import difflib
import io
import json
import re
import sys
import zipfile
from pathlib import Path

import numpy as np
import soundfile as sf
import soxr
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from pulse import Pulse, f_measure, peaks, predict_track     # noqa: E402

ZIP = Path(r"F:\needle\datasets\raveform\raveform.zip")
AUDIO = Path(r"I:\datasets\raveform\downloads")
CKPT = Path(r"F:\needle\beat_this\final0.ckpt")
RUNS = {"pulse_w32": HERE / "runs_pulse" / "p1_w32" / "best.pt", "pulse_w24": HERE / "runs_pulse" / "p1_w24" / "best.pt"}
OUT = Path(r"F:\needle\pulse_raveform.json")


def norm(s):
    s = re.sub(r"\[[^\]]*\]", " ", s.lower())                       # [label], [youtube id]
    s = re.sub(r"\((original|extended)( mix)?\)", " ", s)
    return " ".join(re.sub(r"[^a-z0-9]+", " ", s).split())


def load_annotations():
    z = zipfile.ZipFile(ZIP)
    titles = {}
    for line in io.TextIOWrapper(z.open("raveform/tracks.jsonl"), encoding="utf-8"):
        t = json.loads(line)
        titles[t["id"]] = t["title"]
    ann = {}
    for n in z.namelist():
        m = re.match(r"raveform/structures/beats/\d+\.(.+)\.beat\.csv$", n)
        if not m:
            continue
        rows = list(csv.DictReader(io.TextIOWrapper(z.open(n), encoding="utf-8")))
        b = np.array([float(r["time"]) for r in rows])
        d = np.array([float(r["time"]) for r in rows if r["downbeat"].strip() == "1"])
        ann[m.group(1)] = (titles.get(m.group(1), ""), b, d)
    return ann


def match(files, ann):
    """file -> annotation id: the id in the file name if it is still there, else the closest title (>= 0.85)."""
    by_norm = {norm(v[0]): k for k, v in ann.items()}
    keys = list(by_norm)
    out = {}
    for f in files:
        m = re.search(r"\[([A-Za-z0-9_-]{11})\]", f.stem)
        if m and m.group(1) in ann:
            out[f] = m.group(1)
            continue
        c = difflib.get_close_matches(norm(f.stem), keys, n=1, cutoff=0.85)
        if c:
            out[f] = by_norm[c[0]]
    return out


def tempo_ok(ref, est):
    if len(ref) < 4 or len(est) < 4:
        return False
    return bool(abs(np.median(np.diff(est)) / np.median(np.diff(ref)) - 1) < 0.04)


def main():
    from beat_this.inference import Audio2Frames
    from beat_this.model.postprocessor import Postprocessor
    from beat_this.preprocessing import LogMelSpect
    ann = load_annotations()
    files = sorted(p for p in AUDIO.rglob("*") if p.suffix.lower() in {".flac", ".wav", ".mp3", ".m4a", ".opus"})
    pairs = match(files, ann)
    print(f"{len(ann)} annotated tracks, {len(files)} audio files, {len(pairs)} matched", flush=True)
    spect = LogMelSpect(device="cuda")
    a2f = Audio2Frames(checkpoint_path=str(CKPT), device="cuda", float16=False)
    post = Postprocessor(type="minimal")
    models = {}
    for k, p in RUNS.items():
        ck = torch.load(p, map_location="cpu", weights_only=False)
        m = Pulse(ck["width"])
        m.load_state_dict(ck["model"])
        models[k] = m.to("cuda").eval()
    res = {}
    for i, (f, tid) in enumerate(pairs.items()):
        title, rb, rd = ann[tid]
        try:
            x, sr = sf.read(str(f), dtype="float32", always_2d=True)
        except Exception as ex:                                         # noqa: BLE001
            print("skip", f.name, ex)
            continue
        x = x.mean(1)
        row = {"file": f.name, "id": tid, "title": title}
        with torch.inference_mode():
            bl, dl = a2f(x, sr)
            bb, dd = post(bl, dl)
            row["beat_this"] = [f_measure(rb, np.asarray(bb)), f_measure(rd, np.asarray(dd)), tempo_ok(rb, np.asarray(bb))]
            x22 = soxr.resample(x, sr, 22050) if sr != 22050 else x
            mel = spect(torch.from_numpy(x22).cuda()).T.float().cpu().numpy()
            for k, m in models.items():
                y = predict_track(m, mel, "cuda")
                eb, ed = peaks(y[0]), peaks(y[1], min_gap=0.3)
                row[k] = [f_measure(rb, eb), f_measure(rd, ed), tempo_ok(rb, eb)]
        res[str(f)] = row
        if i % 25 == 0:
            print(f"[{i + 1}/{len(pairs)}] {f.name[:50]}  BT {row['beat_this'][0]:.3f}  w32 {row['pulse_w32'][0]:.3f}", flush=True)
    OUT.write_text(json.dumps(res, indent=1))
    summarize(res)


def summarize(res):
    rows = list(res.values())
    # a wrong file match (other version / edit) shows as near-zero F for the teacher too: leave those out
    good = [r for r in rows if r["beat_this"][0] > 0.3]
    print(f"\n{len(good)} of {len(rows)} matched tracks usable (Beat This! beat F > 0.3; the rest are likely other versions)")
    for k in ("beat_this", "pulse_w32", "pulse_w24"):
        a = np.array([r[k][:2] for r in good])
        t = np.mean([r[k][2] for r in good]) * 100
        print(f"{k:10s} beat F {a[:, 0].mean():.3f} (median {np.median(a[:, 0]):.3f})  downbeat F {a[:, 1].mean():.3f}  tempo family right {t:.0f}%")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "summary":
        summarize(json.loads(OUT.read_text()))
    else:
        main()
