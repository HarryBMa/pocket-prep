"""SDR of Needle, or of stems someone else made (e.g. the teacher), on MUSDB18-HQ with museval.

    # a trained checkpoint, on the 50 test tracks
    python eval.py --checkpoint runs\\b16_f32\\best.pt --musdb E:\\needle-data\\musdb18hq

    # stems that already exist on disk (distill.py output: <dir>/<track>/<stem>.flac)
    python eval.py --estimates E:\\needle-data\\teacher_musdb_test --name htdemucs_ft --musdb ...

    # the 14 validation tracks (use these for any decision; the test set is only for the final table)
    python eval.py --checkpoint ... --subset valid

How it measures, and why
- Separation is done like on the device (chunked.py): `frames`-frame chunks, 8 frames of overlap
  at each side, crossfaded. The input is MUSDB's real mixture.wav.
- museval (BSS Eval v4) computes SDR in 1-second windows. Per track we take the median over the
  windows (silent windows give NaN and are ignored), then the median over tracks, per stem, and
  the average of the four stems. That is how the SiSEC 2018 / MUSDB18 results are reported, so
  the numbers compare directly with papers (htdemucs_ft: about 9 dB average).
- museval is slow (about a minute per track on one core), so scoring runs in worker processes on
  the CPU while the GPU separates the next track.

Writes results/<name>_<subset>.json: per-track SDR/SIR/SAR and the summary.
"""
import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch

from data import SAMPLE_RATE, musdb_split, read_audio, read_stems
from needle import SOURCES

HERE = Path(__file__).parent


def track_folder(musdb_root, subset, name):
    return Path(musdb_root) / ("test" if subset == "test" else "train") / name


def score_track(refs, ests):
    """refs, ests: float32 (S, C, N). Returns {metric: {source: median over 1-s windows}}."""
    import museval
    n = min(refs.shape[-1], ests.shape[-1])
    r = np.ascontiguousarray(refs[..., :n].transpose(0, 2, 1))     # museval wants (sources, samples, channels)
    e = np.ascontiguousarray(ests[..., :n].transpose(0, 2, 1))
    sdr, isr, sir, sar = museval.evaluate(r, e, win=SAMPLE_RATE, hop=SAMPLE_RATE)
    return {k: {s: float(np.nanmedian(v[i])) for i, s in enumerate(SOURCES)}
            for k, v in (("SDR", sdr), ("SIR", sir), ("SAR", sar))}


def summarize(per_track):
    """Median over tracks per source, and the average of the four sources (the headline number)."""
    out = {}
    for metric in ("SDR", "SIR", "SAR"):
        med = {s: float(np.nanmedian([t[metric][s] for t in per_track.values()])) for s in SOURCES}
        med["avg"] = float(np.mean([med[s] for s in SOURCES]))
        out[metric] = med
    return out


def print_table(name, summary):
    sdr = summary["SDR"]
    print(f"{name}: SDR " + "  ".join(f"{s} {sdr[s]:.2f}" for s in SOURCES) + f"  | avg {sdr['avg']:.2f} dB")


def save_results(path, name, subset, per_track, extra=None):
    summary = summarize(per_track)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"name": name, "subset": subset, "tracks": len(per_track), **(extra or {}),
                                "summary": summary, "per_track": per_track}, indent=1))
    print_table(f"{name} ({subset}, {len(per_track)} tracks)", summary)
    print(f"-> {path}")
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    what = ap.add_mutually_exclusive_group(required=True)
    what.add_argument("--checkpoint", help="a train.py checkpoint (best.pt)")
    what.add_argument("--estimates", help="folder of <track>/<stem>.flac|.wav made by something else")
    ap.add_argument("--musdb", required=True)
    ap.add_argument("--subset", default="test", choices=["test", "valid"])
    ap.add_argument("--frames", type=int, help="chunk length (default: the checkpoint's)")
    ap.add_argument("--overlap", type=int, default=8)
    ap.add_argument("--name", help="results name (default: the run folder's name)")
    ap.add_argument("--workers", type=int, default=12, help="museval processes")
    ap.add_argument("--limit", type=int, default=0, help="only the first N tracks (quick check)")
    ap.add_argument("--out", default=str(HERE / "results"))
    a = ap.parse_args()

    names = musdb_split(a.musdb)[a.subset]
    if a.limit:
        names = names[:a.limit]
    extra = {}
    if a.checkpoint:
        from chunked import separate_chunked, torch_predictor
        from needle import Needle, NeedleConfig
        ck = torch.load(a.checkpoint, map_location="cpu")
        cfg = NeedleConfig(**{**ck["config"], **({"frames": a.frames} if a.frames else {})})
        model = Needle(cfg)
        model.load_state_dict(ck["model"])
        model.cuda().eval()
        predict = torch_predictor(model)
        name = a.name or Path(a.checkpoint).parent.name
        extra = {"checkpoint": str(a.checkpoint), "config": ck["config"], "frames": cfg.frames, "overlap": a.overlap,
                 "step": ck.get("step")}

        def estimate(n):
            mix = torch.from_numpy(read_audio(track_folder(a.musdb, a.subset, n) / "mixture.wav", dtype="float32"))
            return separate_chunked(predict, mix.cuda(), cfg, a.overlap).cpu().numpy()
    else:
        name = a.name or Path(a.estimates).name
        extra = {"estimates": str(a.estimates)}

        def estimate(n):
            d = Path(a.estimates) / n
            return read_stems(d, "flac" if (d / "drums.flac").exists() else "wav", dtype="float32")

    t0 = time.time()
    per_track = {}
    with ProcessPoolExecutor(a.workers) as pool:
        futures = {}
        for i, n in enumerate(names):
            refs = read_stems(track_folder(a.musdb, a.subset, n), "wav", dtype="float32")
            futures[n] = pool.submit(score_track, refs, estimate(n))
            print(f"separated {i + 1}/{len(names)}: {n}", flush=True)
        for n, f in futures.items():
            per_track[n] = f.result()
    print(f"evaluated {len(names)} tracks in {(time.time() - t0) / 60:.1f} min")
    save_results(Path(a.out) / f"{name}_{a.subset}.json", name, a.subset, per_track, extra)


if __name__ == "__main__":
    main()
