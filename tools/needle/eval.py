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
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path

import numpy as np
import torch

from data import SAMPLE_RATE, musdb_split, read_audio, read_stems
from needle import SOURCES

HERE = Path(__file__).parent


def track_folder(musdb_root, subset, name):
    return Path(musdb_root) / ("test" if subset == "test" else "train") / name


def track_list(a):
    """[(name, load references (S, C, N), load mixture (C, N) or None = sum of the references)], subset label."""
    if a.refs:
        dirs = sorted(p.parent for p in Path(a.refs).glob("*/done"))
        return [(d.name, (lambda d=d: read_stems(d, "flac", dtype="float32")), None) for d in dirs], Path(a.refs).name
    assert a.musdb, "give --musdb or --refs"
    folders = [(n, track_folder(a.musdb, a.subset, n)) for n in musdb_split(a.musdb)[a.subset]]
    return [(n, (lambda f=f: read_stems(f, "wav", dtype="float32")),
             (lambda f=f: read_audio(f / "mixture.wav", dtype="float32"))) for n, f in folders], a.subset


def score_track(refs, ests, inaudible_db=None):
    """refs, ests: float32 (S, C, N). Returns {metric: {source: median over 1-s windows}}.
    A stem that is silent for the whole song (an instrumental's vocals) has no SDR; museval
    refuses such references, so those stems are left out and reported as NaN (ignored in the
    median over songs)."""
    import museval
    n = min(refs.shape[-1], ests.shape[-1])
    live = [i for i in range(len(SOURCES)) if np.abs(refs[i, :, :n]).max() > 1e-4]
    r = np.ascontiguousarray(refs[live, :, :n].transpose(0, 2, 1))  # museval wants (sources, samples, channels)
    e = np.ascontiguousarray(ests[live, :, :n].transpose(0, 2, 1))
    sdr, isr, sir, sar = museval.evaluate(r, e, win=SAMPLE_RATE, hop=SAMPLE_RATE)
    if inaudible_db is not None:
        # Teacher stems are never exactly silent: an intro without vocals still has a -70 dB
        # residue in the teacher's vocal stem, and its SDR in that window is meaningless (huge
        # negative). Leave out windows where a stem is more than inaudible_db below the mix.
        w = sdr.shape[1]                                             # museval's 1-s windows
        win_power = lambda p: p[:, :w * SAMPLE_RATE].reshape(p.shape[0], w, SAMPLE_RATE).mean(-1)
        stem_e = win_power((r ** 2).sum(-1))                         # (live sources, windows)
        mix_e = win_power(((r.sum(0)) ** 2).sum(-1)[None])           # (1, windows)
        quiet = stem_e < mix_e * 10 ** (-inaudible_db / 10)
        for m in (sdr, sir, sar):
            m[quiet] = np.nan
    return {k: {s: float(np.nanmedian(v[live.index(i)])) if i in live else float("nan") for i, s in enumerate(SOURCES)}
            for k, v in (("SDR", sdr), ("SIR", sir), ("SAR", sar))}


def wiener_post(mix, est, iterations=1, n_fft=2048, hop=512, segment_s=60.0):
    """Multichannel Wiener filter (norbert, as Open-Unmix and Spleeter use at inference): refines
    the four stems by treating each as a Gaussian source with a per-bin spatial covariance and
    re-splitting the mixture accordingly. iterations=0 is the plain soft mask |S_j| / sum |S_k|
    on the mixture. Runs per 60-s segment (the deck would do the same per section) to keep
    memory small. mix (C, N), est (S, C, N) float32 -> (S, C, N)."""
    import norbert
    import numpy as np
    s, c, n = est.shape
    out = np.zeros_like(est)
    win = torch.hann_window(n_fft)
    seg = int(segment_s * SAMPLE_RATE)
    starts = list(range(0, n, seg))
    if len(starts) > 1 and n - starts[-1] < 4 * n_fft:            # a few-ms tail: fold it into the segment before
        starts.pop()
    for i, a in enumerate(starts):
        b = starts[i + 1] if i + 1 < len(starts) else n
        x = torch.stft(torch.from_numpy(np.ascontiguousarray(mix[:, a:b])), n_fft, hop, window=win, return_complex=True)
        v = torch.stft(torch.from_numpy(np.ascontiguousarray(est[:, :, a:b]).reshape(s * c, -1)), n_fft, hop,
                       window=win, return_complex=True).abs().reshape(s, c, *x.shape[1:])
        X = x.permute(2, 1, 0).numpy().astype(np.complex64)                      # (frames, bins, channels)
        V = v.permute(3, 2, 1, 0).numpy().astype(np.float32)                     # (frames, bins, channels, sources)
        Y = norbert.wiener(V, X, iterations=iterations, use_softmask=True)       # (frames, bins, channels, sources)
        y = torch.from_numpy(np.ascontiguousarray(Y.transpose(3, 2, 1, 0))).reshape(s * c, *x.shape[1:])
        out[:, :, a:b] = torch.istft(y, n_fft, hop, window=win, length=b - a).reshape(s, c, -1).numpy()
    return out


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
    ap.add_argument("--musdb", help="MUSDB18-HQ root (references and mixtures from --subset)")
    ap.add_argument("--subset", default="test", choices=["test", "valid"])
    ap.add_argument("--refs", help="instead of MUSDB: a folder of real stems (import_stems.py layout, e.g. "
                                   "stems\\ni_test); the mixture is the sum of the stems")
    ap.add_argument("--frames", type=int, help="chunk length (default: the checkpoint's)")
    ap.add_argument("--overlap", type=int, default=8)
    ap.add_argument("--skip-inaudible-db", type=float,
                    help="leave out 1-s windows where a reference stem is this many dB below the mix "
                         "(for teacher-made references, which are never exactly silent; e.g. 40)")
    ap.add_argument("--shifts", type=int, default=1, help="average this many shifted passes (chunked.separate_chunked)")
    ap.add_argument("--wiener", type=int, help="multichannel Wiener filter post-processing with this many EM "
                    "iterations (0 = soft mask only); needs the norbert package")
    ap.add_argument("--name", help="results name (default: the run folder's name)")
    ap.add_argument("--workers", type=int, default=4, help="museval processes (each needs up to ~6-10 GB for a long song)")
    ap.add_argument("--gpu-mem", type=float, default=0.25, help="cap on this process's share of GPU memory (only the network runs there)")
    ap.add_argument("--limit", type=int, default=0, help="only the first N tracks (quick check)")
    ap.add_argument("--out", default=str(HERE / "results"))
    a = ap.parse_args()

    tracks, subset = track_list(a)
    if a.limit:
        tracks = tracks[:a.limit]
    extra = {}
    if a.checkpoint:
        from chunked import separate_chunked, torch_predictor
        from needle import Needle, NeedleConfig
        ck = torch.load(a.checkpoint, map_location="cpu")
        cfg = NeedleConfig(**{**ck["config"], **({"frames": a.frames} if a.frames else {})})
        model = Needle(cfg)
        model.load_state_dict(ck["model"])
        model.cuda().eval()
        # Share the GPU politely (training may be running): on Windows, going over the card's
        # memory doesn't fail, it spills into system RAM and everything on the GPU crawls.
        torch.cuda.set_per_process_memory_fraction(a.gpu_mem)
        predict = torch_predictor(model)
        name = a.name or Path(a.checkpoint).parent.name
        extra = {"checkpoint": str(a.checkpoint), "config": ck["config"], "frames": cfg.frames, "overlap": a.overlap, "shifts": a.shifts, "wiener": a.wiener,
                 "step": ck.get("step")}

        torch.set_num_threads(4)               # the STFT side runs on the CPU, next to the museval workers

        def estimate(n, mix):
            # only the network runs on the GPU (a few hundred MB); the whole-song STFT, masks and
            # iSTFT stay in system RAM, so a long song can't fill the GPU next to a training run
            est = separate_chunked(predict, torch.from_numpy(mix), cfg, a.overlap, shifts=a.shifts).numpy()
            return wiener_post(mix, est, a.wiener) if a.wiener is not None else est
    else:
        name = a.name or Path(a.estimates).name
        extra = {"estimates": str(a.estimates)}

        def estimate(n, mix):
            d = Path(a.estimates) / n
            return read_stems(d, "flac" if (d / "drums.flac").exists() else "wav", dtype="float32")

    t0 = time.time()
    per_track, pending = {}, {}
    with ProcessPoolExecutor(a.workers) as pool:
        for i, (n, load_refs, load_mix) in enumerate(tracks):
            # Separation is much faster than museval, so wait for a free worker before separating
            # the next song: otherwise every song's stems pile up in RAM (several GB each).
            while len(pending) >= a.workers:
                done, _ = wait(pending, return_when=FIRST_COMPLETED)
                for f in done:
                    per_track[pending.pop(f)] = f.result()
            refs = load_refs()
            mix = load_mix() if load_mix else refs.sum(0)
            pending[pool.submit(score_track, refs, estimate(n, mix), a.skip_inaudible_db)] = n
            print(f"separated {i + 1}/{len(tracks)}: {n}", flush=True)
        for f in list(pending):
            per_track[pending.pop(f)] = f.result()
    per_track = dict(sorted(per_track.items()))
    print(f"evaluated {len(tracks)} tracks in {(time.time() - t0) / 60:.1f} min")
    save_results(Path(a.out) / f"{name}_{subset}.json", name, subset, per_track, extra)


if __name__ == "__main__":
    main()
