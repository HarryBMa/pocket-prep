"""SDR of the quantised models: run needle_int8.tflite / needle_int16.tflite (made by export_vela.py)
with the TFLite interpreter on the CPU, chunked exactly like eval.py, and score with museval.

    .venv-export\\Scripts\\python eval_tflite.py --tflite build\\needle_b16_f32\\needle_int8.tflite ^
        --musdb E:\\needle-data\\musdb18hq --float results\\b16_f32_test.json

Run it with the export environment (.venv-export, which has TensorFlow; the training one doesn't).

What the quantised model is: weights are 8-bit integers; activations (the feature maps between
layers) are 8-bit (int8 model) or 16-bit (int16 model). Each tensor has a scale and zero point,
real = scale x (integer - zero_point), fitted on calibration audio by export_vela.py. The input
spectrogram is quantised the same way before the first layer, and the output masks are turned
back into real numbers after the last. That is exactly what the deck does around the NPU, and
Vela keeps TFLite's integer arithmetic, so these scores are what to expect on the device.

Speed: every worker process owns one single-threaded interpreter and handles whole tracks
(separate, then museval). Each needs up to ~6-10 GB of RAM for a long song (whole-song STFT
and museval), so the default is 4 workers, safe next to a training run. Measured on the
9850X3D, one base-16 chunk takes 2.3 ms in int8 with TFLite's default XNNPACK kernels, 101 ms
with the reference kernels, and 118 ms in int16 (XNNPACK has no int16 path). XNNPACK may round
an integer differently from the reference kernels now and then; --reference-kernels checks that
this doesn't change the scores.
"""
import argparse
import json
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np
import torch

from chunked import separate_chunked
from data import musdb_split, read_audio, read_stems
from eval import save_results, score_track, track_folder
from needle import SOURCES, NeedleConfig

HERE = Path(__file__).parent
_W = {}                                                     # per-worker state (interpreter, config)


def load_config(tflite: Path, frames=None):
    """The model config is in export_vela.py's report.json next to the .tflite file."""
    cfg = json.loads((tflite.parent / "report.json").read_text())["config"]
    cfg["mult"] = tuple(cfg["mult"])
    if frames:
        cfg["frames"] = frames
    return NeedleConfig(**cfg)


def _init(tflite, cfg, overlap, reference):
    import tensorflow as tf
    torch.set_num_threads(1)
    resolver = tf.lite.experimental.OpResolverType
    it = tf.lite.Interpreter(model_path=tflite, num_threads=1, experimental_op_resolver_type=(
        resolver.BUILTIN_REF if reference else resolver.AUTO))
    it.allocate_tensors()
    i, o = it.get_input_details()[0], it.get_output_details()[0]
    _W.update(it=it, i=i, o=o, cfg=cfg, overlap=overlap, inputs=it.get_input_details())
    if cfg.arch == "bs":
        from needle2 import row_of_bin
        _W["row_of_bin"] = row_of_bin(cfg)


def _predict(xb):
    """(N, C, bins, frames) float -> (N, S*C, bins, frames) float masks, one chunk at a time."""
    if _W["cfg"].arch == "bs":
        return _predict_bands(xb)
    it, i, o = _W["it"], _W["i"], _W["o"]
    (si, zi), (so, zo) = i["quantization"], o["quantization"]
    lo, hi = np.iinfo(i["dtype"]).min, np.iinfo(i["dtype"]).max
    out = []
    for x in xb:
        x = x.permute(1, 2, 0).numpy()[None]                 # PyTorch NCHW -> TFLite NHWC
        it.set_tensor(i["index"], np.clip(np.round(x / si + zi), lo, hi).astype(i["dtype"]))
        it.invoke()
        y = (it.get_tensor(o["index"]).astype(np.float32) - zo) * so
        out.append(torch.from_numpy(y[0]).permute(2, 0, 1))
    return torch.stack(out)


def _predict_bands(xb):
    """Needle 2 (export_vela2.py): the band tensors go in, matched to the model's inputs by shape; the filters
    come out on the band grid, and every bin takes its row's filter, as the M85 does on the deck."""
    from needle2 import to_bands
    it, o, cfg = _W["it"], _W["o"], _W["cfg"]
    so, zo = o["quantization"]
    out = []
    for x in xb:
        bands = {tuple(b.shape[1:]): b for b in (b.permute(0, 2, 3, 1).numpy() for b in to_bands(x[None], cfg))}
        for i in _W["inputs"]:
            s, z = i["quantization"]
            lo, hi = np.iinfo(i["dtype"]).min, np.iinfo(i["dtype"]).max
            it.set_tensor(i["index"], np.clip(np.round(bands[tuple(i["shape"][1:])] / s + z), lo, hi).astype(i["dtype"]))
        it.invoke()
        y = (it.get_tensor(o["index"]).astype(np.float32) - zo) * so          # (1, rows, frames, K)
        out.append(torch.from_numpy(y[0]).permute(2, 0, 1)[:, _W["row_of_bin"]])
    return torch.stack(out)


def _work(job):
    name, folder, mix_path = job
    t = time.time()
    if mix_path:                                            # MUSDB: real mixture.wav
        refs = read_stems(folder, "wav", dtype="float32")
        mix = read_audio(mix_path, dtype="float32")
    else:                                                   # --refs: the mixture is the sum of the stems
        refs = read_stems(folder, "flac", dtype="float32")
        mix = refs.sum(0)
    est = separate_chunked(_predict, torch.from_numpy(mix), _W["cfg"], _W["overlap"]).numpy()
    return name, score_track(refs, est), time.time() - t


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tflite", required=True)
    ap.add_argument("--musdb", help="MUSDB18-HQ root")
    ap.add_argument("--subset", default="test", choices=["test", "valid"])
    ap.add_argument("--refs", help="instead of MUSDB: a folder of real stems (import_stems.py layout)")
    ap.add_argument("--frames", type=int, help="must match the .tflite input (default: from report.json)")
    ap.add_argument("--overlap", type=int, default=8)
    ap.add_argument("--workers", type=int, default=4, help="processes (each needs up to ~6-10 GB for a long song: whole-song STFT + museval)")
    ap.add_argument("--reference-kernels", action="store_true",
                    help="TFLite's slow reference kernels (closest to TFLite Micro on the device)")
    ap.add_argument("--float", help="results JSON of the float model, to print the quantisation loss")
    ap.add_argument("--name", help="default: <build folder>_<int8|int16>")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=str(HERE / "results"))
    a = ap.parse_args()

    tflite = Path(a.tflite)
    cfg = load_config(tflite, a.frames)
    # build\<run>\needle_b16_f32\needle_int8.tflite -> "<run>_int8", matching eval.py's "<run>"
    run = tflite.parent.parent.name if tflite.parent.name.startswith("needle_") and tflite.parent.parent.name != "build" \
        else tflite.parent.name
    name = a.name or f"{run}_{tflite.stem.replace('needle_', '')}"
    if a.refs:
        subset = Path(a.refs).name
        jobs = [(p.parent.name, str(p.parent), None) for p in sorted(Path(a.refs).glob("*/done"))]
    else:
        subset = a.subset
        jobs = [(n, str(track_folder(a.musdb, a.subset, n)), str(track_folder(a.musdb, a.subset, n) / "mixture.wav"))
                for n in musdb_split(a.musdb)[a.subset]]
    if a.limit:
        jobs = jobs[:a.limit]

    t0, per_track = time.time(), {}
    with mp.get_context("spawn").Pool(a.workers, initializer=_init,
                                      initargs=(str(tflite), cfg, a.overlap, a.reference_kernels)) as pool:
        for k, (n, scores, secs) in enumerate(pool.imap_unordered(_work, jobs), 1):
            per_track[n] = scores
            print(f"{k}/{len(jobs)} {n}: SDR avg {np.mean([scores['SDR'][s] for s in SOURCES]):.2f} dB ({secs:.0f} s)", flush=True)
    print(f"evaluated {len(jobs)} tracks in {(time.time() - t0) / 60:.1f} min")
    per_track = dict(sorted(per_track.items()))
    summary = save_results(Path(a.out) / f"{name}_{subset}.json", name, subset, per_track,
                           {"tflite": str(tflite), "frames": cfg.frames, "overlap": a.overlap,
                            "reference_kernels": a.reference_kernels})
    if a.float:
        ref = json.loads(Path(a.float).read_text())["summary"]["SDR"]
        print("quantisation loss (float - quantised), dB: " +
              "  ".join(f"{s} {ref[s] - summary['SDR'][s]:+.2f}" for s in (*SOURCES, "avg")))


if __name__ == "__main__":
    main()
