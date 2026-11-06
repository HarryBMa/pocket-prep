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
(separate, then museval). 16 workers keep all 16 hardware threads busy. Measured on the
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
    _W.update(it=it, i=i, o=o, cfg=cfg, overlap=overlap)


def _predict(xb):
    """(N, C, bins, frames) float -> (N, S*C, bins, frames) float masks, one chunk at a time."""
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


def _work(job):
    name, folder, mix_path = job
    t = time.time()
    mix = torch.from_numpy(read_audio(mix_path, dtype="float32"))
    est = separate_chunked(_predict, mix, _W["cfg"], _W["overlap"]).numpy()
    scores = score_track(read_stems(folder, "wav", dtype="float32"), est)
    return name, scores, time.time() - t


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tflite", required=True)
    ap.add_argument("--musdb", required=True)
    ap.add_argument("--subset", default="test", choices=["test", "valid"])
    ap.add_argument("--frames", type=int, help="must match the .tflite input (default: from report.json)")
    ap.add_argument("--overlap", type=int, default=8)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--reference-kernels", action="store_true",
                    help="TFLite's slow reference kernels (closest to TFLite Micro on the device)")
    ap.add_argument("--float", help="results JSON of the float model, to print the quantisation loss")
    ap.add_argument("--name", help="default: <build folder>_<int8|int16>")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=str(HERE / "results"))
    a = ap.parse_args()

    tflite = Path(a.tflite)
    cfg = load_config(tflite, a.frames)
    name = a.name or f"{tflite.parent.name}_{tflite.stem.replace('needle_', '')}"
    names = musdb_split(a.musdb)[a.subset]
    if a.limit:
        names = names[:a.limit]
    jobs = [(n, str(track_folder(a.musdb, a.subset, n)), str(track_folder(a.musdb, a.subset, n) / "mixture.wav")) for n in names]

    t0, per_track = time.time(), {}
    with mp.get_context("spawn").Pool(a.workers, initializer=_init,
                                      initargs=(str(tflite), cfg, a.overlap, a.reference_kernels)) as pool:
        for k, (n, scores, secs) in enumerate(pool.imap_unordered(_work, jobs), 1):
            per_track[n] = scores
            print(f"{k}/{len(jobs)} {n}: SDR avg {np.mean([scores['SDR'][s] for s in SOURCES]):.2f} dB ({secs:.0f} s)", flush=True)
    print(f"evaluated {len(jobs)} tracks in {(time.time() - t0) / 60:.1f} min")
    per_track = dict(sorted(per_track.items()))
    summary = save_results(Path(a.out) / f"{name}_{a.subset}.json", name, a.subset, per_track,
                           {"tflite": str(tflite), "frames": cfg.frames, "overlap": a.overlap,
                            "reference_kernels": a.reference_kernels})
    if a.float:
        ref = json.loads(Path(a.float).read_text())["summary"]["SDR"]
        print("quantisation loss (float - quantised), dB: " +
              "  ".join(f"{s} {ref[s] - summary['SDR'][s]:+.2f}" for s in (*SOURCES, "avg")))


if __name__ == "__main__":
    main()
