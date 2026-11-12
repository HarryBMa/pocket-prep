r"""Export Needle 2 (needle2.py) to int8 TFLite and compile it with Arm Vela for the RA8P1's Ethos-U55-256:
what runs on the NPU, how much working memory it needs and how fast it is. The Needle 2 twin of export_vela.py.

    python export_vela2.py                                           # default Needle 2, random weights
    python export_vela2.py --widths 24 48 96 128 160 --rnn 3         # another size, random weights
    python export_vela2.py --checkpoint runs\n2_a\best.pt --audio F:\needle\data\musdb18hq\train --audio-glob "*/mixture.wav"

Without --checkpoint the weights are random: speed and memory are still right (they depend on the shapes, not the
values), the quantisation error is not meaningful. Run it with the export environment (.venv-export).

Path: PyTorch weights (BatchNorm folded) -> the same network as TensorFlow operations (needle2_tf.py) -> TFLite
converter (post-training quantisation) -> Vela. The TensorFlow copy is checked against PyTorch in float, and the
quantised model against the float one, before Vela runs.
"""
import argparse
import json
import os
import time
from dataclasses import asdict
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import numpy as np
import torch

from export_vela import calibration_chunks, vela
from needle import Needle, NeedleConfig, count_macs, network_input
from needle2 import bs_config, randomize_norms
from needle2_tf import bands_nhwc, build

HERE = Path(__file__).parent


def noise_chunks(cfg: NeedleConfig, n, seed=0):
    """Network inputs (bins, frames, C) from a random spectrogram (log-normal magnitudes, random phases): enough
    to exercise every tensor when there is no audio at hand."""
    g = torch.Generator().manual_seed(seed)
    out = []
    for _ in range(n):
        mag = torch.exp(torch.randn(1, cfg.audio_channels, cfg.n_fft // 2 + 1, cfg.frames, generator=g) * 1.5)
        spec = torch.polar(mag, torch.rand(mag.shape, generator=g) * 6.2832)
        out.append(network_input(spec, cfg)[0].permute(1, 2, 0).numpy())
    return out


def to_tflite(fn, shapes, chunks, mode):
    """chunks: a list of band lists (what bands_nhwc returns), used to fit each tensor's integer range."""
    import tensorflow as tf
    cf = tf.function(fn).get_concrete_function(*[tf.TensorSpec([1, *s], tf.float32) for s in shapes])
    conv = tf.lite.TFLiteConverter.from_concrete_functions([cf])
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    conv.representative_dataset = lambda: (list(c) for c in chunks)
    if mode == "int8":
        conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        conv.inference_input_type = conv.inference_output_type = tf.int8
    else:   # 16-bit activations, 8-bit weights
        conv.target_spec.supported_ops = [tf.lite.OpsSet.EXPERIMENTAL_TFLITE_BUILTINS_ACTIVATIONS_INT16_WEIGHTS_INT8]
        conv.inference_input_type = conv.inference_output_type = tf.int16
    return conv.convert()


def run_tflite(blob, bands):
    """One chunk through the TFLite interpreter (integer arithmetic, as on the NPU). The band inputs are matched
    to the model's inputs by shape (the four bands all differ)."""
    import tensorflow as tf
    it = tf.lite.Interpreter(model_content=blob)
    it.allocate_tensors()
    by_shape = {tuple(b.shape): b for b in bands}
    for i in it.get_input_details():
        s, z = i["quantization"]
        x = by_shape[tuple(i["shape"])]
        it.set_tensor(i["index"], np.round(x / s + z).clip(np.iinfo(i["dtype"]).min, np.iinfo(i["dtype"]).max).astype(i["dtype"]))
    it.invoke()
    o = it.get_output_details()[0]
    s, z = o["quantization"]
    ops = {}
    for d in it._get_ops_details():
        ops[d["op_name"]] = ops.get(d["op_name"], 0) + 1
    return (it.get_tensor(o["index"]).astype(np.float32) - z) * s, ops


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint")
    ap.add_argument("--frames", type=int, help="frames per chunk (default: the checkpoint's, or 128)")
    ap.add_argument("--widths", type=int, nargs="+", help="(random weights only) channels per level")
    ap.add_argument("--tdf", type=int, nargs="+", help="(random weights only) TDF factor per level, 0 = none")
    ap.add_argument("--rnn", type=int, help="(random weights only) dual-path GRU blocks")
    ap.add_argument("--rnn-hidden", type=int, help="(random weights only) GRU state size per direction")
    ap.add_argument("--tsteps", type=int, help="(random weights only) down-samplings that also halve time")
    ap.add_argument("--taps", type=int, help="(random weights only) deep-filter taps")
    ap.add_argument("--audio", help="folder of 44.1 kHz stereo tracks for calibration (training data, never test)")
    ap.add_argument("--audio-glob", default="**/*")
    ap.add_argument("--calib-chunks", type=int, default=64)
    ap.add_argument("--modes", default="int8")
    ap.add_argument("--systems", default="RA8P1_SRAM,RA8P1_SDRAM")
    ap.add_argument("--name", help="folder name under --out (default: from the sizes)")
    ap.add_argument("--out", default=str(HERE / "build"))
    a = ap.parse_args()

    torch.manual_seed(0)
    if a.checkpoint:
        ck = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
        cfg = NeedleConfig(**{**ck["config"], **({"frames": a.frames} if a.frames else {})})
        model = Needle(cfg)
        model.load_state_dict(ck["model"])
    else:
        kw = {k: v for k, v in dict(frames=a.frames, widths=a.widths, tdf=a.tdf, rnn=a.rnn, rnn_hidden=a.rnn_hidden,
                                    tsteps=a.tsteps, taps=a.taps).items() if v is not None}
        cfg = bs_config(**kw)
        model = randomize_norms(Needle(cfg))
    assert cfg.arch == "bs", "this script exports Needle 2; use export_vela.py for Needle 1"
    model.eval()
    name = a.name or ("needle2_w" + "-".join(map(str, cfg.widths)) + f"_r{cfg.rnn}x{cfg.rnn_hidden}_f{cfg.frames}{cfg.tag()}")
    out = Path(a.out) / name
    out.mkdir(parents=True, exist_ok=True)
    report = {"config": asdict(cfg), **count_macs(cfg)}

    fn, shapes = build(model)
    full = calibration_chunks(cfg, a.audio, a.calib_chunks, a.audio_glob) if a.audio else noise_chunks(cfg, a.calib_chunks)
    chunks = [[b[0:1] for b in bands_nhwc(torch.from_numpy(c).permute(2, 0, 1), cfg)] for c in full]
    x = torch.from_numpy(full[0]).permute(2, 0, 1)[None]
    with torch.no_grad():
        ref = model.forward_rows(x).permute(0, 2, 3, 1).numpy()                 # (1, rows, frames, K)
    t0 = time.time()
    tf_out = fn(*chunks[0]).numpy()
    report["tf_vs_pytorch_max_abs"] = float(np.abs(tf_out - ref).max())
    print(f"TensorFlow copy vs PyTorch: max difference {report['tf_vs_pytorch_max_abs']:.2e} ({time.time() - t0:.0f} s)", flush=True)
    assert report["tf_vs_pytorch_max_abs"] < 2e-3, report

    for mode in a.modes.split(","):
        t0 = time.time()
        blob = to_tflite(fn, shapes, chunks, mode)
        path = out / f"needle_{mode}.tflite"
        path.write_bytes(blob)
        q, ops = run_tflite(blob, chunks[0])
        entry = {"tflite_bytes": len(blob), "operators": sum(ops.values()), "operator_types": ops,
                 "out_mean_abs_error_vs_float": float(np.abs(q - ref).mean()),
                 "out_max_abs_error_vs_float": float(np.abs(q - ref).max())}
        print(f"{mode}: converted in {time.time() - t0:.0f} s, {entry['operators']} operators, "
              f"mean |error| vs float {entry['out_mean_abs_error_vs_float']:.4f}", flush=True)
        for system in a.systems.split(","):
            t0 = time.time()
            row, cpu_ops, _ = vela(path, system, out / mode)
            t = float(row["inference_time"])
            entry[system] = {
                "inference_ms_per_chunk": t * 1e3,
                "realtime_factor": t / report["chunk_s"],
                "npu_cycles": int(float(row.get("cycles_npu", 0) or 0)),
                "arena_kib": float(row.get("sram_memory_used", 0) or 0),
                "weights_kib": float(row.get("off_chip_flash_memory_used", 0) or 0),
                "cpu_operators": cpu_ops,
            }
            print(f"{mode} {system}: {t * 1e3:.0f} ms per {report['chunk_s']:.2f}-s chunk (RTF {t / report['chunk_s']:.3f}), "
                  f"arena {entry[system]['arena_kib'] / 1024:.1f} MiB, weights {entry[system]['weights_kib'] / 1024:.2f} MiB, "
                  f"CPU operators {cpu_ops} (Vela took {time.time() - t0:.0f} s)", flush=True)
        report[mode] = entry
    (out / "report.json").write_text(json.dumps(report, indent=1))
    print(f"params {report['params']:,}, {report['gmac_per_audio_s']:.2f} GMAC per second of audio; report: {out / 'report.json'}")


if __name__ == "__main__":
    main()
