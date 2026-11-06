"""Where does quantisation hurt? Layer-by-layer comparison of the int8 model with the float one.

    .venv-export\\Scripts\\python quant_debug.py --checkpoint runs\\b16_f32\\best.pt ^
        --audio F:\\needle\\data\\musdb18hq\\train --audio-glob "*/mixture.wav"

For every layer, TFLite's QuantizationDebugger runs the float and the int8 version on the same
input and measures the difference. The number to read is rmse / scale: the error in units of
one integer step of that layer. Pure rounding gives about 0.29 (1/sqrt(12)); a layer far above
that is losing information, usually because its values span a huge range, so one integer step
is coarse for most of them (a few large outliers stretch the range for everyone).

Also prints each layer's range (255 x scale for int8), which shows where outliers stretch it.
Writes results/quant_debug_<run>.csv.
"""
import argparse
import csv
from pathlib import Path

import torch

from export_vela import build_keras, calibration_chunks
from needle import Needle, NeedleConfig, folded_convs

HERE = Path(__file__).parent


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--frames", type=int, default=32)
    ap.add_argument("--audio", required=True)
    ap.add_argument("--audio-glob", default="**/*")
    ap.add_argument("--chunks", type=int, default=128)
    a = ap.parse_args()

    import tensorflow as tf
    ck = torch.load(a.checkpoint, map_location="cpu")
    cfg = NeedleConfig(**{**ck["config"], "frames": a.frames})
    model = Needle(cfg)
    model.load_state_dict(ck["model"])
    model.eval()
    km = build_keras(cfg, folded_convs(model))
    chunks = calibration_chunks(cfg, a.audio, a.chunks, a.audio_glob)

    conv = tf.lite.TFLiteConverter.from_keras_model(km)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    conv.representative_dataset = lambda: ([c[None]] for c in chunks)
    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    conv.inference_input_type = conv.inference_output_type = tf.int8
    dbg = tf.lite.experimental.QuantizationDebugger(converter=conv, debug_dataset=lambda: ([c[None]] for c in chunks[:32]))
    dbg.run()

    out = HERE / "results" / f"quant_debug_{Path(a.checkpoint).parent.name}.csv"
    out.parent.mkdir(exist_ok=True)
    with open(out, "w", newline="") as f:
        dbg.layer_statistics_dump(f)
    rows = list(csv.DictReader(open(out)))
    print(f"{'#':>3} {'op':<26} {'rmse/scale':>10} {'scale':>9} {'range ~255*scale':>16}  tensor")
    for i, r in enumerate(rows):
        scale = float(r["scale"])
        rmse = float(r["mean_squared_error"]) ** 0.5
        flag = "  <-- far above rounding" if scale > 0 and rmse / scale > 1.0 else ""
        print(f"{i:>3} {r['op_name']:<26} {rmse / scale if scale else float('nan'):>10.2f} {scale:>9.4f} "
              f"{255 * scale:>16.2f}  {r['tensor_name'][-40:]}{flag}")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
