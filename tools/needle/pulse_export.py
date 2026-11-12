r"""Pulse -> int8 / int16 TFLite -> Arm Vela (Ethos-U55-256, RA8P1), and the quantised model scored on the
Raveform test set with the same grid fit as the float one. Run with the export venv (TensorFlow + Vela):

    .venv-export\Scripts\python.exe pulse_export.py --model p2_w24

Path (as export_vela.py does for Needle): PyTorch, BatchNorm folded into the convs -> the same network in
Keras (checked against PyTorch) -> TFLite post-training quantisation, calibrated on Raveform TRAIN mels ->
Vela. Nearest 2x upsampling in time is written as concat(x, x) + reshape: exactly nearest-neighbour, and both
ops are free or native on the U55 (a 1x2 RESIZE may not be).

Reports: CPU operators left (must be 0), NPU time per 10.24-s chunk, SRAM arena, weights; float vs int8 vs
int16: beat F, downbeat F, tempo family, tracks with > 90 % bars right.
"""
import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from export_vela import vela                                        # noqa: E402
from pulse import MELS, T_CHUNK, Pulse, f_measure, predict_track   # noqa: E402
from pulse_grid import bar_share, fit_grid                          # noqa: E402
from pulse_rave import Rave, bar_fit                                # noqa: E402
from pulse_raveform import tempo_ok                                 # noqa: E402


def fold(seq):
    """nn.Sequential(Conv2d(no bias), BatchNorm2d, ReLU) -> (weight OIHW, bias, stride, padding)."""
    conv, bn = seq[0], seq[1]
    s = bn.weight / torch.sqrt(bn.running_var + bn.eps)
    return (conv.weight * s[:, None, None, None]).detach(), (bn.bias - bn.running_mean * s).detach(), conv.stride, conv.padding


def build_keras(model: Pulse):
    import tensorflow as tf
    keras = tf.keras

    def conv(x, seq=None, w=None, b=None, stride=(1, 1), pad=(0, 0), relu=True):
        if seq is not None:
            w, b, stride, pad = fold(seq)
        if pad != (0, 0):
            x = keras.layers.ZeroPadding2D(((pad[0], pad[0]), (pad[1], pad[1])))(x)
        layer = keras.layers.Conv2D(w.shape[0], tuple(w.shape[2:]), strides=tuple(stride), padding="valid",
                                    activation="relu" if relu else None)
        y = layer(x)
        layer.set_weights([w.permute(2, 3, 1, 0).numpy(), b.numpy()])
        return y

    def up2(x):                                                     # nearest 2x in time: concat + reshape
        _, h, wd, c = x.shape
        return keras.layers.Reshape((h, wd * 2, c))(keras.layers.Concatenate(axis=-1)([x, x]))

    inp = keras.Input((MELS, T_CHUNK, 1), batch_size=1)
    h = inp
    for seq in model.stem:
        h = conv(h, seq)
    skips = []
    for d, m in zip(model.down, model.mid):
        skips.append(h)
        h = conv(conv(h, d), m)
    for u in model.up:
        h = conv(keras.layers.Add()([up2(h), skips.pop()]), u)
    out = conv(h, w=model.head.weight.detach(), b=model.head.bias.detach(), relu=False)   # (1, 1, T, 2) logits
    return keras.Model(inp, out)


def calibration(n=256):
    tr = Rave("train")
    rng = random.Random(0)
    out = []
    while len(out) < n:
        m = tr.mels[rng.choice(tr.ids)]
        if m.shape[1] <= T_CHUNK:
            continue
        a = rng.randrange(0, m.shape[1] - T_CHUNK)
        out.append(np.asarray(m[:, a:a + T_CHUNK], dtype=np.float32)[:, :, None])
    return out


def to_tflite(km, chunks, mode):
    import tensorflow as tf
    c = tf.lite.TFLiteConverter.from_keras_model(km)
    c.optimizations = [tf.lite.Optimize.DEFAULT]
    c.representative_dataset = lambda: ([x[None]] for x in chunks)
    if mode == "int8":
        c.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        c.inference_input_type = c.inference_output_type = tf.int8
    else:
        c.target_spec.supported_ops = [tf.lite.OpsSet.EXPERIMENTAL_TFLITE_BUILTINS_ACTIVATIONS_INT16_WEIGHTS_INT8]
        c.inference_input_type = c.inference_output_type = tf.int16
    return c.convert()


class TFLiteNet:
    """Callable like Pulse on a (B, 1, 128, 512) tensor, so pulse.predict_track (chunking, crossfade) is reused."""
    def __init__(self, blob):
        import tensorflow as tf
        self.it = tf.lite.Interpreter(model_content=blob, num_threads=8)
        self.it.allocate_tensors()
        self.i, self.o = self.it.get_input_details()[0], self.it.get_output_details()[0]

    def __call__(self, x):
        ys = []
        for xx in x.numpy():
            s, z = self.i["quantization"]
            q = np.round(xx[0][:, :, None] / s + z)
            lim = np.iinfo(self.i["dtype"])
            self.it.set_tensor(self.i["index"], q.clip(lim.min, lim.max).astype(self.i["dtype"])[None])
            self.it.invoke()
            s, z = self.o["quantization"]
            y = (self.it.get_tensor(self.o["index"]).astype(np.float32) - z) * s    # (1, 1, T, 2)
            ys.append(y[0, 0].T)
        return torch.from_numpy(np.stack(ys))


def score(net, te, device):
    rows = []
    for t in te.ids:
        b, d = te.lab[t]
        mel = np.asarray(te.mels[t], dtype=np.float32)
        with torch.inference_mode():
            y = predict_track(net, mel, device)
        grid, _, _ = fit_grid(y[0], mel.shape[1])
        ed = bar_fit(grid, y[1], bars=32)
        rows.append((f_measure(b, grid), f_measure(d, ed), tempo_ok(b, grid), bar_share(d, ed) > 0.9))
    v = np.array(rows, dtype=float).mean(0)
    return {"beat_F": v[0], "downbeat_F": v[1], "tempo_right": v[2], "bars_right_90": v[3]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="p2_w24")
    ap.add_argument("--modes", default="int8,int16")
    ap.add_argument("--no-eval", action="store_true")
    a = ap.parse_args()
    run = HERE / "runs_pulse" / a.model
    ck = torch.load(run / ("best.pt" if (run / "best.pt").exists() else "last.pt"), map_location="cpu", weights_only=False)
    model = Pulse(ck["width"], ck.get("levels", 5))
    model.load_state_dict(ck["model"])
    model.eval()
    out = HERE / "build" / f"pulse_{a.model}"
    out.mkdir(parents=True, exist_ok=True)

    km = build_keras(model)
    chunks = calibration()
    with torch.no_grad():
        ref = model(torch.from_numpy(chunks[0][:, :, 0])[None, None])[0].numpy()        # (2, T)
    kd = float(np.abs(km(chunks[0][None]).numpy()[0, 0].T - ref).max())
    report = {"model": a.model, "keras_vs_pytorch_max_abs": kd}
    print("keras vs pytorch max abs", kd, flush=True)
    assert kd < 1e-3, kd

    te = None if a.no_eval else Rave("test")
    if te:
        report["float"] = score(model, te, "cpu")
        print("float", report["float"], flush=True)
    for mode in a.modes.split(","):
        blob = to_tflite(km, chunks, mode)
        path = out / f"pulse_{mode}.tflite"
        path.write_bytes(blob)
        entry = {"tflite_bytes": len(blob)}
        for system in ("RA8P1_SRAM", "RA8P1_SDRAM"):
            row, cpu_ops, _ = vela(path, system, out / mode)
            entry[system] = {"cpu_operators": cpu_ops, "inference_ms_per_chunk": float(row["inference_time"]) * 1e3,
                             "arena_kib": float(row.get("sram_memory_used", 0) or 0),
                             "weights_kib": float(row.get("off_chip_flash_memory_used", 0) or 0)}
        if te:
            entry["test"] = score(TFLiteNet(blob), te, "cpu")
        report[mode] = entry
        print(mode, json.dumps(entry), flush=True)
    (out / "report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
