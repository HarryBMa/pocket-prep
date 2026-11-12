r"""Vela probes for Needle-2: which building blocks does the Ethos-U55 run (Vela 5.2.0, RA8P1 configs of
export_vela.py), what do they cost in NPU time and working memory, and which operators fall back to the CPU.

    .venv-export\Scripts\python vela_probe.py [name-substring ...]

Each probe is a small TensorFlow function with random constant weights at the tensor sizes Needle-2 would use
(frequency rows x time frames x channels), converted to int8 TFLite (or 16x8 where noted) and compiled with Vela
for the RA8P1 with the arena in SDRAM. One JSON line per probe goes to build/vela_probe/results.jsonl; Vela's
own output is kept in build/vela_probe/<name>/vela.txt. The results of 12 Nov 2026 are in NEEDLE2.md.
"""
import csv
import json
import os
import re
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import numpy as np
import tensorflow as tf

from export_vela import VELA_INI

HERE = Path(__file__).parent
OUT = HERE / "build" / "vela_probe"
rng = np.random.default_rng(0)


# ── tiny layer helpers (random constant weights) ─────────────────────────────

def W(*shape, fan_in=None):
    fan_in = fan_in or int(np.prod(shape[:-1]))
    return tf.constant((rng.standard_normal(shape) * np.sqrt(2.0 / fan_in)).astype(np.float32))


def B(n):
    return tf.constant((rng.standard_normal(n) * 0.05).astype(np.float32))


def conv(x, cout, k=(3, 3), s=(1, 1), d=(1, 1), act=True, pad="SAME"):
    cin = x.shape[-1]
    y = tf.nn.conv2d(x, W(k[0], k[1], cin, cout), strides=[1, s[0], s[1], 1], padding=pad, dilations=[1, d[0], d[1], 1])
    y = tf.nn.bias_add(y, B(cout))
    return tf.nn.relu(y) if act else y


def dwconv(x, k=(3, 3), act=True):
    c = x.shape[-1]
    y = tf.nn.depthwise_conv2d(x, W(k[0], k[1], c, 1, fan_in=k[0] * k[1]), strides=[1, 1, 1, 1], padding="SAME")
    y = tf.nn.bias_add(y, B(c))
    return tf.nn.relu(y) if act else y


# ── probes: name -> (function, input shapes without batch, modes) ────────────

PROBES = {}


def probe(name, shapes, modes=("int8",)):
    def deco(fn):
        PROBES[name] = (fn, shapes, modes)
        return fn
    return deco


@probe("stem_bands", [(256, 128, 6), (192, 128, 24), (64, 128, 96)])
def stem_bands(lo, mid, hi):
    """Three band inputs (the M85 stacks 4 / 16 neighbouring bins into channels), one conv each, joined along
    frequency, then one conv on the joined 512-row grid."""
    x = tf.concat([conv(lo, 32), conv(mid, 32), conv(hi, 32)], axis=1)
    return conv(x, 32)


@probe("down_22", [(512, 128, 32)])
def down_22(x):
    return conv(x, 64, s=(2, 2))


@probe("down_f", [(128, 32, 96)])
def down_f(x):
    """Halve frequency only (time resolution kept)."""
    return conv(x, 128, s=(2, 1))


@probe("up_f_reshape", [(32, 32, 160)])
def up_f_reshape(x):
    """Frequency-only 2x nearest upsampling with CONCATENATION + RESHAPE: reshape(concat([x, x], time axis))."""
    h, w, c = x.shape[1:]
    return conv(tf.reshape(tf.concat([x, x], axis=2), [1, 2 * h, w, c]), 128)


@probe("up_f_resize", [(32, 32, 160)])
def up_f_resize(x):
    h, w, c = x.shape[1:]
    return conv(tf.image.resize(x, (2 * h, w), method="nearest"), 128)


@probe("up_f_tconv", [(32, 32, 160)])
def up_f_tconv(x):
    h, w, c = x.shape[1:]
    y = tf.nn.conv2d_transpose(x, W(2, 1, 128, c), [1, 2 * h, w, 128], strides=[1, 2, 1, 1], padding="SAME")
    return tf.nn.relu(tf.nn.bias_add(y, B(128)))


@probe("up_22_resize", [(128, 32, 96)])
def up_22_resize(x):
    h, w, c = x.shape[1:]
    return conv(tf.image.resize(x, (2 * h, 2 * w), method="nearest"), 64)


@probe("up_22_slice", [(32, 32, 160)])
def up_22_slice(x):
    """2x2 nearest upsampling (what the U55 has) followed by dropping every other time frame."""
    h, w, c = x.shape[1:]
    y = tf.image.resize(x, (2 * h, 2 * w), method="nearest")
    return conv(y[:, :, ::2, :], 128)


@probe("tdf", [(64, 32, 128)])
def tdf(x):
    """TDF (frequency-wise fully connected layer shared by all channels and frames): move frequency to the
    channel axis, two 1x1 convs (F -> F/4 -> F), move it back, residual."""
    f = x.shape[1]
    y = tf.transpose(x, [0, 3, 2, 1])                      # (1, C, T, F)
    y = conv(conv(y, f // 4, k=(1, 1)), f, k=(1, 1))
    return x + tf.transpose(y, [0, 3, 2, 1])


@probe("tdf_L1", [(256, 64, 64)])
def tdf_l1(x):
    f = x.shape[1]
    y = tf.transpose(x, [0, 3, 2, 1])
    y = conv(conv(y, f // 8, k=(1, 1)), f, k=(1, 1))
    return x + tf.transpose(y, [0, 3, 2, 1])


@probe("fgm", [(64, 32, 128)])
def fgm(x):
    """Frequency-global mixing without TRANSPOSE: a (F x 1) VALID conv squeezes each frame's whole spectrum into
    one vector, a (F x 1) transposed conv spreads it back over frequency, residual."""
    f, t, c = x.shape[1:]
    y = conv(x, 64, k=(f, 1), pad="VALID")                 # (1, 1, T, 64)
    y = tf.nn.conv2d_transpose(y, W(f, 1, c, 64, fan_in=64), [1, f, t, c], strides=[1, 1, 1, 1], padding="VALID")
    return x + tf.nn.bias_add(y, B(c))


@probe("dw_f31", [(64, 32, 128)])
def dw_f31(x):
    return x + conv(dwconv(x, k=(31, 1)), 128, k=(1, 1), act=False)


@probe("dw_f15_L0", [(512, 128, 32)])
def dw_f15_l0(x):
    return x + conv(dwconv(x, k=(15, 1)), 32, k=(1, 1), act=False)


@probe("dil_t", [(32, 32, 160)])
def dil_t(x):
    """Four residual 3x3 convs with time dilation 1, 2, 4, 8."""
    for d in (1, 2, 4, 8):
        x = x + conv(x, 160, d=(1, d))
    return x


def attention(x, heads, dim, mlp=2):
    """Self-attention along the time axis W for every frequency row, LeViT style (BatchNorm folded into the 1x1
    convs, no LayerNorm). Heads are stacked along H so one BATCH_MATMUL serves all of them."""
    f, t, c = x.shape[1:]
    pos = tf.constant((rng.standard_normal((1, 1, t, c)) * 0.1).astype(np.float32))
    xin = x + pos                                           # learned position embedding, broadcast over rows
    q = conv(xin, heads * dim, k=(1, 1), act=False)
    k = conv(xin, heads * dim, k=(1, 1), act=False)
    v = conv(xin, heads * dim, k=(1, 1), act=False)
    if heads > 1:
        q, k, v = (tf.concat(tf.split(z, heads, axis=3), axis=1) for z in (q, k, v))      # (1, heads*F, T, dim)
    a = tf.nn.softmax(tf.matmul(q, k, transpose_b=True) * (dim ** -0.5), axis=-1)             # (1, heads*F, T, T)
    y = tf.matmul(a, v)                                                                        # (1, heads*F, T, dim)
    if heads > 1:
        y = tf.concat(tf.split(y, heads, axis=1), axis=3)                                      # (1, F, T, heads*dim)
    x = x + conv(y, c, k=(1, 1), act=False)
    return x + conv(conv(x, mlp * c, k=(1, 1)), c, k=(1, 1), act=False)


@probe("attn_t_1h", [(32, 32, 160)])
def attn_t_1h(x):
    return attention(x, 1, 64)


@probe("attn_t_4h", [(32, 32, 160)])
def attn_t_4h(x):
    return attention(x, 4, 32)


@probe("attn_f_4h", [(32, 32, 160)])
def attn_f_4h(x):
    """Attention along frequency: swap H and W, same block, swap back."""
    return tf.transpose(attention(tf.transpose(x, [0, 2, 1, 3]), 4, 32), [0, 2, 1, 3])


@probe("attn_t_4h_L3", [(64, 32, 128)])
def attn_t_4h_l3(x):
    return attention(x, 4, 32)


@probe("attn_t_4h_T64", [(32, 64, 160)])
def attn_t_4h_t64(x):
    return attention(x, 4, 32)


def lstm_unrolled(x, hidden):
    """One LSTM direction along time, written out step by step (the frequency rows are the batch): per step one
    1x1 conv on [x_t, h], SPLIT into the four gates, LOGISTIC / TANH, MUL, ADD."""
    f, t, c = x.shape[1:]
    w, b = W(1, 1, c + hidden, 4 * hidden), B(4 * hidden)
    h = tf.zeros([1, f, 1, hidden])
    cell = tf.zeros([1, f, 1, hidden])
    outs = []
    for xt in tf.split(x, t, axis=2):
        g = tf.nn.bias_add(tf.nn.conv2d(tf.concat([xt, h], axis=3), w, [1, 1, 1, 1], "SAME"), b)
        i, fg, o, cand = tf.split(g, 4, axis=3)
        cell = tf.sigmoid(fg) * cell + tf.sigmoid(i) * tf.tanh(cand)
        h = tf.sigmoid(o) * tf.tanh(cell)
        outs.append(h)
    return tf.concat(outs, axis=2)


@probe("lstm_unrolled_t", [(32, 32, 160)], modes=("int8", "int16"))
def lstm_unrolled_t(x):
    return conv(lstm_unrolled(x, 64), 160, k=(1, 1), act=False)


@probe("bilstm_unrolled_t", [(32, 32, 160)])
def bilstm_unrolled_t(x):
    fwd = lstm_unrolled(x, 64)
    bwd = tf.reverse(lstm_unrolled(tf.reverse(x, [2]), 64), [2])
    return x + conv(tf.concat([fwd, bwd], axis=3), 160, k=(1, 1), act=False)


@probe("se_time", [(64, 32, 128)])
def se_time(x):
    """Squeeze-excite over time: mean over frames, 1x1 conv, sigmoid gate broadcast back over time."""
    g = tf.reduce_mean(x, axis=2, keepdims=True)
    return x * tf.sigmoid(conv(conv(g, 32, k=(1, 1)), x.shape[-1], k=(1, 1), act=False))


@probe("head_tanh", [(512, 128, 32)])
def head_tanh(x):
    return tf.tanh(conv(x, 36, k=(1, 1), act=False))


@probe("dec_L0", [(256, 64, 64), (512, 128, 32)])
def dec_l0(x, skip):
    """The decoder's last (largest) stage: 2x2 upsample, concat the skip, one 3x3 conv, the output head."""
    h, w, c = x.shape[1:]
    y = tf.concat([tf.image.resize(x, (2 * h, 2 * w), method="nearest"), skip], axis=3)
    return tf.tanh(conv(conv(y, 32), 36, k=(1, 1), act=False))


# ── harness ──────────────────────────────────────────────────────────────────

def to_tflite(fn, shapes, mode):
    cf = tf.function(fn).get_concrete_function(*[tf.TensorSpec([1, *s], tf.float32) for s in shapes])
    conv_ = tf.lite.TFLiteConverter.from_concrete_functions([cf])
    conv_.optimizations = [tf.lite.Optimize.DEFAULT]
    r = np.random.default_rng(1)
    conv_.representative_dataset = lambda: ([r.standard_normal((1, *s)).astype(np.float32) for s in shapes] for _ in range(4))
    if mode == "int8":
        conv_.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        conv_.inference_input_type = conv_.inference_output_type = tf.int8
    else:
        conv_.target_spec.supported_ops = [tf.lite.OpsSet.EXPERIMENTAL_TFLITE_BUILTINS_ACTIVATIONS_INT16_WEIGHTS_INT8]
        conv_.inference_input_type = conv_.inference_output_type = tf.int16
    return conv_.convert()


def tflite_ops(blob):
    it = tf.lite.Interpreter(model_content=blob)
    return Counter(d["op_name"] for d in it._get_ops_details())


def run_vela(path: Path, out: Path, system="RA8P1_SDRAM"):
    out.mkdir(parents=True, exist_ok=True)
    ini = out / "ra8p1_vela.ini"
    ini.write_text(VELA_INI)
    exe = Path(sys.executable).with_name("vela")
    cmd = [str(exe), str(path), "--accelerator-config", "ethos-u55-256", "--config", str(ini), "--system-config", system,
           "--memory-mode", "Shared_Sram_RA8P1", "--optimise", "Performance", "--output-dir", str(out / system),
           "--show-cpu-operations", "--verbose-performance"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    (out / "vela.txt").write_text(r.stdout + "\n--- stderr ---\n" + r.stderr, encoding="utf-8")
    if r.returncode != 0:
        return {"error": (r.stderr or r.stdout)[-600:]}
    row = next(csv.DictReader(open(next((out / system).glob("*summary*.csv")))))
    m = re.search(r"CPU operators = (\d+)", r.stdout)
    cpu = sorted(set(re.findall(r"(?:Warning|Info): (\w+) '", r.stdout)))
    return {"cpu_operators": int(m.group(1)) if m else None, "cpu_op_types": cpu,
            "inference_ms": float(row["inference_time"]) * 1e3, "npu_cycles": int(float(row.get("cycles_npu", 0) or 0)),
            "arena_kib": float(row.get("sram_memory_used", 0) or 0), "weights_kib": float(row.get("off_chip_flash_memory_used", 0) or 0)}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    wanted = sys.argv[1:]
    rows = []
    for name, (fn, shapes, modes) in PROBES.items():
        if wanted and not any(w in name for w in wanted):
            continue
        for mode in modes:
            tag = name if mode == "int8" else f"{name}_{mode}"
            t0 = time.time()
            res = {"probe": tag, "inputs": shapes}
            try:
                blob = to_tflite(fn, shapes, mode)
                d = OUT / tag
                d.mkdir(exist_ok=True)
                (d / "model.tflite").write_bytes(blob)
                res["tflite_ops"] = dict(tflite_ops(blob))
                res.update(run_vela(d / "model.tflite", d))
            except Exception as e:                                   # a probe that cannot even be converted is a result too
                res["error"] = f"{type(e).__name__}: {str(e)[:500]}"
            res["seconds"] = round(time.time() - t0, 1)
            rows.append(res)
            with open(OUT / "results.jsonl", "a") as fh:
                fh.write(json.dumps(res) + "\n")
            print(json.dumps(res), flush=True)
    print("\nprobe                 cpu  ms(SDRAM)  Mcycles  arena KiB  operators")
    for r in rows:
        if "error" in r:
            print(f"{r['probe']:20s}  ERROR {r['error'][:150]}")
            continue
        ops = " ".join(f"{k}x{v}" for k, v in sorted(r["tflite_ops"].items()))
        print(f"{r['probe']:20s} {str(r['cpu_operators']):>4s} {r['inference_ms']:9.2f} {r['npu_cycles'] / 1e6:8.2f} {r['arena_kib']:9.0f}  {ops}"
              + (f"   CPU: {r['cpu_op_types']}" if r["cpu_operators"] else ""))
    print("PROBES DONE")


if __name__ == "__main__":
    main()
