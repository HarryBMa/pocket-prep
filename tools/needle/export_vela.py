"""Export Needle to an int8 / int16 TFLite file and compile it with Arm Vela for the RA8P1's
Ethos-U55-256, then report what runs on the NPU, the memory it needs and how fast it is.

    python export_vela.py --base 16 --frames 64 [--checkpoint needle.pt] [--audio <dir of tracks>]

Without --checkpoint the weights are random: the speed and memory numbers are still right
(they depend on the shapes, not the values), the quantisation error is not meaningful.
--audio gives real spectrograms for calibrating the quantisation (needed with a checkpoint).

Path: PyTorch weights (BatchNorm folded) -> the same network in Keras -> TFLite converter
(post-training quantisation) -> Vela. The Keras copy is checked against PyTorch (float), and
the quantised model against the float one, before Vela runs.
"""
import argparse
import csv
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

from needle import Needle, NeedleConfig, count_macs, folded_convs, network_input, spectrogram

HERE = Path(__file__).parent


def build_keras(cfg: NeedleConfig, convs):
    import tensorflow as tf
    keras = tf.keras
    it = iter(convs)

    def conv(x, stride, act=True, k=3):
        w, b = next(it)
        layer = keras.layers.Conv2D(w.shape[0], k, strides=stride, padding="valid", activation="relu" if act else None)
        if k == 3:
            x = keras.layers.ZeroPadding2D(1)(x)          # same padding as PyTorch's padding=1
        y = layer(x)
        layer.set_weights([w.permute(2, 3, 1, 0).numpy(), b.numpy()])   # OIHW -> HWIO
        return y

    inp = keras.Input((cfg.bins, cfg.frames, cfg.audio_channels), batch_size=1)
    x, skips = inp, [inp]
    for _ in range(cfg.levels):
        x = conv(conv(x, 2), 1)
        skips.append(x)
    skips.pop()
    for _ in range(cfg.levels):
        # Resizing (not UpSampling2D, which becomes TILE + RESHAPE) -> RESIZE_NEAREST_NEIGHBOR,
        # which the U55 runs for 2x2 and which also exists for 16-bit activations.
        x = keras.layers.Resizing(x.shape[1] * 2, x.shape[2] * 2, interpolation="nearest")(x)
        x = keras.layers.Concatenate()([x, skips.pop()])
        x = conv(conv(x, 1), 1)
    x = conv(x, 1, act=False, k=1)
    return keras.Model(inp, keras.layers.Activation("sigmoid")(x))


def calibration_chunks(cfg: NeedleConfig, audio_dir, n=64, pattern="**/*", max_files=40):
    """Network inputs (NHWC float32) from real tracks, or log-normal noise without any.

    The quantiser fits each tensor's int8/int16 range to what it sees here, so the chunks should
    look like what the deck will get: full mixtures from many songs. Chunks are therefore taken
    at random positions spread over up to `max_files` files (e.g. --audio <MUSDB>/train
    --audio-glob "*/mixture.wav"), not all from the first file."""
    chunks = []
    if audio_dir:
        import soundfile as sf
        rng = np.random.default_rng(0)
        files = sorted(p for p in Path(audio_dir).glob(pattern) if p.suffix.lower() in (".wav", ".flac", ".mp3"))
        files = [files[i] for i in sorted(rng.permutation(len(files))[:max_files])]
        per_file = -(-n // max(1, len(files)))                   # ceil
        for f in files:
            wav, sr = sf.read(f, dtype="float32", always_2d=True)
            if sr != cfg.sample_rate or wav.shape[1] != cfg.audio_channels:
                continue
            x = network_input(spectrogram(torch.from_numpy(wav.T)[None], cfg), cfg)[0]      # (C, bins, T)
            if x.shape[-1] <= cfg.frames:
                continue
            for t0 in rng.integers(0, x.shape[-1] - cfg.frames, per_file):
                chunks.append(x[..., t0:t0 + cfg.frames].permute(1, 2, 0).numpy())
        rng.shuffle(chunks)
    if not chunks:
        rng = np.random.default_rng(0)
        chunks = [np.log1p(rng.lognormal(0, 1.5, (cfg.bins, cfg.frames, cfg.audio_channels))).astype(np.float32) for _ in range(n)]
    return chunks[:n]


def to_tflite(km, chunks, mode):
    import tensorflow as tf
    conv = tf.lite.TFLiteConverter.from_keras_model(km)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    conv.representative_dataset = lambda: ([c[None]] for c in chunks)
    if mode == "int8":
        conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        conv.inference_input_type = conv.inference_output_type = tf.int8
    else:   # 16-bit activations, 8-bit weights: better for audio, half the NPU speed
        conv.target_spec.supported_ops = [tf.lite.OpsSet.EXPERIMENTAL_TFLITE_BUILTINS_ACTIVATIONS_INT16_WEIGHTS_INT8]
        conv.inference_input_type = conv.inference_output_type = tf.int16
    return conv.convert()


def run_tflite(blob, x):
    import tensorflow as tf
    it = tf.lite.Interpreter(model_content=blob)
    it.allocate_tensors()
    i, o = it.get_input_details()[0], it.get_output_details()[0]
    s, z = i["quantization"]
    it.set_tensor(i["index"], np.round(x / s + z).clip(np.iinfo(i["dtype"]).min, np.iinfo(i["dtype"]).max).astype(i["dtype"]))
    it.invoke()
    s, z = o["quantization"]
    return (it.get_tensor(o["index"]).astype(np.float32) - z) * s


VELA_INI = """
; Ethos-U55-256 at 500 MHz as on the Renesas RA8P1. Bandwidths are estimates, not datasheet
; values: Vela's bandwidth = core_clock x clock_scale x 8 bytes (64-bit AXI).
[System_Config.RA8P1_SRAM]
; feature maps in on-chip SRAM, weights in on-chip MRAM
core_clock=500e6
axi0_port=Sram
axi1_port=OffChipFlash
Sram_clock_scale=1.0
Sram_burst_length=32
Sram_read_latency=32
Sram_write_latency=32
Sram_max_reads=4
Sram_max_writes=4
OffChipFlash_clock_scale=0.25
OffChipFlash_burst_length=64
OffChipFlash_read_latency=32
OffChipFlash_write_latency=0
OffChipFlash_max_reads=2
OffChipFlash_max_writes=0

[System_Config.RA8P1_SDRAM]
; feature maps in external 32-bit SDRAM (~350 MB/s usable assumed), weights in MRAM
core_clock=500e6
axi0_port=Sram
axi1_port=OffChipFlash
Sram_clock_scale=0.0875
Sram_burst_length=32
Sram_read_latency=100
Sram_write_latency=100
Sram_max_reads=2
Sram_max_writes=2
OffChipFlash_clock_scale=0.25
OffChipFlash_burst_length=64
OffChipFlash_read_latency=32
OffChipFlash_write_latency=0
OffChipFlash_max_reads=2
OffChipFlash_max_writes=0

[Memory_Mode.Shared_Sram_RA8P1]
const_mem_area=Axi1
arena_mem_area=Axi0
cache_mem_area=Axi0
"""


def vela(tflite_path: Path, system: str, out: Path):
    out.mkdir(parents=True, exist_ok=True)
    ini = out / "ra8p1_vela.ini"
    ini.write_text(VELA_INI)
    exe = Path(sys.executable).with_name("vela")
    cmd = [str(exe), str(tflite_path), "--accelerator-config", "ethos-u55-256", "--config", str(ini),
           "--system-config", system, "--memory-mode", "Shared_Sram_RA8P1", "--optimise", "Performance",
           "--output-dir", str(out / system), "--show-cpu-operations"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"vela failed:\n{r.stdout}\n{r.stderr}")
    summary = next((out / system).glob("*summary*.csv"))
    row = next(csv.DictReader(open(summary)))
    m = re.search(r"CPU operators = (\d+)", r.stdout)
    cpu_ops = int(m.group(1)) if m else None          # operators Vela leaves to the Cortex-M85
    return row, cpu_ops, r.stdout


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=int, default=16)
    ap.add_argument("--frames", type=int, default=64)
    ap.add_argument("--checkpoint")
    ap.add_argument("--audio", help="folder of 44.1 kHz stereo tracks for calibration (training data, never test)")
    ap.add_argument("--audio-glob", default="**/*", help='which files under --audio, e.g. "*/mixture.wav"')
    ap.add_argument("--calib-chunks", type=int, default=256)
    ap.add_argument("--modes", default="int8,int16")
    ap.add_argument("--out", default=str(HERE / "build"))
    a = ap.parse_args()

    if a.checkpoint:
        ck = torch.load(a.checkpoint, map_location="cpu")
        cfg = NeedleConfig(**{**ck["config"], "frames": a.frames})
        model = Needle(cfg)
        model.load_state_dict(ck["model"])
    else:
        cfg = NeedleConfig(base=a.base, frames=a.frames)
        model = Needle(cfg)
        torch.manual_seed(0)
    model.eval()
    out = Path(a.out) / f"needle_b{cfg.base}_f{cfg.frames}"
    out.mkdir(parents=True, exist_ok=True)
    report = {"config": {k: v for k, v in cfg.__dict__.items()}, **count_macs(cfg)}

    km = build_keras(cfg, folded_convs(model))
    chunks = calibration_chunks(cfg, a.audio, a.calib_chunks, a.audio_glob)
    x = chunks[0]
    with torch.no_grad():
        ref = model(torch.from_numpy(x).permute(2, 0, 1)[None]).permute(0, 2, 3, 1).numpy()
    report["keras_vs_pytorch_max_abs"] = float(np.abs(km(x[None]).numpy() - ref).max())
    assert report["keras_vs_pytorch_max_abs"] < 1e-4, report

    for mode in a.modes.split(","):
        blob = to_tflite(km, chunks, mode)
        path = out / f"needle_{mode}.tflite"
        path.write_bytes(blob)
        q = run_tflite(blob, x[None])
        entry = {"tflite_bytes": len(blob), "mask_mean_abs_error_vs_float": float(np.abs(q - ref).mean()),
                 "mask_max_abs_error_vs_float": float(np.abs(q - ref).max())}
        for system in ("RA8P1_SRAM", "RA8P1_SDRAM"):
            row, cpu_ops, _ = vela(path, system, out / mode)
            t = float(row["inference_time"])
            entry[system] = {
                "inference_ms_per_chunk": t * 1e3,
                "realtime_factor": t / report["chunk_s"],          # < 1 = faster than real time
                "npu_cycles": int(float(row.get("cycles_npu", 0) or 0)),
                "arena_kib": float(row.get("sram_memory_used", 0) or 0),
                "weights_kib": float(row.get("off_chip_flash_memory_used", 0) or 0),
                "cpu_operators": cpu_ops,                          # 0 = everything on the NPU
            }
        report[mode] = entry
    (out / "report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
