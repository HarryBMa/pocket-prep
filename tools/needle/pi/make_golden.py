"""Golden test vectors for porting Needle to another language (e.g. DECK001's C++).

Runs the reference pipeline (needle_pi.py) on a fixed synthetic signal and saves every stage as
raw little-endian float32 / int8 files plus golden.json describing shapes and layouts, so a port
can be checked stage by stage: input -> STFT -> network input -> quantised chunk 0 -> masks of
chunk 0 -> crossfaded masks -> stems. No music is included (the signal is generated).

    python make_golden.py --model needle_int8.tflite --out golden
"""
import argparse
import json
from pathlib import Path

import numpy as np

import needle_pi as P


def signal(seconds=6.0, seed=1234):
    """Deterministic stereo test signal: kick-like thumps, a bass tone, a chord and noise bursts."""
    rng = np.random.default_rng(seed)
    n = int(seconds * P.SR)
    t = np.arange(n) / P.SR
    beat = (t * 2.0) % 1.0                                   # 120 BPM
    kick = np.sin(2 * np.pi * (50 + 80 * np.exp(-beat * 30)) * t) * np.exp(-beat * 12)
    bass = 0.3 * np.sin(2 * np.pi * 55 * t) * (beat < 0.5)
    chord = 0.1 * sum(np.sin(2 * np.pi * f * t) for f in (261.6, 329.6, 392.0))
    hats = 0.05 * rng.standard_normal(n) * (((t * 4.0) % 1.0) < 0.05)
    left = 0.5 * kick + bass + chord + hats
    right = 0.5 * kick + bass + 0.8 * chord + hats
    return np.stack([left, right]).astype(np.float32)


def save(out, name, arr, meta, desc):
    arr = np.ascontiguousarray(arr)
    (out / f"{name}.bin").write_bytes(arr.tobytes())
    meta[name] = {"file": f"{name}.bin", "dtype": str(arr.dtype), "shape": list(arr.shape), "about": desc}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", default="golden")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    net = P.Net(a.model, threads=1)
    meta = {"sample_rate": P.SR, "n_fft": P.N_FFT, "hop": P.HOP, "bins": P.BINS, "frames": net.frames,
            "overlap": 8, "sources": P.SOURCES,
            "input_quant": {"scale": float(net.si), "zero_point": int(net.zi), "dtype": str(np.dtype(net.i["dtype"]))},
            "output_quant": {"scale": float(net.so), "zero_point": int(net.zo), "dtype": str(np.dtype(net.o["dtype"]))},
            "byte_order": "little-endian, C order (last index fastest)", "arrays": {}}
    m = meta["arrays"]

    wav = signal()
    save(out, "input_wav", wav, m, "input audio (channel, sample), float32 in [-1, 1]")
    spec = P.stft(wav)
    save(out, "stft_re", spec.real, m, "STFT real part (channel, bin 0..1024, frame)")
    save(out, "stft_im", spec.imag, m, "STFT imaginary part (channel, bin, frame)")
    x = np.log1p(np.abs(spec[:, :P.BINS])).astype(np.float32)
    save(out, "net_input", x, m, "log1p(|STFT|) of bins 0..1023 (channel, bin, frame)")
    chunk = x[..., :net.frames]
    nhwc = np.transpose(chunk, (1, 2, 0))[None]
    q = np.clip(np.round(nhwc / net.si + net.zi), net.lo, net.hi).astype(net.i["dtype"])
    save(out, "chunk0_quantised", q, m, "network input tensor of chunk 0, NHWC (1, bin, frame, channel)")
    net.it.set_tensor(net.i["index"], q)
    net.it.invoke()
    raw = net.it.get_tensor(net.o["index"])
    save(out, "chunk0_output_raw", raw, m, "network output tensor of chunk 0, NHWC (1, bin, frame, k); k = source*2 + channel")
    save(out, "chunk0_masks", net(chunk), m, "dequantised masks of chunk 0 (k, bin, frame); k = source*2 + channel")
    stems = P.separate(net, wav)
    save(out, "stems", stems, m, "final stems (source, channel, sample); sources drums, bass, other, vocals")
    (out / "golden.json").write_text(json.dumps(meta, indent=1))
    err = np.abs(stems.sum(0) - wav)[:, P.N_FFT:-P.N_FFT].max()
    print(f"wrote {len(m)} arrays to {out}; stems sum to input within {err:.1e}")


if __name__ == "__main__":
    main()
