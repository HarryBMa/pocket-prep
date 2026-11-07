"""Needle on a Raspberry Pi (or any computer): separate a selected section of a track and play it.

This is the deck's "separate this section" button as a small program. It does what the RA8P1
will do, in the same order:
  1. cut the selected bars out of the track, plus 1 s of audio on each side as context
     (the deck has the whole track, so the edges of the selection are no harder than the middle),
  2. STFT (numpy here; the M85 will use CMSIS-DSP),
  3. the network in 0.37-s chunks with 8-frame overlaps (the int8 .tflite, the same file Vela
     compiles for the NPU),
  4. masks -> four stems, inverse STFT, crop to the selection.
Only numpy, soundfile and a TFLite interpreter are needed (no PyTorch).

    python needle_pi.py bench   --model needle_int8.tflite
    python needle_pi.py split   --model needle_int8.tflite --track song.wav --start 1:04 --bars 16 --bpm 126 --out stems/
    python needle_pi.py play    --model needle_int8.tflite --track song.wav --start 1:04 --bars 16 --bpm 126

`play` loops the section; type 1-4 + Enter to mute/unmute drums, bass, other, vocals; q quits.
The timing it prints is the "wait after pressing the button" on this computer.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

SOURCES = ["drums", "bass", "other", "vocals"]
SR, N_FFT, HOP, BINS = 44100, 2048, 512, 1024
OTHER = SOURCES.index("other")


# ── TFLite ──────────────────────────────────────────────────────────────────

def interpreter(path, threads):
    try:
        from ai_edge_litert.interpreter import Interpreter          # new name of tflite-runtime
    except ImportError:
        try:
            from tflite_runtime.interpreter import Interpreter
        except ImportError:
            import tensorflow as tf                                   # on a PC with TensorFlow
            Interpreter = tf.lite.Interpreter
    it = Interpreter(model_path=str(path), num_threads=threads)
    it.allocate_tensors()
    return it


class Net:
    """The quantised network: (C, bins, frames) float log-magnitudes -> (S*C, bins, frames) masks."""

    def __init__(self, path, threads=4):
        self.it = interpreter(path, threads)
        self.i, self.o = self.it.get_input_details()[0], self.it.get_output_details()[0]
        self.frames = int(self.i["shape"][2])                         # NHWC: (1, bins, frames, C)
        (self.si, self.zi), (self.so, self.zo) = self.i["quantization"], self.o["quantization"]
        info = np.iinfo(self.i["dtype"])
        self.lo, self.hi = info.min, info.max

    def __call__(self, x):
        x = np.transpose(x, (1, 2, 0))[None]                          # (C, bins, T) -> (1, bins, T, C)
        self.it.set_tensor(self.i["index"], np.clip(np.round(x / self.si + self.zi), self.lo, self.hi).astype(self.i["dtype"]))
        self.it.invoke()
        y = (self.it.get_tensor(self.o["index"]).astype(np.float32) - self.zo) * self.so
        return np.transpose(y[0], (2, 0, 1))                          # (S*C, bins, T)


# ── STFT, the same as torch.stft(center=True) / torch.istft with a periodic Hann window ─────

WINDOW = (0.5 - 0.5 * np.cos(2 * np.pi * np.arange(N_FFT) / N_FFT)).astype(np.float32)
try:
    from scipy import fft as _fft                                     # float32 FFTs on all cores
    FFT_KW = {"workers": -1}
except ImportError:
    _fft, FFT_KW = np.fft, {}


def stft(x):
    """x (C, N) -> complex (C, 1025, 1 + N // HOP)."""
    x = np.pad(x, ((0, 0), (N_FFT // 2, N_FFT // 2)), mode="reflect")
    t = 1 + (x.shape[1] - N_FFT) // HOP
    frames = np.lib.stride_tricks.sliding_window_view(x, N_FFT, axis=1)[:, ::HOP][:, :t]   # (C, T, N_FFT), no copy
    return _fft.rfft(frames * WINDOW, axis=-1, **FFT_KW).transpose(0, 2, 1).astype(np.complex64)


def istft(spec, length):
    """complex (..., 1025, T) -> (..., length): windowed overlap-add divided by the window envelope.
    A frame is 4 hops long, so overlap-add is 4 shifted additions of hop-sized blocks."""
    *lead, _, t = spec.shape
    frames = _fft.irfft(np.swapaxes(spec, -1, -2), n=N_FFT, axis=-1, **FFT_KW).astype(np.float32) * WINDOW
    k = N_FFT // HOP
    frames = frames.reshape(*lead, t, k, HOP)
    out = np.zeros((*lead, t + k - 1, HOP), np.float32)
    env = np.zeros((t + k - 1, HOP), np.float32)
    w2 = (WINDOW ** 2).reshape(k, HOP)
    for j in range(k):
        out[..., j:j + t, :] += frames[..., j, :]
        env[j:j + t] += w2[j]
    out = (out / np.maximum(env, 1e-11)).reshape(*lead, -1)
    return out[..., N_FFT // 2:N_FFT // 2 + length]


# ── Chunked separation (chunked.py without PyTorch) ─────────────────────────

def crossfade(frames, overlap):
    w = np.ones(frames, np.float32)
    ramp = (np.arange(overlap) + 0.5) / overlap
    w[:overlap], w[-overlap:] = ramp, ramp[::-1]
    return w


def separate(net, wav, overlap=8):
    """wav (2, N) float -> stems (4, 2, N)."""
    spec = stft(wav)
    c, f, t = spec.shape
    x = np.log1p(np.abs(spec[:, :BINS]))
    frames, step = net.frames, net.frames - overlap
    n = max(1, -(-(t - overlap) // step))
    t_pad = (n - 1) * step + frames
    x = np.pad(x, ((0, 0), (0, 0), (0, t_pad - t)))
    w = crossfade(frames, overlap)
    acc = np.zeros((len(SOURCES) * c, BINS, t_pad), np.float32)
    wsum = np.zeros(t_pad, np.float32)
    for k in range(n):
        a = k * step
        acc[..., a:a + frames] += net(x[..., a:a + frames]) * w
        wsum[a:a + frames] += w
    m = (acc / wsum)[..., :t].reshape(len(SOURCES), c, BINS, t)
    # Normalise so the four masks sum to 1 and the stems add up to the mix. The int8 model can
    # output exactly 0 for all four sources in a bin (sigmoid rounded to 0); then share it equally
    # instead of dropping it, so the stems still sum to the mix.
    total = m.sum(0, keepdims=True)
    m = np.where(total > 1e-3, m / np.maximum(total, 1e-3), 1.0 / len(SOURCES))
    full = np.zeros((len(SOURCES), c, f, t), np.float32)
    full[:, :, :BINS] = m
    full[OTHER, :, BINS:] = 1.0                                       # the Nyquist bin goes to "other"
    return istft(full * spec, wav.shape[1])                           # all 8 stem channels at once


# ── Selecting the section ───────────────────────────────────────────────────

def seconds(text):
    """'64', '64.5' or '1:04' -> seconds."""
    parts = [float(p) for p in str(text).split(":")]
    return sum(p * 60 ** i for i, p in enumerate(reversed(parts)))


def read_track(path):
    wav, sr = sf.read(str(path), dtype="float32", always_2d=True)
    if sr != SR:
        sys.exit(f"{path} is {sr} Hz; Needle expects 44.1 kHz (convert it first, e.g. with ffmpeg -ar 44100)")
    wav = wav.T
    return np.repeat(wav, 2, 0) if wav.shape[0] == 1 else wav[:2]


def split_section(net, track, start, length, context=1.0):
    """Separate [start, start+length) seconds of `track` (2, N), using `context` seconds on each
    side. Returns stems (4, 2, length samples) and the time it took."""
    a, b = int(start * SR), int((start + length) * SR)
    a0, b0 = max(0, a - int(context * SR)), min(track.shape[1], b + int(context * SR))
    t = time.perf_counter()
    stems = separate(net, np.ascontiguousarray(track[:, a0:b0]))
    return stems[..., a - a0:b - a0], time.perf_counter() - t


def section_args(a):
    length = a.seconds if a.seconds else a.bars * 4 * 60.0 / a.bpm    # 4/4: 4 beats per bar
    return seconds(a.start), length


# ── Commands ────────────────────────────────────────────────────────────────

def cmd_bench(a):
    net = Net(a.model, a.threads)
    x = np.random.rand(2, BINS, net.frames).astype(np.float32) * 3
    net(x)                                                            # warm-up
    t = time.perf_counter()
    for _ in range(a.n):
        net(x)
    per = (time.perf_counter() - t) / a.n
    step = (net.frames - 8) * HOP / SR
    print(f"network: {per * 1000:.1f} ms per chunk ({net.frames} frames), one chunk every {step * 1000:.0f} ms of audio")
    wav = (np.random.randn(2, 30 * SR) * 0.1).astype(np.float32)
    t = time.perf_counter()
    separate(net, wav)
    full = time.perf_counter() - t
    print(f"whole pipeline (STFT + network + iSTFT): {full:.2f} s for 30 s of audio "
          f"(= 16 bars at 128 BPM), {30 / full:.1f}x faster than real time")


def cmd_split(a):
    net = Net(a.model, a.threads)
    track = read_track(a.track)
    start, length = section_args(a)
    stems, took = split_section(net, track, start, length)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for s, name in enumerate(SOURCES):
        sf.write(out / f"{name}.wav", stems[s].T, SR, subtype="PCM_16")
    print(f"{length:.1f} s section separated in {took:.2f} s ({length / took:.1f}x real time) -> {out}")
    (out / "section.json").write_text(json.dumps({"track": str(a.track), "start": start, "seconds": length}))


def cmd_play(a):
    import sounddevice as sd
    net = Net(a.model, a.threads)
    track = read_track(a.track)
    start, length = section_args(a)
    print("separating ...", flush=True)
    stems, took = split_section(net, track, start, length)
    print(f"ready after {took:.2f} s. Type 1-4 + Enter to mute/unmute "
          + ", ".join(f"{i + 1}={n}" for i, n in enumerate(SOURCES)) + "; q quits.")
    on = np.ones(len(SOURCES), np.float32)
    pos = [0]
    n = stems.shape[-1]

    def callback(outdata, frames, _time, _status):
        idx = (pos[0] + np.arange(frames)) % n                        # loop the section
        outdata[:] = np.einsum("s,scn->nc", on, stems[..., idx])
        pos[0] = (pos[0] + frames) % n

    with sd.OutputStream(samplerate=SR, channels=2, dtype="float32", callback=callback, blocksize=512):
        for line in sys.stdin:
            line = line.strip()
            if line == "q":
                break
            if line in ("1", "2", "3", "4"):
                k = int(line) - 1
                on[k] = 1.0 - on[k]
            print("  " + "  ".join(f"{n}:{'on' if v else 'off'}" for n, v in zip(SOURCES, on)))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("bench", "split", "play"):
        p = sub.add_parser(name)
        p.add_argument("--model", required=True, help="needle_int8.tflite")
        p.add_argument("--threads", type=int, default=4)
        if name == "bench":
            p.add_argument("--n", type=int, default=50)
        else:
            p.add_argument("--track", required=True, help="44.1 kHz WAV / FLAC (MP3 if your libsndfile reads it)")
            p.add_argument("--start", default="0", help="start of the section: seconds or m:ss")
            p.add_argument("--bars", type=float, default=16)
            p.add_argument("--bpm", type=float, default=128)
            p.add_argument("--seconds", type=float, help="section length in seconds (instead of bars/bpm)")
        if name == "split":
            p.add_argument("--out", default="stems")
    a = ap.parse_args()
    {"bench": cmd_bench, "split": cmd_split, "play": cmd_play}[a.cmd](a)


if __name__ == "__main__":
    main()
