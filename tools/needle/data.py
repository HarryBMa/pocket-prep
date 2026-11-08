"""Training data for Needle: MUSDB18-HQ stems and teacher stems (distill.py), kept in RAM.

Where the data comes from
- MUSDB18-HQ: <root>/train/<track>/{drums,bass,other,vocals}.wav (44.1 kHz, 16-bit stereo).
  The 14-track validation split is the official one that ships with the `musdb` package (the
  same split Open-Unmix and Demucs use), so validation numbers are comparable with papers.
  The 50 test tracks are never loaded by training.
- Teacher stems: <cache>/<id>/{drums,bass,other,vocals}.flac written by distill.py.

Why it is kept like this
- 16-bit integers in RAM: 10.6 MB per minute per stem (half of float32). The 86 MUSDB training
  tracks are about 14 GB for four stems, so even a few hundred extra teacher tracks fit in 128 GB.
- No DataLoader worker processes: on Windows every worker is a fresh process that would need its
  own copy of the audio. Cutting a batch of crops is only memory copies, so two background
  threads keep a queue of ready batches, and all the arithmetic (augmentation, STFT) runs on
  the GPU.
"""
import ctypes
import os
import queue
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from needle import SOURCES

SAMPLE_RATE = 44100


# ── Reading audio ────────────────────────────────────────────────────────────

def read_audio(path, sample_rate=SAMPLE_RATE, dtype="int16"):
    """Audio file -> array (2, N) at `sample_rate`. Mono is duplicated to stereo, more than two
    channels are cut to the first two. Anything not at 44.1 kHz is resampled in memory
    (the file on disk is never touched)."""
    src = path if hasattr(path, "read") else str(path)                   # a path, or a file-like (e.g. from a zip)
    wav, sr = sf.read(src, dtype="float32", always_2d=True)              # (N, C)
    wav = torch.from_numpy(wav.T.copy())
    if wav.shape[0] == 1:
        wav = wav.repeat(2, 1)
    wav = wav[:2]
    if sr != sample_rate:
        import torchaudio
        wav = torchaudio.functional.resample(wav, sr, sample_rate)
    if dtype == "int16":
        return (wav.clamp(-1, 1 - 1 / 32768) * 32768).round().to(torch.int16).numpy()
    return wav.numpy()


def read_stems(folder, ext, dtype="int16"):
    """<folder>/{drums,bass,other,vocals}.<ext> -> (4, 2, N). Stems are cut to the shortest."""
    stems = [read_audio(Path(folder) / f"{s}.{ext}", dtype=dtype) for s in SOURCES]
    n = min(s.shape[-1] for s in stems)
    return np.stack([s[:, :n] for s in stems])


# ── MUSDB18-HQ ───────────────────────────────────────────────────────────────

def musdb_split(root):
    """Track names of the official split: {"train": 86, "valid": 14, "test": 50}."""
    import musdb
    def names(**kw):
        return sorted(t.name for t in musdb.DB(root=str(root), is_wav=True, **kw))
    return {"train": names(subsets="train", split="train"), "valid": names(subsets="train", split="valid"),
            "test": names(subsets="test")}


# ── Disk-backed arrays ───────────────────────────────────────────────────────
# Decoded int16 stems are written once to .npy files on an SSD and memory-mapped. Windows then
# keeps them in its file cache (free RAM) without reserving them for the process: when another
# program needs memory, cached pages are dropped and re-read, instead of an allocation failing.
# With ~80 GB of data on a 128 GB PC that is the difference between a run finishing or not.

MMAP_DIR = None          # set by train.py --cache; None keeps everything in process memory


def cached(folder, tag, make):
    """make() -> array, saved under MMAP_DIR (keyed by the folder and `tag`) and memory-mapped."""
    if MMAP_DIR is None:
        return make()
    folder = Path(folder)
    path = Path(MMAP_DIR) / f"{folder.parent.name}__{folder.name}{tag}.npy"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.stem + ".tmp.npy")
        np.save(tmp, make())
        os.replace(tmp, path)
    return np.load(path, mmap_mode="r")


def load_musdb(root, subset, names, threads=8):
    """Load the stems of the named tracks from <root>/<subset>/ -> list of (name, int16 (4, 2, N))."""
    folders = [Path(root) / subset / n for n in names]
    with ThreadPoolExecutor(threads) as ex:
        return list(zip(names, ex.map(lambda f: cached(f, "", lambda: read_stems(f, "wav")), folders)))


def load_teacher(cache, threads=8):
    """Load every finished track of a distill.py cache -> list of (id, int16 (4, 2, N))."""
    folders = sorted(p.parent for p in Path(cache).glob("*/done"))
    with ThreadPoolExecutor(threads) as ex:
        return list(zip((f.name for f in folders),
                        ex.map(lambda f: cached(f, "", lambda: read_stems(f, "flac")), folders)))


def busiest_excerpt(stems, seconds):
    """stems int16 (S, C, N) -> the `seconds`-long excerpt whose quietest stem is loudest (1-s
    steps): the part of a song where all four stems play, which teaches the most per second.
    Lets hundreds of long DJ tracks fit in RAM (2 minutes each instead of 6+)."""
    n = stems.shape[-1]
    secs, L = n // SAMPLE_RATE, seconds * SAMPLE_RATE
    if n <= L:
        return stems
    p = (stems[..., :secs * SAMPLE_RATE].astype(np.float32) ** 2).mean(1)
    p = p.reshape(len(stems), secs, SAMPLE_RATE).mean(-1)              # (S, secs)
    win = np.stack([np.convolve(q, np.ones(seconds), "valid") for q in p]).min(0)
    start = int(win.argmax()) * SAMPLE_RATE
    return np.ascontiguousarray(stems[..., start:start + L])


def load_excerpts(cache, seconds, threads=8):
    """Like load_teacher, but keeps only the busiest `seconds` of each track."""
    folders = sorted(p.parent for p in Path(cache).glob("*/done"))
    with ThreadPoolExecutor(threads) as ex:
        return list(zip((f.name for f in folders),
                        ex.map(lambda f: cached(f, f"_busiest{seconds}",
                                                lambda: busiest_excerpt(read_stems(f, "flac"), seconds)), folders)))


def hours(tracks):
    return sum(t.shape[-1] for _, t in tracks) / SAMPLE_RATE / 3600


# ── Random crops ─────────────────────────────────────────────────────────────

class StemBank:
    """Draws random training crops from groups of tracks.

    groups: list of (tracks, share). `share` is the fraction of stems drawn from that group
    (e.g. MUSDB 0.5, teacher 0.5); inside a group, longer tracks are drawn more often, so every
    second of audio is equally likely.

    Remixing: with probability `remix`, each of the four stems of a training example comes from
    a different random track (drums from one song, vocals from another, ...). The model then
    sees far more distinct mixtures than there are songs, which is the strongest augmentation
    for a small dataset. Otherwise all four stems come from the same place in the same song.
    """

    def __init__(self, groups, length):
        self.length = length
        self.tracks, p = [], []
        for tracks, share in groups:
            tracks = [t for _, t in tracks if t.shape[-1] > length]
            lens = np.array([t.shape[-1] for t in tracks], dtype=np.float64)
            self.tracks += tracks
            p.append(share * lens / lens.sum())
        p = np.concatenate(p)
        self.p = p / p.sum()

    def batch(self, n, remix, rng):
        S, C = self.tracks[0].shape[:2]
        out = np.empty((n, S, C, self.length), dtype=np.int16)
        for i in range(n):
            idx = rng.choice(len(self.tracks), size=S, p=self.p) if rng.random() < remix \
                else np.repeat(rng.choice(len(self.tracks), p=self.p), S)
            if idx[0] == idx[-1] and (idx == idx[0]).all():          # same song: same position
                t = self.tracks[idx[0]]
                o = rng.integers(0, t.shape[-1] - self.length)
                out[i] = t[:, :, o:o + self.length]
            else:
                for s, k in enumerate(idx):
                    t = self.tracks[k]
                    o = rng.integers(0, t.shape[-1] - self.length)
                    out[i, s] = t[s, :, o:o + self.length]
        return out


class Prefetcher:
    """Background threads that keep `depth` batches ready in pinned memory, so copying them
    to the GPU overlaps with the training step."""

    def __init__(self, bank, batch, remix, seed=0, depth=8, threads=2):
        self.q = queue.Queue(depth)
        for i in range(threads):
            threading.Thread(target=self._run, args=(bank, batch, remix, seed * 100 + i), daemon=True).start()

    def _run(self, bank, batch, remix, seed):
        rng = np.random.default_rng(seed)
        failures = 0
        while True:
            try:
                self.q.put(torch.from_numpy(bank.batch(batch, remix, rng)).pin_memory())
                failures = 0
            except MemoryError as e:                 # brief memory pressure: wait and retry
                failures += 1
                if failures > 20:
                    self.q.put(e)
                    return
                time.sleep(15)
            except Exception as e:                   # anything else: hand it to the training loop
                self.q.put(e)
                return

    def next(self):
        """The next batch. A worker that failed passes its exception on, so training stops with
        the error instead of waiting forever (which is what happened before)."""
        b = self.q.get()
        if isinstance(b, BaseException):
            raise RuntimeError("a data thread failed") from b
        return b


# ── Augmentation (on the GPU) ────────────────────────────────────────────────

def augment(stems, gain_db=6.0, p_swap=0.5, p_flip=0.5):
    """stems (B, S, C, L) float. Each stem of each example independently gets
    - a random gain, uniform in +-gain_db dB (new balances between instruments),
    - its left and right channels swapped with probability p_swap,
    - its polarity flipped with probability p_flip (the ear can't hear it, the model must not care).
    The mixture is built afterwards as the sum of the augmented stems, so it always matches."""
    b, s = stems.shape[:2]
    shape = (b, s, 1, 1)
    gain = 10 ** (torch.empty(shape, device=stems.device).uniform_(-gain_db, gain_db) / 20)
    sign = torch.where(torch.rand(shape, device=stems.device) < p_flip, -1.0, 1.0)
    stems = stems * (gain * sign)
    swap = torch.rand(shape, device=stems.device) < p_swap
    return torch.where(swap, stems.flip(2), stems)


# ── Keeping a Windows PC awake during long runs ──────────────────────────────

def keep_awake():
    """Ask Windows not to go to sleep while this process runs (the screen may still turn off).
    No settings are changed; the request ends when the process ends."""
    if sys.platform == "win32":
        ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
