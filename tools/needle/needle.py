"""Needle: a tiny all-convolution stem separator meant for an Arm Ethos-U55 NPU (Renesas RA8P1).

Why it looks like this (every choice comes from what the U55 can run, see tools/needle/README.md):
- Only CONV_2D (3x3, stride 1 or 2x2), RELU, CONCATENATION, nearest 2x2 upsampling and LOGISTIC.
  All of these run on the NPU; no attention, no LSTM, no normalisation layers at inference
  (BatchNorm is folded into the convolutions on export).
- Magnitude masks: the network sees log-magnitude spectrograms and outputs one mask per source
  and audio channel. The STFT, the log, applying the masks to the complex mixture and the iSTFT
  run on the Cortex-M85 (Helium FFT), not on the NPU.
- 2x2 strides everywhere: Vela only accelerates nearest-neighbour upsampling with equal H/W
  factors, so every level halves frequency and time together (like Spleeter's U-Net).

Shapes (PyTorch, NCHW): input  (B, audio_channels, bins, frames)   log1p(|STFT|)
                         output (B, sources*audio_channels, bins, frames)   masks in 0..1
Channel order of the output: source-major, i.e. [drums L, drums R, bass L, bass R, ...].
"""
from dataclasses import dataclass, field, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F

SOURCES = ("drums", "bass", "other", "vocals")   # same order as Demucs / MUSDB18


@dataclass
class NeedleConfig:
    base: int = 16                                # channels at the first level
    mult: tuple = (1, 2, 3, 4, 6)                 # channels per level = base * mult
    sources: int = 4
    audio_channels: int = 2
    sample_rate: int = 44100
    n_fft: int = 2048
    hop: int = 512
    bins: int = 1024                              # STFT bins the network sees (drops Nyquist)
    frames: int = 64                              # frames per chunk on the device (0.74 s)

    @property
    def levels(self):
        return len(self.mult)

    def check(self):
        assert self.bins % 2 ** self.levels == 0 and self.frames % 2 ** self.levels == 0, \
            f"bins and frames must be divisible by {2 ** self.levels}"
        assert self.bins <= self.n_fft // 2


class Block(nn.Module):
    """3x3 conv + BN + ReLU, twice. The first conv may stride 2x2."""

    def __init__(self, cin, cout, stride):
        super().__init__()
        self.c1, self.b1 = nn.Conv2d(cin, cout, 3, stride, 1), nn.BatchNorm2d(cout)
        self.c2, self.b2 = nn.Conv2d(cout, cout, 3, 1, 1), nn.BatchNorm2d(cout)

    def forward(self, x):
        x = F.relu(self.b1(self.c1(x)))
        return F.relu(self.b2(self.c2(x)))


class Needle(nn.Module):
    def __init__(self, cfg: NeedleConfig = NeedleConfig()):
        super().__init__()
        cfg.check()
        self.cfg = cfg
        chans = [cfg.base * m for m in cfg.mult]
        self.enc = nn.ModuleList()
        cin = cfg.audio_channels
        for c in chans:
            self.enc.append(Block(cin, c, 2))
            cin = c
        self.dec = nn.ModuleList()
        for i in reversed(range(len(chans))):          # upsample, concat the skip, refine
            skip = chans[i - 1] if i > 0 else cfg.audio_channels
            out = chans[i - 1] if i > 0 else cfg.base
            self.dec.append(Block(cin + skip, out, 1))
            cin = out
        self.head = nn.Conv2d(cin, cfg.sources * cfg.audio_channels, 1)

    def forward(self, x):
        skips = [x]
        for b in self.enc:
            x = b(x)
            skips.append(x)
        skips.pop()
        for b in self.dec:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
            x = b(torch.cat([x, skips.pop()], 1))
        return torch.sigmoid(self.head(x))


# ── Front and back end (on the device these run on the Cortex-M85) ───────────

def spectrogram(wav, cfg: NeedleConfig):
    """wav (B, C, N) -> complex STFT (B, C, n_fft//2+1, T)."""
    b, c, n = wav.shape
    spec = torch.stft(wav.reshape(b * c, n), cfg.n_fft, cfg.hop, window=torch.hann_window(cfg.n_fft, device=wav.device),
                      return_complex=True, center=True)
    return spec.reshape(b, c, spec.shape[-2], spec.shape[-1])


def network_input(spec, cfg: NeedleConfig):
    """Complex STFT -> what the network sees: log1p magnitude of the first `bins` bins."""
    return torch.log1p(spec[:, :, : cfg.bins].abs())


def apply_masks(masks, spec, cfg: NeedleConfig, eps=1e-4):
    """Masks (B, S*C, bins, T) + mixture STFT -> per-source STFT (B, S, C, F, T).
    The masks are normalised to sum to 1 per bin, so the stems add back up to the mix.
    Bins above `bins` (only the Nyquist bin by default) go to "other"."""
    b, c, f, t = spec.shape
    m = masks.reshape(b, cfg.sources, c, cfg.bins, t)
    m = m / (m.sum(1, keepdim=True) + eps)
    full = torch.zeros(b, cfg.sources, c, f, t, device=spec.device)
    full[:, :, :, : cfg.bins] = m
    full[:, SOURCES.index("other"), :, cfg.bins:] = 1.0
    return full * spec.unsqueeze(1)


def istft(spec, cfg: NeedleConfig, length):
    """Per-source STFT (B, S, C, F, T) -> waveforms (B, S, C, length)."""
    b, s, c, f, t = spec.shape
    wav = torch.istft(spec.reshape(b * s * c, f, t), cfg.n_fft, cfg.hop, window=torch.hann_window(cfg.n_fft, device=spec.device),
                      length=length)
    return wav.reshape(b, s, c, length)


def separate(model: Needle, wav):
    """Float reference separation of a whole track (B, C, N) -> (B, S, C, N). Pads the frame
    count to a multiple of 2**levels; on the device the same is done chunk by chunk."""
    cfg = model.cfg
    spec = spectrogram(wav, cfg)
    t = spec.shape[-1]
    pad = (-t) % (2 ** cfg.levels)
    x = F.pad(network_input(spec, cfg), (0, pad))
    masks = model(x)[..., :t]
    return istft(apply_masks(masks, spec, cfg), cfg, wav.shape[-1])


# ── Export helpers ───────────────────────────────────────────────────────────

def folded_convs(model: Needle):
    """Every conv in forward order with BatchNorm folded in: list of (weight OIHW, bias O)."""
    out = []

    def fold(conv, bn):
        w, b = conv.weight.detach(), conv.bias.detach()
        if bn is None:
            return w.clone(), b.clone()
        scale = bn.weight.detach() / torch.sqrt(bn.running_var + bn.eps)
        return w * scale[:, None, None, None], (b - bn.running_mean) * scale + bn.bias.detach()

    for blk in list(model.enc) + list(model.dec):
        out += [fold(blk.c1, blk.b1), fold(blk.c2, blk.b2)]
    out.append(fold(model.head, None))
    return out


def count_macs(cfg: NeedleConfig):
    """Multiply-accumulates per chunk and per second of audio, and the parameter count."""
    m = Needle(cfg).eval()
    total = 0
    def hook(mod, inp, outp):
        nonlocal total
        k = mod.kernel_size[0] * mod.kernel_size[1] * mod.in_channels
        total += outp.numel() * k
    hs = [mod.register_forward_hook(hook) for mod in m.modules() if isinstance(mod, nn.Conv2d)]
    with torch.no_grad():
        m(torch.zeros(1, cfg.audio_channels, cfg.bins, cfg.frames))
    for h in hs:
        h.remove()
    chunk_s = cfg.frames * cfg.hop / cfg.sample_rate
    params = sum(p.numel() for p in m.parameters())
    return {"params": params, "macs_per_chunk": total, "gmac_per_audio_s": total / chunk_s / 1e9, "chunk_s": chunk_s}


def selftest():
    """Shapes, and that the four stems add back up to the mixture (normalised masks)."""
    torch.manual_seed(0)
    cfg = NeedleConfig(base=8)
    model = Needle(cfg).eval()
    wav = torch.randn(1, 2, cfg.sample_rate * 3) * 0.1
    with torch.no_grad():
        stems = separate(model, wav)
    assert stems.shape == (1, 4, 2, wav.shape[-1]), stems.shape
    err = (stems.sum(1) - wav).abs().max().item()
    assert err < 1e-3, f"stems don't sum to the mix: {err}"
    print("selftest ok: stems", tuple(stems.shape), "sum-to-mix error", f"{err:.2e}")


if __name__ == "__main__":
    selftest()
    for base in (8, 12, 16, 24):
        cfg = NeedleConfig(base=base)
        print(f"base={base}", count_macs(cfg))
