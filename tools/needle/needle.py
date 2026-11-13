"""Needle: a tiny all-convolution stem separator meant for an Arm Ethos-U55 NPU (Renesas RA8P1).

This file is Needle 1 plus everything the two models share: the config, the front end (STFT, network input) and
the back end (masks / deep filters, iSTFT). Needle 2, the band-split model (NeedleConfig.arch = "bs"), is in
needle2.py; Needle(cfg) builds whichever the config asks for.

Why it looks like this (every choice comes from what the U55 can run, see tools/needle/README.md):
- Only CONV_2D (3x3, stride 1 or 2x2), RELU, CONCATENATION, nearest 2x2 upsampling and LOGISTIC.
  All of these run on the NPU; no attention, no LSTM, no normalisation layers at inference
  (BatchNorm is folded into the convolutions on export).
- Magnitude masks: the network sees log-magnitude spectrograms and outputs one mask per source
  and audio channel. The STFT, the log, applying the masks to the complex mixture and the iSTFT
  run on the Cortex-M85 (Helium FFT), not on the NPU.
- 2x2 strides everywhere: Vela only accelerates nearest-neighbour upsampling with equal H/W
  factors, so every level halves frequency and time together (like Spleeter's U-Net).

Shapes (PyTorch, NCHW): input  (B, in_channels, bins, frames)   log1p(|STFT|) [+ coordinate and harmonic channels]
                         output (B, sources*audio_channels, bins, frames)   masks in 0..1
Channel order of the output: source-major, i.e. [drums L, drums R, bass L, bass R, ...].

Optional "frontier" features (all off by default, so the plain model and its checkpoints are unchanged):
- coord / harmonics: extra input channels computed on the M85 (see network_input);
- taps >= 1: deep filtering instead of masks (see apply_filters): the head outputs, per source except
  "other", channel and bin, a complex filter over `taps` neighbouring STFT frames (tanh instead of
  sigmoid), and "other" is the residual. Same layer types, so still 0 CPU operators in Vela.
"""
import math
from dataclasses import dataclass

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
    # Input features beyond the log-magnitudes (all computed on the M85, no NPU cost):
    coord: bool = False                           # + 1 channel: bin index / bins (CoordConv)
    harmonics: tuple = ()                         # + 1 channel per entry h: mono log-magnitude at bin h*k
    # Output: 0 = sigmoid magnitude masks (sum to 1); taps >= 1 = deep filtering: per source
    # (except "other"), channel and bin a complex FIR over `taps` frames (1 = complex mask);
    # "other" is the residual, so the stems still add up to the mix.
    taps: int = 0
    # ── Needle 2 (arch = "bs", see needle2.py). The defaults leave Needle 1 and its checkpoints unchanged. ──
    arch: str = "unet"                            # "unet" = Needle 1 (this file); "bs" = Needle 2, the band-split model
    cplx: bool = False                            # + 2 input channels per audio channel: the STFT's real and imaginary
                                                  # part, log-compressed like the magnitudes (lets the filters use phase)
    bands: tuple = ((128, 1), (384, 2), (512, 4), (1024, 16))   # bs: (bins, bins per row), from the bottom of the spectrum up
    widths: tuple = (32, 64, 96, 128, 160)        # bs: channels per level (level 0 = the band grid)
    tsteps: int = 2                               # bs: the first `tsteps` down-samplings halve time too, the rest frequency only
    tdf: tuple = (0, 0, 4, 2, 2)                  # bs: TDF bottleneck factor per level (0 = no TDF at that level)
    rnn: int = 2                                  # bs: dual-path GRU blocks at the deepest level
    rnn_hidden: int = 96                          # bs: GRU state size per direction
    sub: int = 0                                  # bs: + `sub` input channels per audio channel: the mixture's envelope in
                                                  # `sub` time slots inside each frame's hop (subframe_envelope), so short
                                                  # hits are visible although the STFT window is long. 0 = off

    def __post_init__(self):
        self.mult, self.harmonics = tuple(self.mult), tuple(self.harmonics)
        self.bands = tuple(tuple(b) for b in self.bands)
        self.widths, self.tdf = tuple(self.widths), tuple(self.tdf)

    @property
    def levels(self):
        return len(self.mult)

    @property
    def time_multiple(self):
        """A chunk's frame count must be a multiple of this (how far the network down-samples time)."""
        return 2 ** (self.tsteps if self.arch == "bs" else self.levels)

    @property
    def rows(self):
        """bs: rows of the band grid (one bin per row at the bottom of the spectrum, several further up)."""
        return sum(b // s for b, s in self.bands)

    @property
    def context(self):
        """Frames of mixture STFT the output stage looks at on each side of a chunk (deep filtering)."""
        return self.taps // 2

    @property
    def in_channels(self):
        return (self.audio_channels + int(self.coord) + len(self.harmonics) + (2 * self.audio_channels if self.cplx else 0)
                + self.sub_channels)

    @property
    def sub_channels(self):
        """The last input channels: the sub-frame envelope (one value per block of bins, not per bin)."""
        return self.audio_channels * self.sub

    @property
    def sub_block(self):
        """Bins per block of the sub-frame envelope: as many as it takes for a time slot of hop / sub samples."""
        return self.n_fft * self.sub // self.hop

    def band_in(self, stack):
        """bs: input channels of a band whose rows hold `stack` bins: the per-bin channels of all its bins, plus
        the sub-frame envelope once."""
        return (self.in_channels - self.sub_channels) * stack + self.sub_channels

    @property
    def out_channels(self):
        if self.taps == 0:
            return self.sources * self.audio_channels
        return (self.sources - 1) * self.audio_channels * 2 * self.taps

    def check(self):
        assert self.arch in ("unet", "bs"), self.arch
        if self.arch == "bs":
            down = 2 ** (len(self.widths) - 1)
            assert sum(b for b, _ in self.bands) == self.bins, "the bands must add up to `bins`"
            assert all(b % s == 0 and (b // s) % down == 0 for b, s in self.bands), \
                f"every band's row count must be divisible by {down}"
            assert self.frames % self.time_multiple == 0, f"frames must be divisible by {self.time_multiple}"
            assert len(self.tdf) == len(self.widths) and 0 <= self.tsteps < len(self.widths)
            if self.sub:
                g = self.sub_block
                assert self.hop % self.sub == 0 and self.bins % g == 0, "sub must divide the hop, and its blocks the bins"
                assert all(b % g == 0 and g % s == 0 for b, s in self.bands), f"every band must hold whole blocks of {g} bins"
        else:
            assert self.sub == 0, "the sub-frame envelope is a Needle 2 input"
            assert self.bins % 2 ** self.levels == 0 and self.frames % 2 ** self.levels == 0, \
                f"bins and frames must be divisible by {2 ** self.levels}"
        assert self.bins <= self.n_fft // 2
        assert self.taps == 0 or self.taps % 2 == 1, "taps must be 0 (masks) or odd (a centre frame and equal sides)"

    def tag(self):
        """Short name of the non-default features, for folder and file names ('' for the plain model)."""
        return (("_c" if self.coord else "") + (("_h" + "".join(map(str, self.harmonics))) if self.harmonics else "")
                + (f"_t{self.taps}" if self.taps else "") + ("_x" if self.cplx else "") + (f"_s{self.sub}" if self.sub else ""))


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
    def __new__(cls, cfg: NeedleConfig = None, *args, **kwargs):
        # Needle(cfg) builds the model the config asks for, so everything that says Needle(cfg) (training,
        # evaluation, export, the scans) also works for Needle 2 without knowing about it.
        if cls is Needle and cfg is not None and cfg.arch == "bs":
            from needle2 import Needle2
            return Needle2(cfg)                    # not a Needle instance, so Python skips Needle.__init__
        return super().__new__(cls)

    def __init__(self, cfg: NeedleConfig = NeedleConfig()):
        super().__init__()
        cfg.check()
        self.cfg = cfg
        chans = [cfg.base * m for m in cfg.mult]
        self.enc = nn.ModuleList()
        cin = cfg.in_channels
        for c in chans:
            self.enc.append(Block(cin, c, 2))
            cin = c
        self.dec = nn.ModuleList()
        for i in reversed(range(len(chans))):          # upsample, concat the skip, refine
            skip = chans[i - 1] if i > 0 else cfg.in_channels
            out = chans[i - 1] if i > 0 else cfg.base
            self.dec.append(Block(cin + skip, out, 1))
            cin = out
        self.head = nn.Conv2d(cin, cfg.out_channels, 1)
        if cfg.taps:
            # Deep filtering starts as a plain mask: the centre tap's real part is 0.25 for the
            # three estimated stems (tanh(0.2554) = 0.25), everything else 0. Without this the
            # first steps all go into finding the "identity" filter.
            with torch.no_grad():
                self.head.bias.zero_()
                c = cfg.audio_channels
                for s in range((cfg.sources - 1) * c):
                    self.head.bias[(s * cfg.taps + cfg.taps // 2) * 2] = 0.2554

    def forward(self, x):
        skips = [x]
        for b in self.enc:
            x = b(x)
            skips.append(x)
        skips.pop()
        for b in self.dec:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
            x = b(torch.cat([x, skips.pop()], 1))
        x = self.head(x)
        return torch.sigmoid(x) if self.cfg.taps == 0 else torch.tanh(x)


# ── Front and back end (on the device these run on the Cortex-M85) ───────────

def spectrogram(wav, cfg: NeedleConfig):
    """wav (B, C, N) -> complex STFT (B, C, n_fft//2+1, T)."""
    b, c, n = wav.shape
    spec = torch.stft(wav.reshape(b * c, n), cfg.n_fft, cfg.hop, window=torch.hann_window(cfg.n_fft, device=wav.device),
                      return_complex=True, center=True)
    return spec.reshape(b, c, spec.shape[-2], spec.shape[-1])


def network_input(spec, cfg: NeedleConfig):
    """Complex STFT (B, C, F, T) -> what the network sees (B, in_channels, bins, T):
    log1p magnitudes, then optionally
    - a coordinate channel, bin index / bins: a convolution is blind to where in the spectrum it
      is, this tells it (bass lives at the bottom, cymbals at the top);
    - harmonic channels: the mono log-magnitude at bin h*k for each h in cfg.harmonics, so a
      3x3 kernel sees a note's fundamental and its harmonics in one place (as pitch trackers
      do). Bins beyond the spectrum take the last bin."""
    mag = spec.abs()
    chans = [torch.log1p(mag[:, :, : cfg.bins])]
    b, c, f, t = mag.shape
    if cfg.coord:
        k = torch.arange(cfg.bins, device=spec.device, dtype=torch.float32) / cfg.bins
        chans.append(k[None, None, :, None].expand(b, 1, cfg.bins, t))
    if cfg.harmonics:
        mono = torch.log1p(mag.mean(1))                                  # (B, F, T)
        k = torch.arange(cfg.bins, device=spec.device)
        for h in cfg.harmonics:
            chans.append(mono[:, (h * k).clamp(max=f - 1)][:, None])
    if cfg.cplx:
        # The complex STFT with its magnitude log-compressed and its phase kept: real and imaginary part of
        # exp(j phase) * log1p|X| per audio channel, [L re, L im, R re, R im]. With only magnitudes the network
        # cannot know how neighbouring frames relate in phase, which a deep filter needs. Same range as the
        # log-magnitudes on purpose: in int8 all channels of the input tensor share one scale.
        m = mag[:, :, : cfg.bins]
        z = spec[:, :, : cfg.bins] * (torch.log1p(m) / m.clamp(min=1e-8))
        chans.append(torch.view_as_real(z).permute(0, 1, 4, 2, 3).reshape(b, 2 * c, cfg.bins, t))
    if cfg.sub:
        chans.append(subframe_envelope(spec, cfg))                          # must stay last (NeedleConfig.sub_channels)
    return torch.cat(chans, 1)


def subframe_envelope(spec, cfg: NeedleConfig):
    """Complex STFT (B, C, F, T) -> (B, C * sub, bins, T): log1p of the mixture's envelope in `sub` time slots
    inside each frame's hop, for every block of g = n_fft * sub / hop neighbouring bins (16 bins = 172 Hz for
    4096 / 1024 and 4 slots of 5.8 ms). All bins of a block get the same values.

    Why: a 4096-sample window is 93 ms long. A hi-hat lasts 20 ms; in the magnitudes of such a frame it is a small
    bump on top of whatever else is bright, and Needle 2 lost hats that Needle 1 (46-ms windows) kept. The
    information is still in the frame: g neighbouring bins, added up with the phase turned to a time tau inside the
    window, give the band's signal around tau (an inverse DFT over g bins; the time resolution is n_fft / g
    samples). So this is a short-window view computed from the long-window STFT, with no second STFT: on the
    deck it is `sub` dot products of g complex numbers per block and frame, on the M85.

    The slots tile the hop in the middle of the window (where the Hann window is above 0.9), so the slots of
    successive frames follow each other without gaps."""
    g, s = cfg.sub_block, cfg.sub
    b, c, _, t = spec.shape
    z = spec[:, :, : cfg.bins].reshape(b, c, cfg.bins // g, g, t)
    tau = cfg.n_fft / 2 - cfg.hop / 2 + (torch.arange(s, device=spec.device) + 0.5) * (cfg.hop / s)   # slot centres, samples from the window start
    angle = 2 * math.pi * tau[:, None] * torch.arange(g, device=spec.device)[None] / cfg.n_fft        # (s, g)
    turn = torch.polar(torch.ones_like(angle), angle).to(z.dtype)
    with torch.autocast(spec.device.type, enabled=False):                 # complex sums stay in float32
        y = torch.einsum("bcngt,sg->bcsnt", z, turn)                      # (B, C, s, blocks, T)
        env = torch.log1p(y.abs())
    return env.repeat_interleave(g, dim=3).reshape(b, c * s, cfg.bins, t)


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


def apply_filters(out, spec, cfg: NeedleConfig):
    """Deep filtering. out (B, K, bins, T) in -1..1 with K = (sources-1) * C * 2 * taps, order
    [drums L tap0 re, im, tap1 re, im, ..., drums R ..., bass ..., vocals ...] (source-major,
    "other" left out). spec: the mixture STFT (B, C, F, T + taps - 1), i.e. the chunk's frames
    plus taps//2 frames of context on each side (zeros at the track ends).
    Each stem is a complex FIR over `taps` neighbouring frames of the mixture, per bin: like a
    mask, but it can also rotate phase and use the frames around (DeepFilterNet's idea).
    "other" = mixture - the three estimated stems, so the four stems add up to the mix exactly.
    Returns (B, S, C, F, T) complex in SOURCES order."""
    b, c, f, t_ext = spec.shape
    taps = cfg.taps
    t = t_ext - taps + 1
    h = out.reshape(b, cfg.sources - 1, c, taps, 2, cfg.bins, t)
    h = torch.complex(h[:, :, :, :, 0], h[:, :, :, :, 1])            # (B, S-1, C, taps, bins, T)
    x = spec[:, :, : cfg.bins]
    y = torch.zeros(b, cfg.sources - 1, c, cfg.bins, t, dtype=spec.dtype, device=spec.device)
    for j in range(taps):
        y = y + h[:, :, :, j] * x[:, :, :, j:j + t].unsqueeze(1)
    mid = spec[..., taps // 2: taps // 2 + t]                        # the chunk's own frames
    stems = []
    other = SOURCES.index("other")
    k = 0
    for i in range(cfg.sources):
        if i == other:
            stems.append(None)
            continue
        full = torch.zeros(b, c, f, t, dtype=spec.dtype, device=spec.device)
        full[:, :, : cfg.bins] = y[:, k]
        stems.append(full)
        k += 1
    stems[other] = mid - sum(s for s in stems if s is not None)
    return torch.stack(stems, 1)


def pad_time(spec, cfg: NeedleConfig):
    """Zero frames on both sides so apply_filters can look taps//2 frames around each frame."""
    p = cfg.taps // 2
    return F.pad(spec, (p, p)) if p else spec


def apply_output(out, spec, cfg: NeedleConfig):
    """Network output + the chunk's mixture STFT (B, C, F, T) -> per-source STFT (B, S, C, F, T),
    for either output type."""
    if cfg.taps == 0:
        return apply_masks(out, spec, cfg)
    return apply_filters(out, pad_time(spec, cfg), cfg)


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
    pad = (-t) % cfg.time_multiple
    x = F.pad(network_input(spec, cfg), (0, pad))
    out = model(x)[..., :t]
    return istft(apply_output(out, spec, cfg), cfg, wav.shape[-1])


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
        if isinstance(mod, nn.Conv2d):
            total += outp.numel() * mod.kernel_size[0] * mod.kernel_size[1] * mod.in_channels // mod.groups
        elif isinstance(mod, nn.Linear):
            total += outp.numel() * mod.in_features
        else:                                       # GRU: three gates, each fed by the input and the state, per step and direction
            y = outp[0]
            total += y.shape[0] * y.shape[1] * (2 if mod.bidirectional else 1) * 3 * mod.hidden_size * (mod.input_size + mod.hidden_size)
    hs = [mod.register_forward_hook(hook) for mod in m.modules() if isinstance(mod, (nn.Conv2d, nn.Linear, nn.GRU))]
    with torch.no_grad():
        m(torch.zeros(1, cfg.in_channels, cfg.bins, cfg.frames))
    for h in hs:
        h.remove()
    chunk_s = cfg.frames * cfg.hop / cfg.sample_rate
    params = sum(p.numel() for p in m.parameters())
    return {"params": params, "macs_per_chunk": total, "gmac_per_audio_s": total / chunk_s / 1e9, "chunk_s": chunk_s}


def selftest():
    """Shapes, and that the four stems add back up to the mixture, for masks and for deep
    filtering with the extra input channels."""
    torch.manual_seed(0)
    for cfg in (NeedleConfig(base=8), NeedleConfig(base=8, coord=True, harmonics=(2, 3, 4), taps=5)):
        model = Needle(cfg).eval()
        wav = torch.randn(1, 2, cfg.sample_rate * 3) * 0.1
        with torch.no_grad():
            stems = separate(model, wav)
        assert stems.shape == (1, 4, 2, wav.shape[-1]), stems.shape
        err = (stems.sum(1) - wav).abs().max().item()
        assert err < 1e-3, f"stems don't sum to the mix: {err}"
        print(f"selftest ok (taps {cfg.taps}, in {cfg.in_channels}, out {cfg.out_channels}): stems",
              tuple(stems.shape), "sum-to-mix error", f"{err:.2e}")


if __name__ == "__main__":
    selftest()
    for base in (8, 12, 16, 24):
        cfg = NeedleConfig(base=base)
        print(f"base={base}", count_macs(cfg))
