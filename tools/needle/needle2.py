"""Needle 2 (NeedleConfig.arch = "bs"): the band-split stem separator for the Ethos-U55.

Same job and same interface as Needle 1 (needle.py): forward(network_input) -> (B, K, bins, frames) masks or
deep-filter taps, so training, chunked inference and evaluation are shared. What is different, and why
(the research blueprint of 12 Nov 2026 and vela_probe.py; every piece below was compiled with Vela 5.2.0 for
the RA8P1 and runs on the NPU with 0 CPU operators):

- A longer STFT (n_fft 4096, hop 1024, set in the config). Bins are 10.8 Hz wide instead of 21.5 Hz, so kick
  and bass can be told apart, and a 128-frame chunk holds 3 s of music instead of 1.5 s.
- The band grid. The network does not work on 2048 bins but on 512 rows: one bin per row up to 1.4 kHz,
  then 2, 4 and 16 bins per row (NeedleConfig.bands). The bins of a row are stacked into channels, which
  loses nothing; on the deck the Cortex-M85 fills the band tensors directly. Feature maps need half the memory
  of a 1024-bin grid at the same width, and the low end has twice the resolution.
- The deep levels halve frequency only. After two 2x2 steps time stays at frames/4, so the deepest level
  still has 32 time steps to work with (Needle 1 was left with 4).
- TDF blocks (from the TFC-TDF / KUIELab separators): a small fully connected layer along frequency, the
  same for every channel and frame, added to its input. A 3x3 convolution only sees neighbouring rows; a TDF
  sees the whole spectrum at once (a note's harmonics, how the bands relate).
- Dual-path GRU blocks at the deepest level (as in BSRNN, SCNet and DTTNet): a bidirectional GRU along time
  for every row, then one along frequency for every frame. On the NPU a GRU is written out step by step
  (1x1 convolutions, LOGISTIC, TANH, MUL, ADD). That costs about 7 ms per direction for 32 steps, against
  62 ms for one attention block, which is why this is a GRU and not a transformer.
- A light decoder (one convolution per level) and an added, not concatenated, skip at level 0: the
  full-resolution concat was the largest thing in the NPU's working memory.
- The output stays on the band grid: one filter per row, source and channel. forward() repeats it for the
  bins of its row; on the deck the M85 does that when it applies the filters.

At inference the layers are CONV_2D, RELU, CONCATENATION, RESHAPE, TRANSPOSE, ADD, SUB, MUL, LOGISTIC, TANH,
SPLIT and 2x2 RESIZE_NEAREST_NEIGHBOR. BatchNorm is folded into the convolutions on export (needle2_tf.py).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from needle import NeedleConfig


def bs_config(**kw):
    """A Needle 2 config: the band-split defaults (4096 / 1024 STFT, 2048 bins, 128 frames = 3 s, complex
    input, deep filtering over 3 frames) with anything in `kw` on top."""
    return NeedleConfig(**{**dict(arch="bs", n_fft=4096, hop=1024, bins=2048, frames=128, cplx=True, taps=3), **kw})


# ── The band grid ────────────────────────────────────────────────────────────

def to_bands(x, cfg: NeedleConfig):
    """Network input (B, C, bins, T) -> one tensor per band, (B, C * stack, rows, T): the `stack` bins of a row
    become channels (channel index = c * stack + position in the row). Nothing is lost; it is only a different
    layout, and it is the layout the M85 fills in on the deck."""
    b, c, _, t = x.shape
    out, k = [], 0
    for bins, s in cfg.bands:
        band = x[:, :, k:k + bins].reshape(b, c, bins // s, s, t).permute(0, 1, 3, 2, 4)
        out.append(band.reshape(b, c * s, bins // s, t))
        k += bins
    return out


def row_of_bin(cfg: NeedleConfig):
    """For every bin, the row of the band grid it belongs to (a LongTensor of length `bins`)."""
    idx, r = [], 0
    for bins, s in cfg.bands:
        idx.append(r + torch.arange(bins) // s)
        r += bins // s
    return torch.cat(idx)


# ── Blocks ───────────────────────────────────────────────────────────────────

class ConvBN(nn.Sequential):
    """3x3 conv + BatchNorm + ReLU. The stride is (frequency, time)."""

    def __init__(self, cin, cout, stride=(1, 1)):
        super().__init__(nn.Conv2d(cin, cout, 3, stride, 1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class TDF(nn.Module):
    """x + FC(relu(FC(x))) along frequency: rows -> rows / factor -> rows, the same weights for every channel
    and frame. The BatchNorms act on the FC outputs, so they fold into the FC weights on export. The second one
    starts at 0: a new TDF starts as "do nothing"."""

    def __init__(self, rows, factor):
        super().__init__()
        assert rows % factor == 0, (rows, factor)
        self.fc1, self.bn1 = nn.Linear(rows, rows // factor, bias=False), nn.BatchNorm1d(rows // factor)
        self.fc2, self.bn2 = nn.Linear(rows // factor, rows, bias=False), nn.BatchNorm1d(rows)
        nn.init.zeros_(self.bn2.weight)

    def forward(self, x):                                   # (B, C, F, T)
        b, c, f, t = x.shape
        y = x.permute(0, 1, 3, 2).reshape(b * c * t, f)     # frequency last: one row of numbers per (channel, frame)
        y = self.bn2(self.fc2(F.relu(self.bn1(self.fc1(y)))))
        return x + y.reshape(b, c, t, f).permute(0, 1, 3, 2)


class DualPath(nn.Module):
    """A bidirectional GRU along time for every row, then one along frequency for every frame. Each path is
    BatchNorm -> GRU -> 1x1 conv -> BatchNorm, added to its input; the last BatchNorm starts at 0."""

    def __init__(self, c, hidden):
        super().__init__()
        self.bn_t, self.bn_f = nn.BatchNorm2d(c), nn.BatchNorm2d(c)
        self.gru_t = nn.GRU(c, hidden, batch_first=True, bidirectional=True)
        self.gru_f = nn.GRU(c, hidden, batch_first=True, bidirectional=True)
        self.out_t, self.out_f = nn.Conv2d(2 * hidden, c, 1, bias=False), nn.Conv2d(2 * hidden, c, 1, bias=False)
        self.bno_t, self.bno_f = nn.BatchNorm2d(c), nn.BatchNorm2d(c)
        nn.init.zeros_(self.bno_t.weight)
        nn.init.zeros_(self.bno_f.weight)

    @staticmethod
    def run(gru, seq):
        # cuDNN's GRU has no bfloat16 path, so it runs in float32 whatever autocast says (a tiny part of the compute).
        with torch.autocast(seq.device.type, enabled=False):
            return gru(seq.float())[0]

    def forward(self, x):                                   # (B, C, F, T)
        b, c, f, t = x.shape
        y = self.run(self.gru_t, self.bn_t(x).permute(0, 2, 3, 1).reshape(b * f, t, c))        # sequences along time
        x = x + self.bno_t(self.out_t(y.reshape(b, f, t, -1).permute(0, 3, 1, 2)))
        y = self.run(self.gru_f, self.bn_f(x).permute(0, 3, 2, 1).reshape(b * t, f, c))        # sequences along frequency
        return x + self.bno_f(self.out_f(y.reshape(b, t, f, -1).permute(0, 3, 2, 1)))


# ── The network ──────────────────────────────────────────────────────────────

class Needle2(nn.Module):
    """Level 0 is the band grid (rows x frames, widths[0] channels); level i has widths[i] channels and
    rows / 2**i rows. strides[i] is the step from level i down to level i + 1."""

    # chunked.torch_predictor runs models that set this in slices of that many chunks. Needle 2's output for 64
    # three-second chunks is 2.3 GB; in slices a full-track evaluation fits on the GPU next to a training run.
    eval_batch = 8

    def __init__(self, cfg: NeedleConfig):
        super().__init__()
        cfg.check()
        assert cfg.arch == "bs"
        self.cfg = cfg
        w, n = cfg.widths, len(cfg.widths)
        self.strides = [(2, 2) if i < cfg.tsteps else (2, 1) for i in range(n - 1)]
        rows = [cfg.rows >> i for i in range(n)]

        def tdf(i):
            return [TDF(rows[i], cfg.tdf[i])] if cfg.tdf[i] else []

        self.stems = nn.ModuleList(ConvBN(cfg.in_channels * s, w[0]) for _, s in cfg.bands)
        self.enc = nn.ModuleList([nn.Sequential(ConvBN(w[0], w[0]), *tdf(0))])
        for i in range(1, n):
            self.enc.append(nn.Sequential(ConvBN(w[i - 1], w[i], self.strides[i - 1]), ConvBN(w[i], w[i]), *tdf(i)))
        self.rnn = nn.ModuleList(DualPath(w[-1], cfg.rnn_hidden) for _ in range(cfg.rnn))
        # Decoder stage for level i: upsample, join the skip, one conv. Level 1 already comes out with widths[0]
        # channels so that level 0 can ADD its skip (a concat there would be the largest tensor in the arena).
        self.dec = nn.ModuleList()
        for i in reversed(range(n - 1)):
            cin = w[0] if i == 0 else w[i + 1] + w[i]
            self.dec.append(nn.Sequential(ConvBN(cin, w[max(i - 1, 0)] if i <= 1 else w[i]), *tdf(i)))
        self.head = nn.Conv2d(w[0], cfg.out_channels, 1)
        if cfg.taps:
            # As in Needle 1: deep filtering starts as a plain mask of 0.25 on the centre tap (tanh(0.2554) = 0.25).
            with torch.no_grad():
                self.head.bias.zero_()
                for s in range((cfg.sources - 1) * cfg.audio_channels):
                    self.head.bias[(s * cfg.taps + cfg.taps // 2) * 2] = 0.2554
        self.register_buffer("row_of_bin", row_of_bin(cfg), persistent=False)

    def forward_rows(self, x):
        """Network input (B, C, bins, T) -> output on the band grid (B, K, rows, T). This is the part that
        runs on the NPU."""
        cfg = self.cfg
        x = torch.cat([stem(band) for stem, band in zip(self.stems, to_bands(x, cfg))], 2)
        skips = []
        for blk in self.enc:
            x = blk(x)
            skips.append(x)
        skips.pop()                                         # the deepest level is not a skip
        for blk in self.rnn:
            x = blk(x)
        for i, blk in zip(reversed(range(len(self.dec))), self.dec):
            x = F.interpolate(x, scale_factor=self.strides[i], mode="nearest")
            x = blk(x + skips.pop() if i == 0 else torch.cat([x, skips.pop()], 1))
        x = self.head(x)
        return torch.tanh(x) if cfg.taps else torch.sigmoid(x)

    def forward(self, x):
        return self.forward_rows(x)[:, :, self.row_of_bin]  # every bin takes its row's output


def randomize_norms(model, seed=0):
    """Give every BatchNorm random statistics and scales (tests and random-weight exports only). A fresh model
    has running mean 0 / variance 1 and its residual branches switched off (scale 0), so without this an export
    test would not notice a wrong BatchNorm fold, and quantisation would be calibrated on branches that output 0."""
    g = torch.Generator().manual_seed(seed)
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
            with torch.no_grad():
                m.weight.copy_(torch.rand(m.weight.shape, generator=g) * 0.5 + 0.75)
                m.bias.copy_(torch.randn(m.bias.shape, generator=g) * 0.1)
                m.running_mean.copy_(torch.randn(m.running_mean.shape, generator=g) * 0.1)
                m.running_var.copy_(torch.rand(m.running_var.shape, generator=g) * 0.5 + 0.75)
    return model


def selftest():
    from needle import count_macs, separate
    torch.manual_seed(0)
    small = dict(frames=32, widths=(8, 8, 16, 16, 16), rnn=1, rnn_hidden=8)
    for cfg in (bs_config(**small), bs_config(**small, taps=0, cplx=False)):
        model = randomize_norms(Needle2(cfg)).eval()
        x = torch.randn(2, cfg.in_channels, cfg.bins, cfg.frames)
        bands = to_bands(x, cfg)
        assert [tuple(b.shape[1:3]) for b in bands] == [(cfg.in_channels * s, n // s) for n, s in cfg.bands]
        assert torch.equal(bands[1][0, 1, 3], x[0, 0, 128 + 3 * 2 + 1])          # band 1, row 3, second bin of channel 0
        with torch.no_grad():
            rows, out = model.forward_rows(x), model(x)
        assert rows.shape == (2, cfg.out_channels, cfg.rows, cfg.frames) and out.shape == (2, cfg.out_channels, cfg.bins, cfg.frames)
        assert torch.equal(out[:, :, 127], rows[:, :, 127]) and torch.equal(out[:, :, 2047], rows[:, :, 511]) \
            and torch.equal(out[:, :, 128], out[:, :, 129])
        wav = torch.randn(1, 2, cfg.sample_rate * 2) * 0.1
        with torch.no_grad():
            stems = separate(model, wav)
        err = (stems.sum(1) - wav).abs().max().item()
        assert stems.shape == (1, 4, 2, wav.shape[-1]) and err < 1e-3, (stems.shape, err)
        print(f"selftest ok (taps {cfg.taps}, cplx {cfg.cplx}): rows {cfg.rows}, in {cfg.in_channels}, out {cfg.out_channels}, "
              f"stems sum to the mix within {err:.1e}")
    cfg = bs_config()
    info = count_macs(cfg)
    print(f"default Needle 2: {info['params']:,} parameters, {info['gmac_per_audio_s']:.2f} GMAC per second of audio, "
          f"{info['chunk_s']:.2f}-s chunks")


if __name__ == "__main__":
    selftest()
