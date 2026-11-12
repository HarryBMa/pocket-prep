r"""Device-style chunked inference, shared by train.py (validation), eval.py and eval_tflite.py.

On the deck the Cortex-M85 runs the STFT continuously and the NPU sees one chunk of `frames`
STFT frames at a time. Neighbouring chunks overlap by `overlap` frames, and inside the overlap
the two chunks' masks are crossfaded linearly, so there is no seam at chunk borders:

    frames = 32, overlap = 8  ->  a new chunk every 24 frames (0.28 s)

    chunk 0   [0 ......................... 31]
    chunk 1                        [24 ......................... 55]
    weight     1 1 1 ... 1 1 1 \ \ \          / / / 1 1 1 ... 1 \ \ \
                              crossfade 24..31

Every chunk is run through the same `predict` function, which is either the PyTorch model
(float) or a TFLite interpreter (int8 / int16), so float and quantised models are evaluated
in exactly the same way.
"""
import torch
import torch.nn.functional as F

from needle import SOURCES, NeedleConfig, apply_filters, istft, network_input, pad_time, spectrogram


def crossfade_weights(frames: int, overlap: int) -> torch.Tensor:
    """Per-frame weight of one chunk: linear ramps over `overlap` frames at both ends.
    A ramp-down and the next chunk's ramp-up add up to exactly 1 in every frame."""
    w = torch.ones(frames)
    if overlap:
        ramp = (torch.arange(overlap) + 0.5) / overlap      # 1/16, 3/16, ... 15/16 for overlap 8
        w[:overlap] = ramp
        w[-overlap:] = ramp.flip(0)
    return w


def chunk_starts(t: int, frames: int, overlap: int):
    """First frame of every chunk needed to cover t frames, and the padded length."""
    step = frames - overlap
    n = max(1, -(-(t - overlap) // step))                   # ceil((t - overlap) / step)
    return [i * step for i in range(n)], (n - 1) * step + frames


@torch.no_grad()
def chunked_masks(predict, x: torch.Tensor, frames: int, overlap: int = 8, batch: int = 64) -> torch.Tensor:
    """x: network input of a whole track, (C, bins, T).
    predict: (N, C, bins, frames) -> (N, K, bins, frames) masks (K = sources * channels).
    Returns the crossfaded masks for the whole track, (K, bins, T).

    The end of the track is padded with silence (log1p(0) = 0) to fill the last chunk. At the
    very start and end there is no neighbour, so we divide by the summed weights instead of
    assuming they add up to 1."""
    c, bins, t = x.shape
    starts, t_pad = chunk_starts(t, frames, overlap)
    x = F.pad(x, (0, t_pad - t))
    w = crossfade_weights(frames, overlap).to(x.device)
    acc, wsum = None, torch.zeros(t_pad, device=x.device)
    for i in range(0, len(starts), batch):
        group = starts[i:i + batch]
        masks = predict(torch.stack([x[..., a:a + frames] for a in group]))
        if acc is None:
            acc = torch.zeros(masks.shape[1], bins, t_pad, device=x.device)
        for m, a in zip(masks, group):
            acc[..., a:a + frames] += m.float() * w
            wsum[a:a + frames] += w
    return (acc / wsum)[..., :t]


@torch.no_grad()
def chunked_pieces(predict, x: torch.Tensor, frames: int, overlap: int = 8, batch: int = 64):
    """Like chunked_masks, but yields (first_frame, outputs (K, bins, n)) pieces in order instead
    of one array for the whole track: a frame is final as soon as the next chunk (the last one
    that can cover it) has been added. Needed when K is large (deep filtering: 60 channels would
    be 8 GB for a 6-minute song). The pieces cover the padded track; the caller clips to T."""
    c, bins, t = x.shape
    starts, t_pad = chunk_starts(t, frames, overlap)
    x = F.pad(x, (0, t_pad - t))
    w = crossfade_weights(frames, overlap).to(x.device)
    step = frames - overlap
    buf = wbuf = None
    base = 0
    for i in range(0, len(starts), batch):
        group = starts[i:i + batch]
        out = predict(torch.stack([x[..., a:a + frames] for a in group]))
        need = group[-1] + frames - base
        if buf is None:
            buf = torch.zeros(out.shape[1], bins, need, device=x.device)
            wbuf = torch.zeros(need, device=x.device)
        else:
            buf = F.pad(buf, (0, need - buf.shape[-1]))
            wbuf = F.pad(wbuf, (0, need - wbuf.shape[-1]))
        for m, a in zip(out, group):
            buf[..., a - base:a - base + frames] += m.float().to(x.device) * w
            wbuf[a - base:a - base + frames] += w
        cut = t_pad if i + batch >= len(starts) else group[-1] + step      # frames below `cut` are final
        n = cut - base
        yield base, buf[..., :n] / wbuf[:n]
        buf, wbuf, base = buf[..., n:], wbuf[n:], cut


@torch.no_grad()
def separate_filtered(predict, wav: torch.Tensor, cfg: NeedleConfig, overlap: int = 8, batch: int = 64) -> torch.Tensor:
    """separate_once() for deep-filtering models (cfg.taps >= 1). The network's outputs (the
    filter coefficients) are crossfaded between chunks exactly like masks are; each filter is then
    applied to the mixture STFT, which has the neighbouring frames the filter needs (taps//2 on
    each side), so chunk borders cost nothing. "other" is the residual (apply_filters)."""
    spec = spectrogram(wav[None], cfg)[0]                    # (C, F, T) complex
    c, f, t = spec.shape
    ext = pad_time(spec[None], cfg)[0]                       # (C, F, T + taps - 1)
    stems = torch.zeros(cfg.sources, c, f, t, dtype=spec.dtype)            # on the CPU: a full song does not fit the GPU next to training
    net_in = network_input(spec[None], cfg)[0].cpu()        # chunks and filter outputs stay on the CPU; torch_predictor moves each batch to the GPU and back
    for base, piece in chunked_pieces(predict, net_in, cfg.frames, overlap, batch):
        n = min(piece.shape[-1], t - base)
        if n <= 0:
            break
        for a in range(0, n, 128):                           # 128 frames at a time on the GPU
            e = min(n, a + 128)
            stems[..., base + a:base + e] = apply_filters(piece[None, ..., a:e].to(ext.device), ext[None, ..., base + a:base + e + cfg.taps - 1], cfg)[0].cpu()
    return torch.stack([istft(stems[s][None, None], cfg, wav.shape[-1])[0, 0] for s in range(cfg.sources)])


@torch.no_grad()
def separate_chunked(predict, wav: torch.Tensor, cfg: NeedleConfig, overlap: int = 8, batch: int = 64,
                     shifts: int = 1) -> torch.Tensor:
    """separate_once(), optionally `shifts` times with the audio delayed a little each time, and
    the stems averaged (the "shift trick" Demucs uses). Pass k is delayed by k/shifts of a hop plus
    k/shifts of a chunk step, so every pass sees different STFT frames and different chunk
    borders; their errors partly cancel. Costs `shifts` times the compute, no extra memory or
    weights: useful when only a selected section is separated and there is time to spare."""
    if shifts <= 1:
        return separate_once(predict, wav, cfg, overlap, batch)
    step = (cfg.frames - overlap) * cfg.hop
    out = None
    for k in range(shifts):
        d = k * cfg.hop // shifts + k * step // shifts
        stems = separate_once(predict, F.pad(wav, (d, 0)), cfg, overlap, batch)[..., d:d + wav.shape[-1]]
        out = stems if out is None else out + stems
    return out / shifts


@torch.no_grad()
def separate_once(predict, wav: torch.Tensor, cfg: NeedleConfig, overlap: int = 8, batch: int = 64) -> torch.Tensor:
    """Separate a whole track the way the device would. wav (C, N) -> stems (S, C, N), on the CPU.

    Front end (STFT, log1p) and back end (masks normalised to sum to 1, iSTFT) are the same as
    needle.separate() / needle.apply_masks(). The back end runs one source at a time, which gives
    the same result with a quarter of the memory: a 6-minute song as one tensor of four complex
    stems is several GB, and on Windows a GPU that runs out of memory doesn't fail, it silently
    spills into system RAM and becomes ~100x slower."""
    if cfg.taps:
        return separate_filtered(predict, wav, cfg, overlap, batch)
    spec = spectrogram(wav[None], cfg)[0]                    # (C, F, T) complex
    masks = chunked_masks(predict, network_input(spec[None], cfg)[0], cfg.frames, overlap, batch)
    c, f, t = spec.shape
    m = masks.reshape(cfg.sources, c, cfg.bins, t)
    m /= m.sum(0, keepdim=True) + 1e-4                       # same normalisation (and eps) as apply_masks
    other = SOURCES.index("other")
    stems = []
    for s in range(cfg.sources):
        full = torch.zeros(c, f, t, device=spec.device)
        full[:, :cfg.bins] = m[s]
        if s == other:
            full[:, cfg.bins:] = 1.0                         # bins above `bins` (the Nyquist bin) go to "other"
        stems.append(istft((full * spec)[None, None], cfg, wav.shape[-1])[0, 0].cpu())
        del full
    return torch.stack(stems)


def torch_predictor(model, batch: int = 64):
    """Wrap a PyTorch Needle as a `predict` function (float, in eval mode). The last group of
    chunks of a track is padded to the full batch, so the GPU only ever sees one input shape.
    Chunks are moved to the model's device and the masks back to the chunks' device, so a track
    can be handled on the CPU (STFT, masks, iSTFT: several GB for a long song) while only the
    network runs on the GPU.
    A model that sets `eval_batch` (Needle 2, whose chunks and outputs are several times larger) is run
    in slices of that many chunks, each moved back before the next one runs, so a whole batch of
    outputs never sits on the GPU at once."""
    model.eval()
    device = next(model.parameters()).device
    step = min(batch, getattr(model, "eval_batch", batch))

    def predict(xb):
        n, home = xb.shape[0], xb.device
        if step == batch:
            if n < batch:
                xb = F.pad(xb, (0, 0, 0, 0, 0, 0, 0, batch - n))
            return model(xb.to(device))[:n].to(home)
        m = -(-n // step) * step                             # the last slice is padded to a full one
        if n < m:
            xb = F.pad(xb, (0, 0, 0, 0, 0, 0, 0, m - n))
        return torch.cat([model(xb[i:i + step].to(device)).to(home) for i in range(0, m, step)])[:n]
    return predict


def nsdr(ref: torch.Tensor, est: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Fast SDR per source over a whole track, as in the Music Demixing Challenge:
    10 log10(sum s^2 / sum (s - s_hat)^2). ref, est (S, C, N) -> (S,).
    It is a quick stand-in for museval's SDR (which is ~100x slower) during training.
    The two usually agree to within a fraction of a dB but are not identical; the numbers we
    report come from museval (eval.py)."""
    num = ref.double().pow(2).sum((-2, -1))
    den = (ref.double() - est.double()).pow(2).sum((-2, -1))
    return 10 * torch.log10((num + eps) / (den + eps))
