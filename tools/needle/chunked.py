"""Device-style chunked inference, shared by train.py (validation), eval.py and eval_tflite.py.

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

from needle import NeedleConfig, apply_masks, istft, network_input, spectrogram


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
def separate_chunked(predict, wav: torch.Tensor, cfg: NeedleConfig, overlap: int = 8, batch: int = 64) -> torch.Tensor:
    """Separate a whole track the way the device would. wav (C, N) -> stems (S, C, N).
    Front end (STFT, log1p) and back end (mask normalisation, iSTFT) are needle.py's own
    functions, so this matches needle.separate() except for the chunking."""
    spec = spectrogram(wav[None], cfg)                       # (1, C, F, T) complex
    x = network_input(spec, cfg)[0]                          # (C, bins, T)
    masks = chunked_masks(predict, x, cfg.frames, overlap, batch)
    return istft(apply_masks(masks[None], spec, cfg), cfg, wav.shape[-1])[0]


def torch_predictor(model):
    """Wrap a PyTorch Needle as a `predict` function (float, in eval mode)."""
    model.eval()
    return lambda xb: model(xb)


def nsdr(ref: torch.Tensor, est: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Fast SDR per source over a whole track, as in the Music Demixing Challenge:
    10 log10(sum s^2 / sum (s - s_hat)^2). ref, est (S, C, N) -> (S,).
    It is a quick stand-in for museval's SDR (which is ~100x slower) during training.
    The two usually agree to within a fraction of a dB but are not identical; the numbers we
    report come from museval (eval.py)."""
    num = ref.double().pow(2).sum((-2, -1))
    den = (ref.double() - est.double()).pow(2).sum((-2, -1))
    return 10 * torch.log10((num + eps) / (den + eps))
