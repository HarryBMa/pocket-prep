"""Checks for the optional features (coord / harmonics inputs, deep filtering) that need no training.

    .venv\\Scripts\\python test_frontier.py

1. chunked_pieces gives exactly what chunked_masks gives (it is the same crossfade, streamed).
2. Deep filtering through the chunked path equals the whole-track path when the network output
   does not depend on position (constant filters): the context frames and the chunk borders line up.
3. Filters that pick "centre tap = 1" for one stem give that stem = the mix and the rest 0.
4. The training loss runs, is finite and backpropagates, for masks and for deep filtering.
5. The training crop's STFT/iSTFT alignment holds with the extra context frames.
"""
import torch

from chunked import chunked_masks, chunked_pieces, separate_chunked
from needle import Needle, NeedleConfig, apply_output, istft, network_input, spectrogram
from train import check_alignment, compute_loss


def test_pieces():
    torch.manual_seed(0)
    x = torch.randn(2, 64, 300)
    predict = lambda xb: torch.cat([xb.cos() * 0.3, xb.sin()], 1)           # (N, 4, 64, frames), per-frame function
    whole = chunked_masks(predict, x, 32, 8, batch=5)
    parts = list(chunked_pieces(predict, x, 32, 8, batch=5))
    cat = torch.cat([p for _, p in parts], -1)[..., :300]
    assert [b for b, _ in parts][0] == 0 and torch.allclose(cat, whole, atol=1e-6), (cat - whole).abs().max()
    print("pieces == masks: ok", len(parts), "pieces")


def constant_predict(cfg, out_vec):
    """A fake network whose output is the same at every bin and frame: out_vec (K,)."""
    def predict(xb):
        return out_vec.view(1, -1, 1, 1).expand(xb.shape[0], -1, cfg.bins, cfg.frames).clone()
    return predict


def test_filters_chunked_equals_whole():
    torch.manual_seed(1)
    cfg = NeedleConfig(base=8, frames=32, taps=5)
    wav = torch.randn(2, 44100 * 4) * 0.1
    out_vec = torch.rand(cfg.out_channels) * 0.6 - 0.3
    chunked = separate_chunked(constant_predict(cfg, out_vec), wav, cfg, overlap=8, batch=7)
    spec = spectrogram(wav[None], cfg)
    t = spec.shape[-1]
    out = out_vec.view(1, -1, 1, 1).expand(1, -1, cfg.bins, t)
    whole = istft(apply_output(out, spec, cfg), cfg, wav.shape[-1])[0]
    err = (chunked - whole).abs().max().item()
    assert err < 1e-4, err
    print(f"deep filtering chunked == whole track: ok (max error {err:.1e})")
    assert (chunked.sum(0) - wav).abs().max() < 1e-4


def test_centre_tap():
    cfg = NeedleConfig(base=8, frames=32, taps=3)
    wav = torch.randn(2, 44100 * 3) * 0.1
    out = torch.zeros(cfg.out_channels)
    # layout (source, channel, tap, re/im): source 0 = drums, both channels, centre tap (index 1), real part
    for ch in range(cfg.audio_channels):
        out[((0 * cfg.audio_channels + ch) * cfg.taps + 1) * 2] = 1.0
    stems = separate_chunked(constant_predict(cfg, out), wav, cfg, overlap=8)
    # the Nyquist bin (1 of 1025) is never filtered and goes to "other", as with masks: allow that much
    rms = lambda x: x.pow(2).mean().sqrt()
    assert rms(stems[0] - wav) < 0.05 * rms(wav) and rms(stems[1]) < 1e-4 and rms(stems[3]) < 1e-4 and rms(stems[2]) < 0.05 * rms(wav), \
        [rms(x).item() for x in (stems[0] - wav, *stems[1:])]
    print("centre tap = 1 gives drums = mix, others = 0: ok")


def test_loss():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    for cfg in (NeedleConfig(base=8, frames=32), NeedleConfig(base=8, frames=32, coord=True, harmonics=(2, 3), taps=5)):
        check_alignment(cfg, dev)
        model = Needle(cfg).to(dev)
        from train import crop_length
        stems = torch.randn(3, 4, 2, crop_length(cfg), device=dev) * 0.05
        spec_norm = torch.hann_window(cfg.n_fft).pow(2).sum().sqrt().item()
        loss, spec_l1, wave_l1 = compute_loss(model, stems, cfg, 0.1, torch.bfloat16 if dev == "cuda" else None, spec_norm, 0.5)
        loss.backward()
        assert torch.isfinite(loss), loss
        g = sum(p.grad.abs().sum().item() for p in model.parameters() if p.grad is not None)
        assert g > 0
        print(f"loss ok (taps {cfg.taps}, in {cfg.in_channels}): {loss.item():.4f} spec {spec_l1.item():.4f} wave {wave_l1.item():.4f}")


if __name__ == "__main__":
    test_pieces()
    test_filters_chunked_equals_whole()
    test_centre_tap()
    test_loss()
    print("all ok")
