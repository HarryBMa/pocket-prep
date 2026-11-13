r"""Checks for Needle 2 (needle2.py) that need no training.

    .venv\Scripts\python test_needle2.py                       # all checks (uses the GPU for the loss check if it is free)
    .venv\Scripts\python test_needle2.py --ref <folder>        # + Needle 1 is unchanged against the needle.py in <folder>
    .venv\Scripts\python test_needle2.py --speed 8             # training speed and GPU memory at batch 8 (random audio)

1. Needle(cfg) builds Needle 1 or Needle 2 from the config, and a checkpoint round-trips through the format
   train.py writes and eval.py / export read.
2. The band grid: stacking bins into rows loses nothing, and every bin takes its row's output.
3. Device-style chunked separation runs and the four stems add up to the mix.
4. The training loss runs, is finite and backpropagates into every part (stems, TDF, GRU, decoder, head).
5. The sub-frame envelope input (--sub): a click lands in the right time slot, a row takes it once.
6. (--ref) With the Needle 2 changes in needle.py, Needle 1 still gives bit-identical outputs.
"""
import argparse
import importlib.util
import tempfile
import time
from pathlib import Path

import torch

from chunked import separate_chunked, torch_predictor
from needle import Needle, NeedleConfig, apply_output, count_macs, network_input, spectrogram, subframe_envelope
from needle2 import Needle2, bs_config, randomize_norms, row_of_bin, to_bands
from train import check_alignment, compute_loss, crop_length, save_model

SMALL = dict(frames=32, widths=(8, 8, 16, 16, 16), rnn=1, rnn_hidden=8)


def test_dispatch():
    assert type(Needle(NeedleConfig(base=8))).__name__ == "Needle"
    cfg = bs_config(**SMALL)
    model = Needle(cfg)
    assert isinstance(model, Needle2)
    with tempfile.TemporaryDirectory() as d:
        save_model(Path(d) / "x.pt", model, cfg, step=1)
        ck = torch.load(Path(d) / "x.pt", map_location="cpu", weights_only=False)
    cfg2 = NeedleConfig(**ck["config"])
    assert cfg2 == cfg, (cfg2, cfg)
    Needle(cfg2).load_state_dict(ck["model"])
    old = {k: v for k, v in ck["config"].items() if k in ("base", "mult", "sources", "audio_channels", "sample_rate", "n_fft",
                                                         "hop", "bins", "frames", "coord", "harmonics", "taps")}
    assert NeedleConfig(**old).arch == "unet"               # a checkpoint from before Needle 2 has none of the new keys
    print("dispatch and checkpoint round trip: ok")


def test_bands():
    cfg = bs_config(**SMALL)
    x = torch.randn(2, cfg.in_channels, cfg.bins, 8)
    bands = to_bands(x, cfg)
    back, k = [], 0
    for band, (bins, s) in zip(bands, cfg.bands):           # undo the stacking: channels back to bins
        b, cs, rows, t = band.shape
        back.append(band.reshape(b, cs // s, s, rows, t).permute(0, 1, 3, 2, 4).reshape(b, cs // s, bins, t))
    assert torch.equal(torch.cat(back, 2), x)
    rb = row_of_bin(cfg)
    assert rb.shape == (cfg.bins,) and rb[0] == 0 and rb[-1] == cfg.rows - 1 and (rb[1:] - rb[:-1]).min() >= 0
    assert [int((rb == r).sum()) for r in (0, 128, 320, 448)] == [1, 2, 4, 16]
    print(f"band grid: ok ({cfg.bins} bins -> {cfg.rows} rows, nothing lost)")


def test_chunked():
    torch.manual_seed(0)
    cfg = bs_config(**SMALL)
    model = randomize_norms(Needle(cfg)).eval()
    wav = torch.randn(2, cfg.sample_rate * 3) * 0.1
    stems = separate_chunked(torch_predictor(model, 4), wav, cfg, overlap=8, batch=4)
    err = (stems.sum(0) - wav).abs().max().item()
    assert stems.shape == (4, 2, wav.shape[-1]) and err < 1e-4, (stems.shape, err)
    print(f"chunked separation: ok (stems add up to the mix within {err:.1e})")


def test_loss():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if dev == "cuda":
        torch.cuda.set_per_process_memory_fraction(0.15)    # a training run may own the rest of the card
    for cfg in (bs_config(**SMALL), bs_config(**SMALL, taps=0, cplx=False), bs_config(**SMALL, sub=4)):
        check_alignment(cfg, dev)
        model = Needle(cfg).to(dev).train()
        stems = torch.randn(2, 4, 2, crop_length(cfg), device=dev) * 0.05
        spec_norm = torch.hann_window(cfg.n_fft).pow(2).sum().sqrt().item()
        opt = torch.optim.AdamW(model.parameters(), 1e-3)
        for _ in range(3):                                  # a few steps: the residual branches start at 0 and must wake up
            loss, spec_l1, wave_l1 = compute_loss(model, stems, cfg, 0.1, torch.bfloat16 if dev == "cuda" else None, spec_norm, 0.5, 0.005)
            opt.zero_grad()
            loss.backward()
            opt.step()
        assert torch.isfinite(loss), loss
        dead = [n for n, p in model.named_parameters() if p.grad is None or not torch.isfinite(p.grad).all() or p.grad.abs().sum() == 0]
        assert not dead, f"no gradient in: {dead[:8]}"
        print(f"loss ok on {dev} (taps {cfg.taps}, cplx {cfg.cplx}, sub {cfg.sub}): {loss.item():.4f} spec {spec_l1.item():.4f} wave {wave_l1.item():.4f}; "
              f"every parameter gets a gradient")


def test_sub():
    cfg = bs_config(**SMALL, sub=4)
    cfg.check()
    g, e, slot = cfg.sub_block, cfg.sub_channels, cfg.hop // cfg.sub
    assert (g, e, cfg.in_channels, cfg.band_in(16)) == (16, 8, 14, 6 * 16 + 8)
    # A click 300 samples after the centre of frame 20: its hop starts 512 samples before the centre, so the click is
    # 812 samples into it = slot 3 (of 256 samples each). A frame is centred on t * hop (center=True).
    wav = torch.zeros(1, 2, cfg.hop * 40)
    wav[..., 20 * cfg.hop + 300] = 1.0
    env = subframe_envelope(spectrogram(wav, cfg), cfg)[0, :cfg.sub]                # left channel: (slots, bins, T)
    assert env.shape == (cfg.sub, cfg.bins, 41)
    for block in (0, 40, cfg.bins // g - 1):
        top = env[:, block * g].flatten().argmax().item()                            # index = slot * T + frame
        assert (top // 41, top % 41) == (3, 20), (block, top // 41, top % 41)
    assert torch.equal(env[:, 5 * g], env[:, 6 * g - 1])                             # one value per block
    # A steady tone fills every slot alike (the Hann window is 0.92 to 1 over the middle hop).
    n = torch.arange(cfg.hop * 40)
    tone = torch.sin(2 * torch.pi * 8000 / cfg.sample_rate * n)[None, None].repeat(1, 2, 1)
    env = torch.expm1(subframe_envelope(spectrogram(tone, cfg), cfg)[0, :cfg.sub, int(8000 * cfg.n_fft / cfg.sample_rate), 10:30])
    assert env.max() / env.min() < 1.15, (env.max() / env.min()).item()
    # The band grid stacks the per-bin channels and takes the envelope once per row.
    x = network_input(torch.randn(2, 2, cfg.bins + 1, 8, dtype=torch.complex64), cfg)
    assert x.shape == (2, cfg.in_channels, cfg.bins, 8)
    k = 0
    for band, (bins, s) in zip(to_bands(x, cfg), cfg.bands):
        b, c, rows, t = band.shape
        assert c == cfg.band_in(s) and torch.equal(band[:, c - e:], x[:, -e:, k:k + bins:s])
        main = band[:, :c - e].reshape(b, (c - e) // s, s, rows, t).permute(0, 1, 3, 2, 4).reshape(b, (c - e) // s, bins, t)
        assert torch.equal(main, x[:, :-e, k:k + bins])
        k += bins
    torch.manual_seed(0)
    model = randomize_norms(Needle(cfg)).eval()
    wav = torch.randn(2, cfg.sample_rate * 3) * 0.1
    stems = separate_chunked(torch_predictor(model, 4), wav, cfg, overlap=8, batch=4)
    err = (stems.sum(0) - wav).abs().max().item()
    assert stems.shape == (4, 2, wav.shape[-1]) and err < 1e-4, (stems.shape, err)
    print(f"sub-frame envelope: ok (a click lands in its {slot}-sample slot, {e} extra channels per row, chunked separation runs)")


def test_needle1_unchanged(ref_dir, checkpoint=None):
    spec = importlib.util.spec_from_file_location("needle_ref", Path(ref_dir) / "needle.py")
    ref = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ref)
    import needle as new
    pairs = [(ref.NeedleConfig(base=8), new.NeedleConfig(base=8)),
             (ref.NeedleConfig(base=8, coord=True, harmonics=(2, 3, 4), taps=5), new.NeedleConfig(base=8, coord=True, harmonics=(2, 3, 4), taps=5))]
    states = [None, None]
    if checkpoint:
        ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
        pairs.append((ref.NeedleConfig(**ck["config"]), new.NeedleConfig(**ck["config"])))
        states.append(ck["model"])
    for (c_ref, c_new), state in zip(pairs, states):
        torch.manual_seed(0)
        m_ref = ref.Needle(c_ref).eval()
        torch.manual_seed(0)
        m_new = new.Needle(c_new).eval()
        if state:
            m_ref.load_state_dict(state)
            m_new.load_state_dict(state)
        torch.manual_seed(1)
        wav = torch.randn(1, 2, 44100 * 2) * 0.1
        s_ref, s_new = ref.spectrogram(wav, c_ref), new.spectrogram(wav, c_new)
        x_ref, x_new = ref.network_input(s_ref, c_ref), new.network_input(s_new, c_new)
        t = x_ref.shape[-1] // 2 ** c_ref.levels * 2 ** c_ref.levels
        with torch.no_grad():
            o_ref, o_new = m_ref(x_ref[..., :t]), m_new(x_new[..., :t])
            y_ref = ref.apply_output(o_ref, s_ref[..., :t], c_ref)
            y_new = new.apply_output(o_new, s_new[..., :t], c_new)
        assert torch.equal(x_ref, x_new) and torch.equal(o_ref, o_new) and torch.equal(y_ref, y_new)
        assert ref.count_macs(c_ref) == new.count_macs(c_new)
        print(f"Needle 1 unchanged (base {c_ref.base}, taps {c_ref.taps}{', checkpoint' if state else ''}): "
              f"inputs, outputs and stems bit-identical, {new.count_macs(c_new)['params']:,} parameters")


def speed(batch, steps=12, gpu_mem=0.3):
    """Steps per second and peak GPU memory of the default Needle 2 on random audio (the optimiser step
    included, data loading not)."""
    dev = "cuda"
    torch.cuda.set_per_process_memory_fraction(gpu_mem)
    torch.backends.cudnn.benchmark = True
    cfg = bs_config()
    model = Needle(cfg).to(dev).train()
    opt = torch.optim.AdamW(model.parameters(), 1e-3, fused=True)
    spec_norm = torch.hann_window(cfg.n_fft).pow(2).sum().sqrt().item()
    stems = torch.randn(batch, 4, 2, crop_length(cfg), device=dev) * 0.05
    info = count_macs(cfg)
    times = []
    for i in range(steps):
        t0 = time.time()
        loss, _, _ = compute_loss(model, stems, cfg, 0.1, torch.bfloat16, spec_norm, 0.5, 0.005)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        torch.cuda.synchronize()
        times.append(time.time() - t0)
    best = sorted(times[2:])[len(times[2:]) // 2]            # median, after cuDNN has picked its kernels
    print(f"Needle 2 ({info['params']:,} parameters, {info['gmac_per_audio_s']:.2f} GMAC/s), batch {batch} x {crop_length(cfg) / cfg.sample_rate:.2f} s: "
          f"{1 / best:.2f} steps/s (median of {len(times) - 2}), peak GPU memory {torch.cuda.max_memory_allocated() / 2 ** 30:.2f} GiB, "
          f"loss {loss.item():.4f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", help="folder with the needle.py to compare Needle 1 against")
    ap.add_argument("--checkpoint", help="with --ref: also compare on this Needle 1 checkpoint")
    ap.add_argument("--speed", type=int, help="only measure training speed at this batch size")
    ap.add_argument("--gpu-mem", type=float, default=0.3)
    a = ap.parse_args()
    if a.speed:
        speed(a.speed, gpu_mem=a.gpu_mem)
    else:
        test_dispatch()
        test_bands()
        test_chunked()
        test_loss()
        test_sub()
        if a.ref:
            test_needle1_unchanged(a.ref, a.checkpoint)
        print("all ok")
