"""Export Demucs (htdemucs) to ONNX in ClubCraft's model format. Run once by a developer:

    uv run --with demucs==4.0.1 --with torch==2.7.1 --with onnx --with onnxruntime \
        tools/export_demucs_onnx.py models

Stem separation runs in ClubCraft's `clubcraft-stems` crate (MIT OR Apache-2.0,
https://github.com/beneaze/ClubCraft). It loads `<dir>/htdemucs.json` (manifest) and
`<dir>/htdemucs.onnx`. The graph is the network between HTDemucs's STFT and iSTFT,
which the crate does in Rust:

Inputs:  mix  [1, 2, 343980]         one 7.8 s stereo segment at 44.1 kHz
         spec [1, 4, 2048, 336]      its STFT, real/imag as channels ("cac")
Outputs: spec_out [1, 4, 4, 2048, 336]   per source (drums, bass, other, vocals)
         wave_out [1, 4, 2, 343980]      per source, time-branch waveform

This mirrors ClubCraft's tools/model-export/export_htdemucs.py.

On a Mac, `--coreml` makes `<dir>/htdemucs.mlmodelc` instead (CoreML ML Program, fp32,
CPU+GPU, macOS 13+), which clubcraft-stems prefers there (several times faster):

    uv run --python 3.11 --with demucs==4.0.1 --with torch==2.5.1 --with torchaudio==2.5.1 \
        --with coremltools==8.3 --with "numpy<2" tools/export_demucs_onnx.py models --coreml
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from demucs.pretrained import get_model


class Core(torch.nn.Module):
    """HTDemucs.forward without the spectral transforms (normalisation included)."""

    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, mix, mag):
        m = self.m
        x = mag
        B, C, Fq, T = x.shape
        mean = x.mean(dim=(1, 2, 3), keepdim=True)
        std = x.std(dim=(1, 2, 3), keepdim=True)
        x = (x - mean) / (1e-5 + std)
        xt = mix
        meant = xt.mean(dim=(1, 2), keepdim=True)
        stdt = xt.std(dim=(1, 2), keepdim=True)
        xt = (xt - meant) / (1e-5 + stdt)

        saved, saved_t, lengths, lengths_t = [], [], [], []
        for idx, encode in enumerate(m.encoder):
            lengths.append(x.shape[-1])
            inject = None
            if idx < len(m.tencoder):
                lengths_t.append(xt.shape[-1])
                tenc = m.tencoder[idx]
                xt = tenc(xt)
                if not tenc.empty:
                    saved_t.append(xt)
                else:
                    inject = xt
            x = encode(x, inject)
            if idx == 0 and m.freq_emb is not None:
                frs = torch.arange(x.shape[-2], device=x.device)
                emb = m.freq_emb(frs).t()[None, :, :, None].expand_as(x)
                x = x + m.freq_emb_scale * emb
            saved.append(x)
        if m.crosstransformer:
            if m.bottom_channels:
                b, c, f, t = x.shape
                x = x.reshape(b, c, f * t)
                x = m.channel_upsampler(x)
                x = x.reshape(b, -1, f, t)
                xt = m.channel_upsampler_t(xt)
            x, xt = m.crosstransformer(x, xt)
            if m.bottom_channels:
                b, c, f, t = x.shape
                x = x.reshape(b, c, f * t)
                x = m.channel_downsampler(x)
                x = x.reshape(b, -1, f, t)
                xt = m.channel_downsampler_t(xt)
        for idx, decode in enumerate(m.decoder):
            skip = saved.pop(-1)
            x, pre = decode(x, skip, lengths.pop(-1))
            offset = m.depth - len(m.tdecoder)
            if idx >= offset:
                tdec = m.tdecoder[idx - offset]
                length_t = lengths_t.pop(-1)
                if tdec.empty:
                    pre = pre[:, :, 0]
                    xt, _ = tdec(pre, None, length_t)
                else:
                    xt, _ = tdec(xt, saved_t.pop(-1), length_t)
        S = len(m.sources)
        x = x.view(B, S, -1, Fq, T) * std[:, None] + mean[:, None]
        xt = xt.view(B, S, -1, mix.shape[-1]) * stdt[:, None] + meant[:, None]
        return x, xt


def export_coreml(model, out_dir: Path) -> None:
    """CoreML ML Program (fp32, CPU+GPU), compiled to .mlmodelc by macOS's CoreML."""
    import shutil
    import coremltools as ct
    length = int(model.segment * model.samplerate)
    frames = math.ceil(length / model.hop_length)
    mix, mag = torch.zeros(1, 2, length), torch.zeros(1, 4, model.nfft // 2, frames)
    with torch.no_grad():
        traced = torch.jit.trace(Core(model).eval(), (mix, mag), check_trace=False)
    ml = ct.convert(
        traced,
        inputs=[ct.TensorType(name="mix", shape=mix.shape), ct.TensorType(name="spec", shape=mag.shape)],
        outputs=[ct.TensorType(name="spec_out"), ct.TensorType(name="wave_out")],
        convert_to="mlprogram", compute_precision=ct.precision.FLOAT32,
        compute_units=ct.ComputeUnit.CPU_AND_GPU, minimum_deployment_target=ct.target.macOS13,
    )
    package = out_dir / "htdemucs.mlpackage"
    shutil.rmtree(package, ignore_errors=True)
    ml.save(str(package))
    shutil.rmtree(out_dir / "htdemucs.mlmodelc", ignore_errors=True)
    # Compile with macOS's own CoreML (no Xcode needed, unlike `xcrun coremlcompiler`).
    compiled = ct.models.utils.compile_model(str(package))
    shutil.move(str(compiled), str(out_dir / "htdemucs.mlmodelc"))
    shutil.rmtree(package)                       # only the compiled model is needed at runtime

    # Check against PyTorch on the same test segment as the ONNX export.
    g = torch.Generator().manual_seed(1)
    t = torch.arange(length) / model.samplerate
    x = 0.1 * torch.randn(1, 2, length, generator=g) + 0.3 * torch.sin(2 * torch.pi * 110 * t)
    with torch.no_grad():
        ref = Core(model).eval()(x, model._magnitude(model._spec(x)))
    got = ct.models.CompiledMLModel(str(out_dir / "htdemucs.mlmodelc")).predict(
        {"mix": x.numpy(), "spec": model._magnitude(model._spec(x)).numpy()})
    for name, r in zip(["spec_out", "wave_out"], ref):
        r = r.numpy()
        sdr = 10 * np.log10((r ** 2).sum() / max(((r - got[name]) ** 2).sum(), 1e-30))
        print(f"CoreML vs PyTorch {name}: {sdr:.1f} dB SDR")
        assert sdr > 40, "CoreML model differs from PyTorch"
    print("wrote", out_dir / "htdemucs.mlmodelc")


def main(out_dir: Path) -> None:
    torch.backends.mha.set_fastpath_enabled(False)
    model = get_model("htdemucs").models[0].eval()
    assert model.sources == ["drums", "bass", "other", "vocals"], model.sources
    length = int(model.segment * model.samplerate)
    frames = math.ceil(length / model.hop_length)
    core = Core(model).eval()

    # Check the cut-out network plus PyTorch's own STFT against the real model.
    g = torch.Generator().manual_seed(1)
    t = torch.arange(length) / model.samplerate
    mix = 0.1 * torch.randn(1, 2, length, generator=g) + 0.3 * torch.sin(2 * torch.pi * 110 * t)
    with torch.no_grad():
        z = model._spec(mix)
        mag = model._magnitude(z)
        full = model(mix)
        x, xt = core(mix, mag)
        rebuilt = model._ispec(model._mask(z, x), length) + xt
    err = (rebuilt - full).abs().max().item()
    print(f"core + STFT vs full model: max error {err:.2e}")
    assert err < 1e-4

    out_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = out_dir / "htdemucs.onnx"
    torch.onnx.export(core, (mix, mag), str(onnx_path), input_names=["mix", "spec"],
                      output_names=["spec_out", "wave_out"], opset_version=17,
                      do_constant_folding=True, dynamo=False)

    import onnxruntime as ort
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    ox, oxt = sess.run(None, {"mix": mix.numpy(), "spec": mag.numpy()})
    err = max(np.abs(ox - x.numpy()).max(), np.abs(oxt - xt.numpy()).max())
    print(f"ONNX vs PyTorch: max error {err:.2e}")
    assert err < 1e-2, "ONNX export differs from PyTorch"

    manifest = {
        "name": "htdemucs", "architecture": "htdemucs", "sources": list(model.sources),
        "samplerate": model.samplerate, "audio_channels": model.audio_channels,
        "segment_samples": length, "spec_frames": frames, "nfft": model.nfft,
        "hop_length": model.hop_length, "models": ["htdemucs"], "weights": [[1.0] * 4],
    }
    (out_dir / "htdemucs.json").write_text(json.dumps(manifest, indent=2))
    print("wrote", onnx_path, f"({onnx_path.stat().st_size / 1e6:.0f} MB) and htdemucs.json")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    out = Path(args[0] if args else "models")
    if "--coreml" in sys.argv:
        torch.backends.mha.set_fastpath_enabled(False)   # the fused attention path can't be traced
        export_coreml(get_model("htdemucs").models[0].eval(), out)
    else:
        main(out)
