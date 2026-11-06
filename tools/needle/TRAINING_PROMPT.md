# Prompt: train Needle on the desktop

Paste everything below the line into Claude Code, started in a checkout of this repository on
the training PC (Windows or Linux, Ryzen 9 9950X3D, RTX 3080 Ti 12 GB, 128 GB RAM).

---

You are training **Needle**, a tiny stem separator (drums, bass, other, vocals) for a DJ deck
whose NPU is an Arm Ethos-U55-256 (Renesas RA8P1). The model, its STFT front end and the
export/compile check are already written and tested: read `tools/needle/README.md`,
`tools/needle/needle.py` and `tools/needle/export_vela.py` first. **Do not change the network's
layer types**, because every layer was chosen so that the NPU runs 100 % of it. You may change
`base`, `mult` and `frames`, and add training-only parts (losses, augmentation, BatchNorm).
After any architecture change, run `export_vela.py` and check it still reports 0 CPU operators.

Work in `tools/needle/`. Commit after each working step on a new branch `needle/training`,
and keep large files (datasets, checkpoints, caches) out of git (add them to `.gitignore`).
Explain what you do as you go: I'm learning, and I want to understand the choices.

## 1. Environment
- Python 3.11 virtual environment in `tools/needle/.venv` with PyTorch + CUDA for the RTX 3080 Ti,
  torchaudio, soundfile, numpy, tqdm, tensorboard, `musdb`, `museval`, and `demucs==4.0.1` (the teacher).
- Check that the GPU is seen and that `python needle.py` passes its self-test.
- Export needs `tensorflow-cpu==2.18.*` and `ethos-u-vela==5.2.0` (a second venv is fine). If Vela
  won't install on Windows, use WSL2 for the export step only.

## 2. Data
- **MUSDB18-HQ** (150 tracks with real stems; 100 train, 50 test) from Zenodo
  (https://zenodo.org/records/3338373). It's licensed for non-commercial research. Ask me to
  download it if it needs accepting terms, then put it at a path I give you.
- **Distillation data from my own music** (makes the tiny model much better than MUSDB alone):
  run the teacher `htdemucs_ft` from the `demucs` package on the GPU over a music folder I give you,
  and save its four stems as 44.1 kHz 16-bit FLAC in a cache folder. Resume where it stopped if
  interrupted, and log tracks that fail. These are for my private training only; never commit or
  upload them.
- Everything is 44.1 kHz stereo; resample anything that isn't (training only, never the
  originals).

## 3. Training
- **Input/output** exactly as `needle.separate()`: STFT (n_fft 2048, hop 512), log1p magnitude of
  the first 1024 bins in, sigmoid masks out, masks normalised to sum to 1.
- **Chunks:** train on random crops of `cfg.frames` frames, so training matches the device. Train
  at frames=32 first; also try 64, which needs the SDRAM on the device.
- **Loss:** L1 between the masked mixture magnitude and the target stem magnitude per source,
  plus 0.1 × L1 on the waveform after iSTFT (multi-resolution STFT loss is a good alternative).
  For distillation tracks, the targets are the teacher's stems.
- **Augmentation:** random gain (±6 dB per stem), stereo swap, polarity flip, and **remixing**
  (stems from different songs in one batch; this is the strongest one for small data). Optional:
  ±2 semitone pitch shift and tempo change on the fly.
- **Optimiser:** AdamW, lr 1e-3, weight decay 1e-4, cosine decay, warm-up 1k steps, batch 64,
  mixed precision (bf16 or fp16), gradient clipping at 5. Train until validation stops improving
  (start with about 200k steps for base 16). Use a 14-track validation split taken from the
  MUSDB train set; never tune on the 50 test tracks.
- Do the STFT on the GPU (`torch.stft`), keep audio in RAM (there's 128 GB), and use several
  DataLoader workers, so the GPU isn't waiting.
- Save checkpoints as `{"model": state_dict, "config": asdict(cfg)}`; `export_vela.py
  --checkpoint` reads exactly that.

## 4. Sizes to compare
Train base 16, then base 12 and base 8 (keep `mult` and the rest the same), all at frames=32.
If time allows, base 16 at frames=64. Stop and tell me if base 16 doesn't beat about 4 dB SDR
on validation: that would mean something is wrong in the data or loss, not that the model is too small.

## 5. Evaluation (what I care about)
1. **SDR on the 50 MUSDB18-HQ test tracks** with `museval` (median over frames, then over
   tracks, per stem and average). Use chunked inference exactly as on the device: `frames`-frame
   chunks, with 8-frame overlap at each side and a crossfade.
2. **The same for the quantised models.** Run each `needle_int8.tflite` and `needle_int16.tflite`
   from `export_vela.py --checkpoint ... --audio <some training tracks>` with the TFLite interpreter
   on CPU (use all 16 cores), chunked the same way. Report the SDR loss from quantisation. If int8
   loses more than about 0.5 dB, try quantisation-aware fine-tuning and explain the options to me.
3. **Baselines:** run the teacher (`htdemucs_ft`) on the same test tracks for comparison.
4. **A listening set:** 30-second excerpts of 5 test tracks, as stems from the teacher, float
   Needle and int8/int16 Needle, in `tools/needle/listening/` (git-ignored), named so I can
   compare by ear.

## 6. Deliverables
- `tools/needle/train.py`, `eval.py`, `eval_tflite.py`, `distill.py` (teacher stems), readable and
  commented, with their commands in the README.
- A results table in `tools/needle/README.md` for each size:
  - parameters
  - SDR per stem and average: float, int8, int16
  - the Vela numbers for the trained checkpoint (ms per chunk, RTF, arena, weights, CPU operators)
  - training time
- A short recommendation: which size and precision for the RA8P1 with on-chip SRAM only, and
  which with the 128 MB SDRAM.

Ask me before downloading anything large, deleting anything, or running longer than about
12 hours in one go. Report progress (loss curves, validation SDR) as you go.
