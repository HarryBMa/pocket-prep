# Needle: tiny stem separation for an Ethos-U55 NPU (Renesas RA8P1)

Research tooling, not part of the app. Goal: find out whether live stem separation can run on
an RA8P1 (1 GHz Cortex-M85 + Ethos-U55-256 at 500 MHz, 2 MB SRAM, optional external SDRAM), and
train the model for it. The app keeps pre-rendered Demucs stems; this is an experiment.

- `needle.py`: the model (PyTorch), its front and back end (STFT, masks, iSTFT), `separate()` for
  float reference separation, BatchNorm folding, MAC counting. `python needle.py` self-tests.
- `export_vela.py`: PyTorch -> the same network in Keras -> int8 / int16 TFLite (post-training
  quantisation) -> Arm Vela for the U55, with two RA8P1 memory set-ups. Writes `build/.../report.json`.
- `TRAINING_PROMPT.md`: the brief for training it on a desktop GPU.

## The model
A U-Net on log-magnitude spectrograms (n_fft 2048, hop 512, 1024 bins, stereo in), 5 levels of
two 3x3 convolutions, 2x2 strides down, 2x2 nearest upsampling up, skip concatenations, then a 1x1
convolution and a sigmoid: one mask per source (drums, bass, other, vocals) and channel. The
Cortex-M85 does the STFT, log, mask normalisation (masks sum to 1, so stems add up to the mix)
and iSTFT. Every layer is chosen from what Vela puts on the U55: CONV_2D, PAD, CONCATENATION,
RESIZE_NEAREST_NEIGHBOR (equal 2x factors only, hence 2x2 strides), LOGISTIC. No attention,
LSTM or normalisation layers at inference.

## Results (Vela 5.2.0, Ethos-U55-256 @ 500 MHz, random weights)
Chunks of `frames` STFT frames (32 = 0.37 s, 64 = 0.74 s). RTF = time per chunk / chunk length
(0.02 = 50x faster than real time). Arena = working memory for feature maps. All runs: **0 CPU
operators** (everything on the NPU).

| base | params | GMAC per audio s | mode | arena in SRAM: ms / RTF | arena in SDRAM: ms / RTF | arena | weights |
|---|---|---|---|---|---|---|---|
| 8, frames 32 | 126 k | 0.31 | int8 | 2.9 / 0.008 | 10.3 / 0.028 | 1.0 MB | 141 KB |
| 8, frames 32 | | | int16 | 5.8 / 0.016 | 27.0 / 0.073 | 2.0 MB | 132 KB |
| 12, frames 32 | 282 k | 0.67 | int8 | 5.7 / 0.015 | 13.8 / 0.037 | 1.0 MB | 326 KB |
| 12, frames 32 | | | int16 | 11.3 / 0.031 | 30.9 / 0.083 | 2.0 MB | 310 KB |
| 16, frames 32 | 501 k | 1.18 | int8 | 7.1 / 0.019 | 14.2 / 0.038 | 1.5 MB | 520 KB |
| 16, frames 32 | | | int16 | 13.9 / 0.037 | 37.0 / 0.100 | 3.2 MB | 496 KB |
| 16, frames 64 | 501 k | 1.18 | int8 | 14.0 / 0.019 | 28.7 / 0.039 | 3.0 MB | 520 KB |
| 16, frames 64 | | | int16 | 27.5 / 0.037 | 75.2 / 0.101 | 6.4 MB | 496 KB |

What it means:
- **Compute is not the limit**: every size runs 10-100x faster than real time on the NPU.
- **Memory is**: with only the 2 MB on-chip SRAM, int8 with base <= 12 and 32-frame chunks fits
  (1 MB arena, leaving room for the firmware). 16-bit activations or longer chunks need the
  external SDRAM, where it still runs 10x faster than real time.
- **Quality is unknown until trained**: the quantisation error above is measured on random
  weights and only shows the pipeline works. Expect Spleeter-class quality (about 5-6 dB SDR
  on MUSDB18) for base 16, lower for base 8, against about 9 dB for HTDemucs. Training decides.
- Not counted: the Cortex-M85 work (STFT, iSTFT, masks: roughly 10 FFTs of 2048 points per frame,
  86 frames/s; estimated at a few percent of the M85 with CMSIS-DSP, unmeasured).
- Bandwidths in the Vela config are estimates (on-chip SRAM 4 GB/s, MRAM 1 GB/s, SDRAM 350 MB/s
  usable), not Renesas datasheet values. Check them against the RA8P1 hardware manual.

## Running it
    pip install torch tensorflow-cpu==2.18.* ethos-u-vela==5.2.0 soundfile
    python needle.py                                    # self-test and MAC counts
    python export_vela.py --base 12 --frames 32         # random weights: speed and memory
    python export_vela.py --checkpoint needle.pt --frames 32 --audio <folder of 44.1 kHz stereo tracks>

On Windows, if `ethos-u-vela` doesn't install, run the export in WSL2 or on Linux. (With uv on
Windows 11, ethos-u-vela 5.2.0 installs and runs natively.)

## Training

Files:
- `train.py`: training (MUSDB18-HQ, optionally teacher stems), checkpoints in the format `export_vela.py --checkpoint` reads.
- `data.py`: audio in RAM as int16, the official MUSDB split, random crops with remixing, GPU augmentation.
- `chunked.py`: device-style chunked inference (chunks of `frames` frames, 8-frame overlap at each side, crossfade), used by validation and both evaluations.
- `distill.py`: teacher stems (Demucs `htdemucs_ft`) for a music folder or for MUSDB mixtures, resumable.
- `eval.py`: museval SDR of a checkpoint or of stems on disk (median over 1-s windows, then over tracks).
- `eval_tflite.py`: the same for `needle_int8.tflite` / `needle_int16.tflite` on the CPU.

### Environments (Windows, uv)
Two virtual environments, because TensorFlow 2.18 needs an older numpy than the rest:

    cd tools\needle
    uv venv .venv --python 3.11
    uv pip install --python .venv\Scripts\python.exe torch==2.7.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128
    uv pip install --python .venv\Scripts\python.exe soundfile numpy tqdm tensorboard musdb museval demucs==4.0.1
    uv venv .venv-export --python 3.11
    uv pip install --python .venv-export\Scripts\python.exe torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu
    uv pip install --python .venv-export\Scripts\python.exe "tensorflow-cpu==2.18.*" "ethos-u-vela==5.2.0" soundfile museval tqdm

`.venv` trains, distills and evaluates float models (GPU); `.venv-export` exports and evaluates TFLite models (CPU).

### Data (in F:\needle\data, outside the repository, never committed)
- MUSDB18-HQ (https://zenodo.org/records/3338373, non-commercial research licence) unzipped to
  `F:\needle\data\musdb18hq` (`train\`, `test\`). 86 training tracks, 14 validation tracks (musdb's
  official split, taken from the training set), 50 test tracks (only for the final numbers).
- Teacher stems: `F:\needle\data\teacher` (your music) and `F:\needle\data\teacher_musdb_test` (baseline, listening set).
- Real stems (import_stems.py, same layout as teacher stems) in `F:\needle\data\stems\<source>`:
  NI free Stems, mshoxxDB, sample-pack construction kits and loops, MoisesDB; held-out test sets in
  `*_test` folders (never trained on). Deliberately imperfect sets (SDXDB23 bleeding / label noise)
  in `F:\needle\data\stems_noisy\` so they never mix into a clean run by accident.

### Commands
    # teacher stems (htdemucs_ft on the GPU; resumable, failures in <out>\failed.tsv)
    .venv\Scripts\python distill.py --musdb F:\needle\data\musdb18hq --subset test --out F:\needle\data\teacher_musdb_test
    .venv\Scripts\python distill.py --input "<music folder>" --out F:\needle\data\teacher

    # train (writes runs\<name>\best.pt, last.pt, log.csv, val.csv, TensorBoard events)
    .venv\Scripts\python train.py --musdb F:\needle\data\musdb18hq --base 16 --frames 32 --name b16_f32
    .venv\Scripts\python train.py ... --teacher F:\needle\data\teacher --teacher-share 0.5 --name b16_f32_kd
    .venv\Scripts\python train.py ... --name b16_f32 --resume           # continue a stopped run
    .venv\Scripts\tensorboard --logdir runs                             # loss and validation curves

    # float SDR on the 50 test tracks, and the teacher baseline
    .venv\Scripts\python eval.py --checkpoint runs\b16_f32\best.pt --musdb F:\needle\data\musdb18hq
    .venv\Scripts\python eval.py --estimates F:\needle\data\teacher_musdb_test --name htdemucs_ft --musdb F:\needle\data\musdb18hq

    # quantise (calibrated on training mixtures, never test), compile with Vela, then int8/int16 SDR
    .venv-export\Scripts\python export_vela.py --checkpoint runs\b16_f32\best.pt --frames 32 --out build\b16_f32 `
        --audio F:\needle\data\musdb18hq\train --audio-glob "*/mixture.wav"
    .venv-export\Scripts\python eval_tflite.py --tflite build\b16_f32\needle_b16_f32\needle_int8.tflite `
        --musdb F:\needle\data\musdb18hq --float results\b16_f32_test.json
