# Needle 2: the band-split separator (started 12 Nov 2026)

Needle 1 (`needle.py`) is a plain convolutional U-Net. It stopped improving: doubling its size gave about +0.1 dB
(`SCALING.md` §2b), and it sits at 4.4-4.5 dB on MUSDB18-HQ like every other plain mask U-Net. The research
blueprint of 12 Nov 2026 (project doc `DECK001_research_blueprint_2026-11.md`) found that small separators which do
reach 7-9 dB all share three things Needle 1 lacks: a band-split frequency grid, a layer that sees the whole spectrum
at once, and a sequence model over time. Needle 2 (`needle2.py`, `NeedleConfig.arch = "bs"`) adds those, using only
what the Ethos-U55 runs.

No Needle 2 model has been trained yet. Everything below about speed and memory is measured with Vela 5.2.0 on random
weights; nothing below is a quality result.

## What the NPU can run (Vela 5.2.0, RA8P1 configs of `export_vela.py`, arena in SDRAM)

`vela_probe.py` compiles one small graph per building block at the tensor sizes Needle 2 uses. Times are per block.

| Block | CPU operators | Time | Verdict |
|---|---|---|---|
| Band inputs + one conv each, joined along frequency | 0 | 26 ms | used |
| Conv with stride 2 in frequency only | 0 | 6 ms | used |
| Frequency-only 2x upsampling as CONCATENATION + RESHAPE | 0 | 7 ms | used |
| Frequency-only upsampling as RESIZE or TRANSPOSE_CONV | 1 | - | not possible on the U55 |
| TDF (TRANSPOSE, two 1x1 convs, TRANSPOSE, ADD) at 64 x 32 x 128 | 0 | 6 ms | used |
| Four dilated 3x3 convs at 32 x 32 x 160 | 0 | 22 ms | possible, not used |
| Attention, 4 heads, at 32 x 32 x 160 | 0 | 62 ms | possible but slow: BATCH_MATMUL is memory-bound |
| GRU written out step by step, 32 steps, one direction | 0 | about 9 ms (37 ms for a dual-path block of four) | used |
| REVERSE_V2 (for a backward GRU) | on the CPU | - | avoided: the backward pass is the same wiring walked from the other end |
| Last decoder stage with a concatenated skip (96 channels at 512 x 128) | 0 | 54 ms, 9.0 MiB arena | replaced by an added skip |

Two results decided the design. Attention does run on the NPU (the earlier assumption that it cannot is out of date),
but one block costs 62 ms against about 9 ms for a GRU direction, so the sequence model is a GRU. And the
full-resolution concat was the largest tensor by far, so level 0 adds its skip instead. (An LSTM written out the same
way took 6 ms per direction in int8 and 13 ms with 16-bit activations; it is not used because its cell state is
unbounded, which int8 handles worse than a GRU's state between -1 and 1.)

## The model

| Part | What it does |
|---|---|
| STFT (M85) | n_fft 4096, hop 1024, 2048 bins: 10.8 Hz per bin; a 128-frame chunk is 2.97 s |
| Input (M85) | per bin and channel: log1p magnitude, and the complex value with its magnitude log-compressed (6 channels) |
| Band grid (M85) | 512 rows: 1 bin per row to 1.4 kHz, 2 to 5.5 kHz, 4 to 11 kHz, 16 above; a row's bins are stacked into channels |
| Encoder (NPU) | 5 levels, widths 32 / 64 / 96 / 128 / 160; two 2x2 steps, then two frequency-only steps (the deepest level is 32 rows x 32 frames) |
| TDF (NPU) | fully connected layer along frequency at levels 2-4, encoder and decoder |
| Dual-path GRU (NPU) | 2 blocks at the deepest level: a bidirectional GRU along time per row, then along frequency per frame (96 units per direction) |
| Decoder (NPU) | one conv per level; concatenated skips, an added skip at level 0 |
| Output (NPU -> M85) | one 3-tap complex filter per row, source and channel; the M85 uses it for all bins of the row; "other" is the residual |

The interface is Needle 1's (`forward(network_input) -> (B, K, bins, frames)`), so `train.py`, `chunked.py`, `eval.py`
and the scans are shared. `Needle(cfg)` builds whichever model the config asks for. Needle 1 is unchanged: the
same checkpoint gives bit-identical stems before and after (`test_needle2.py --ref`).

## Size, speed and memory (random weights)

| | Needle 1 (`b20`, 128 frames; `SCALING.md`) | Needle 2, default |
|---|---|---|
| Parameters | 0.78 M | 2.25 M |
| Compute | 1.83 GMAC per second of audio | 2.43 |
| Chunk | 1.49 s | 2.97 s |
| NPU time per chunk, arena in SDRAM | 97 ms (RTF 0.066) | 250 ms (RTF 0.084) |
| NPU time per chunk, at on-chip SRAM speed | 49 ms | 95 ms (RTF 0.032) |
| Arena | 8.0 MiB | 5.9 MiB |
| Weights (int8) | 0.86 MiB | 2.97 MiB |
| Operators left on the CPU | 0 | 0 (of 3,297) |

Each dual-path GRU block costs about 37 ms per chunk (no GRU: 177 ms; three blocks: 287 ms). The budget is RTF 0.1
per player.

The TensorFlow graph that is exported agrees with the PyTorch model to 3e-7. With int8 weights and activations
(post-training, random weights, calibrated on MUSDB mixtures) the outputs differ from float by 0.003 on average, and
one test track scored the same in int8 as in float. Whether a trained model's GRUs survive int8 as well is the first
thing to check after training; the fallback is 16-bit activations for the GRU part only.

## Files

| File | What |
|---|---|
| `needle2.py` | the model, the band grid (`to_bands`, `row_of_bin`), `bs_config()`, self-test |
| `needle2_tf.py` | the same network as TensorFlow operations with the PyTorch weights, BatchNorm folded |
| `export_vela2.py` | int8 TFLite + Vela report (random weights or `--checkpoint`) |
| `vela_probe.py` | the building-block probes above |
| `test_needle2.py` | checks without training; `--ref <folder>` Needle 1 unchanged; `--speed 8` training speed |
| `needle.py`, `train.py`, `chunked.py`, `eval_tflite.py` | extended: config fields, `--arch bs`, evaluation in slices, band inputs for int8 |

```powershell
.venv\Scripts\python needle2.py                        # self-test, parameters, GMAC
.venv\Scripts\python test_needle2.py                   # all checks
.venv-export\Scripts\python export_vela2.py            # random weights: speed and memory on the RA8P1
.venv-export\Scripts\python vela_probe.py              # the building blocks
.venv\Scripts\python train.py --musdb F:\needle\data\musdb18hq --arch bs --batch 8 --name n2_a ...
```

## First run: `n2_a` (`queue_n2a.example.ps1`, live copy `F:\needle\queue_n2a.ps1`)

Same data, shares, augmentation and losses as `b20_jam3_frontier_env` (queue v16), so the two differ only in the
model. Batch 8 x 3.09 s is the same amount of audio per step as Needle 1's 16 x 1.5 s. The queue waits for the GPU,
measures the training speed, picks the step count that fits one night (at most 150,000), trains, runs the five
evaluations, compares song by song with `b20_jam3_frontier_env` (`paired.py`), runs the drum hit scan, exports
int8 and scores it on MoisesDB. Log: `F:\needle\logs\queue_n2a.log` and `logs\n2_a.log`.

What would count as success: clearly better than `b20_jam3_frontier_env` in the paired comparison on MoisesDB and on
the house hold-out set, at an int8 loss under 0.2 dB. The blueprint's estimate for this class of model is 6-7.5 dB on
MUSDB18-HQ against 4.5 dB today; that is an extrapolation from published models, not a promise.

## Open questions, in the order they should be answered

1. Does the family beat Needle 1 at all on the same data (`n2_a`)?
2. Does int8 hold for the GRUs of a trained model?
3. Which parts carry the gain: `--rnn 0` (no GRU), `--tdf 0 0 0 0 0`, `--no-cplx`, and the old STFT
   (`--n-fft 2048 --hop 512 --bins 1024` with a three-band grid) are one flag each.
4. Time resolution at the deepest level: `--tsteps 1` keeps 64 steps instead of 32 (about 1.5x the compute).
5. Per-stem refiners that run ahead of the playhead (blueprint, Candidate C), vocals first.
6. Real hardware: every time here is a Vela estimate with assumed SDRAM bandwidth (350 MB/s).
