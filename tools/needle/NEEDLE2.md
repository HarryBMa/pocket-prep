# Needle 2: the band-split separator (started 12 Nov 2026)

Needle 1 (`needle.py`) is a plain convolutional U-Net. It stopped improving: doubling its size gave about +0.1 dB
(`SCALING.md` §2b), and it sits at 4.4-4.5 dB on MUSDB18-HQ like every other plain mask U-Net. The research
blueprint of 12 Nov 2026 (project doc `DECK001_research_blueprint_2026-11.md`) found that small separators which do
reach 7-9 dB all share three things Needle 1 lacks: a band-split frequency grid, a layer that sees the whole spectrum
at once, and a sequence model over time. Needle 2 (`needle2.py`, `NeedleConfig.arch = "bs"`) adds those, using only
what the Ethos-U55 runs.

**Result (13 Nov 2026).** The first full run, `n2_a`, beats Needle 1 trained on the same data with the same amount of
audio (`b20_jam3_frontier_env`) on every test set: +0.81 dB on MUSDB18-HQ (4.52 -> 5.19 dB), +1.24 dB on MoisesDB
(5.47 -> 6.46 dB, better on all 30 songs), and it loses 0.09 dB in int8 with 0 CPU operators. Its one weak spot is
the high-frequency attack of drum hits. Details under "First run" below. Speed and memory figures are Vela 5.2.0
estimates.

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
| Input, optional (M85) | `--sub 4` (13 Nov, on trial in `n2_d`): per block of 16 bins, the mixture's envelope in four 5.8-ms slots inside the frame's hop; 8 more channels per row |
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
| `hit_scan2.py` | the drum-hit checks of `hit_scan.py` on any reference set (real stems); `--compare` two models hit by hit |
| `oracle_hf.py` | ideal gains in each output format: what the format allows, without a network |
| `ckinfo.py` | which weights a checkpoint holds (trained or an averaged copy), its step and validation SDR |
| `queue_n2x.example.ps1` | one run from waiting for the GPU to the int8 score, with parameters (live copy `F:\needle\queue_n2x.ps1`) |
| `needle.py`, `train.py`, `chunked.py`, `eval_tflite.py` | extended: config fields, `--arch bs`, evaluation in slices, band inputs for int8; `train.py --ema`, `--init`, `--sub` (13 Nov) |

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

### Result (13 Nov 2026)

150,000 steps in 4.9 h (9.7 steps/s; Needle 1 needed 5.6 h for the same audio). Validation on the 14 MUSDB tracks:
best 5.56 dB at step 120,000, 5.51 at the end (Needle 1: best 4.60). It passed Needle 1's best after 25,000 steps.

SDR in dB as `eval.py` reports it (drums / bass / other / vocals in brackets); the last column is `paired.py`:
the mean difference per song with its 95 % bootstrap interval, and on how many songs Needle 2 is better.

| Test set | Needle 1 `b20_jam3_frontier_env` | Needle 2 `n2_a` | Needle 2 minus Needle 1, per song |
|---|---|---|---|
| MUSDB18-HQ test, 50 songs | 4.52 (5.64 / 3.96 / 3.44 / 5.04) | 5.19 (6.08 / 4.56 / 4.07 / 6.06) | +0.81 [+0.63, +0.96], 46/50 |
| MoisesDB test, 30 songs | 5.47 (6.08 / 6.44 / 3.36 / 6.01) | 6.46 (7.16 / 7.40 / 4.55 / 6.75) | +1.24 [+1.09, +1.39], 30/30 |
| NI stems, 6 songs | 4.49 (7.96 / 6.87 / 2.39 / 0.75) | 4.63 (9.03 / 7.57 / 1.68 / 0.24) | +0.39 [-0.17, +0.88], 4/6 |
| House hold-out, 50 songs, teacher stems | 6.07 (10.87 / 7.28 / 6.15 / -0.00) | 7.00 (12.30 / 8.69 / 7.00 / 0.01) | +1.06 [+0.76, +1.41], 47/50 |
| Ibiza, 43 songs, teacher stems | 8.47 (18.53 / 9.86 / 4.48 / 1.00) | 9.07 (19.88 / 10.51 / 5.25 / 0.63) | +0.76 [+0.51, +1.01], 36/43 |

Per stem, song by song: MUSDB drums +0.76, bass +0.66, other +0.61, vocals +1.19; MoisesDB +1.34, +1.11, +1.02,
+1.49. Every one of these intervals excludes 0. The NI set is too small to tell.

**int8** (post-training, calibrated on MUSDB training mixtures): 3,270 operators, 0 on the CPU, 250 ms per chunk
with the arena in SDRAM (RTF 0.084), arena 5.9 MiB, weights 2.78 MiB; TensorFlow copy against PyTorch 9e-6.
MoisesDB: 6.38 dB in int8 against 6.46 dB in float. Loss: drums 0.11, bass 0.03, other 0.24, vocals -0.03,
average 0.09 dB. The GRUs hold in int8; a 16-bit sub-graph is not needed.

**Drum hits** on the house hold-out set against the teacher's drums (`hit_scan.py`, 82,253 hits):

| | Needle 1 | Needle 2 |
|---|---|---|
| Level of a hit, median | -0.06 dB | -0.11 dB |
| Hits more than 10 dB too quiet | 1.9 % | 2.1 % |
| Hit-to-hit spread | 2.74 dB | 2.75 dB |
| The same inside chunk crossfades / elsewhere | 1.6 % / 2.0 % | 1.6 % / 2.2 % |
| 6-16 kHz attack more than 6 dB down | 1.0 % | 3.0 % |

### The weak spot: high-frequency attacks

Needle 2 keeps the level of drum hits as well as Needle 1, but it dulls the bright first 20 ms of more of them.

- House hold-out, against the teacher (`hit_scan2.py --compare` on `hit_scan.json`): attack more than 6 dB down on
  3.8 % of the hits that have high-frequency content, against 1.2 %; more than 10 dB down on 2.5 % against 0.6 %.
  At the hits Needle 2 dulls, Needle 1 is fine (median +0.18 dB). It is concentrated: the worst 5 of 50 tracks hold
  37 % of the dulled hits (45 % and 29 % of all hits in the two worst, where Needle 1 has 4.5 % and 2.2 %). It does
  not depend on the position in the chunk (3.9 % inside crossfades, 3.8 % elsewhere).
- MoisesDB test, against real drum stems (`hit_scan2.py`, 28,303 hits): 6.0 % against 4.9 % more than 6 dB down,
  2.9 % against 2.3 % more than 10 dB down; Needle 2 dulls more hits on 26 of 30 tracks (median per track 5.3 %
  against 2.9 %). The hit-to-hit spread is better (2.61 against 2.98 dB), the median level 0.3 dB lower.

So the effect is real, moderate on acoustic kits and large on a few dense electronic tracks. SDR does not show
it: on three of the five worst tracks Needle 2's drums still score 0.7-1.2 dB higher than Needle 1's (on the other
two 0.7-0.8 dB lower).

**It is not the output format** (`oracle_hf.py`). The first suspicion was time resolution at the output: a 93-ms
window (Needle 1: 46 ms) and one filter for the 4 or 16 bins of a row. The oracle test rebuilds the drums of the
five worst tracks with the ideal real gain per cell of each format:

| Format, ideal gains | Drums SDR | 6-16 kHz SDR | Attack > 6 dB down |
|---|---|---|---|
| Needle 2: 4096 / 1024, one gain per row | 14.33 dB | 18.42 dB | 2.9 % |
| 4096 / 1024, one gain per bin | 14.34 dB | 19.04 dB | 2.4 % |
| Needle 1: 2048 / 512, one gain per bin | 13.88 dB | 19.40 dB | 2.8 % |
| Split: 1024 / 256 above 5.5 kHz (5.8-ms frames) | 14.33 dB | 18.39 dB | 3.7 % |

With ideal gains Needle 2's format keeps the attacks as well as Needle 1's (2.9 % against 2.8 %), and a
short-window output above 5.5 kHz gains nothing. The trained `n2_a` dulls 10-45 % of the hits on these tracks where
the oracle in its own format dulls 0-9 %. So the network does not find these hits; it could pass them.

**What it probably cannot see.** In the magnitudes of a 93-ms frame a 20-ms hat is a small bump on whatever else
is bright. The frame still holds the timing, in the phases of neighbouring bins, but the first layer would have to
learn an inverse DFT to read it. `--sub 4` (`needle.subframe_envelope`) hands it over: for every block of 16 bins
(172 Hz) the envelope in four 5.8-ms slots inside the frame's hop, computed from the same STFT (16 complex
multiply-adds per slot on the M85, no second STFT), 8 extra input channels per row. On the RA8P1 it costs 1 ms
(251 ms per chunk, RTF 0.085, arena 5.9 MiB, 0 CPU operators; random weights). Whether it brings the hats back is
what `n2_d` tests. If `n2_b` (no GRU) does not dull them, the cause is the GRU's wide view instead.

## Second run: `n2_b` = `n2_a` without the GRU (`queue_n2b.example.ps1`, live copy `F:\needle\queue_n2b.ps1`)

`--rnn 0`, everything else and the number of steps as `n2_a`; starts when `queue_n2a` has finished. The GRU blocks are
73 ms of the 250 ms per chunk (RTF 0.084 against 0.060 without them), so the deck needs to know what they buy. The
queue prints the song-by-song difference `n2_a` minus `n2_b`. Started 13 Nov 09:02 (1.53 M parameters, 10.5 steps/s).

### Result (13 Nov 2026)

Validation: best 5.40 dB (`n2_a`: 5.56). SDR in dB, `n2_a` minus `n2_b` song by song:

| Test set | `n2_a` (GRU) | `n2_b` (no GRU) | What the GRU buys |
|---|---|---|---|
| MUSDB18-HQ test | 5.19 | 5.30 | +0.03 [-0.03, +0.10], 32/50 |
| MoisesDB test | 6.46 | 6.43 | +0.13 [+0.06, +0.21], 25/30 (drums +0.25) |
| NI stems | 4.63 | 4.65 | +0.05 [-0.42, +0.51] |
| House hold-out, teacher stems | 7.00 | 6.92 | +0.13 [+0.02, +0.23], 32/50 |
| Ibiza, teacher stems | 9.07 | 9.23 | +0.10 [-0.12, +0.32] |
| MoisesDB in int8 | 6.38 (loss 0.09) | 4.81 (loss 1.63) | |
| NPU time per chunk, SDRAM | 250 ms | 177 ms | |

In float the GRU blocks buy 0.0-0.15 dB for 73 ms. In int8 the model without them breaks: drums -0.51, bass
-1.36, other -2.78, vocals -1.85 dB (mean output error 0.0233 against 0.0084 for `n2_a`). Why is open; until it is
understood `n2_a`'s layout stays. If the cause can be fixed (calibration, a layer whose range explodes), the
no-GRU model is 30 % faster at the same float quality.

**Hats.** `n2_b` loses them too, so the GRU is not the cause: attack more than 6 dB down on 2.7 % of the house
hits (Needle 1 1.2 %, `n2_a` 3.8 %) and on 8.1 % of the MoisesDB hits (4.9 %, 6.0 %). `n2_d` (`--sub 4`) is the
right next test.

## Runs queued on 13 Nov: `n2_d` and `n2_c` (`queue_n2x.example.ps1`)

| Run | What changes against `n2_a` | Question |
|---|---|---|
| `n2_d` | `--sub 4` (the sub-frame envelope input), from scratch, 150,000 steps; `--ema 0.999` | Do the hats come back, at what price in SDR? |
| `n2_c` | starts from `n2_a`'s best weights (`--init`), 150,000 more steps, new schedule with peak lr 5e-4; `--ema 0.999 0.9999` | Does more training help? Do averaged weights help? |

**Averaged weights (`--ema`).** A copy of the weights follows the trained ones as an exponential moving average
(decay 0.999 = about the last 1,000 steps; warmed up so that the first steps count fully). It is never trained.
Every validation scores the trained weights and each copy; `best.pt` is the best of all, `best_raw.pt` the trained
weights' own best, so runs with and without `--ema` still compare like for like. Reason to expect a gain: v16's and
`n2_a`'s validation SDR move by up to 0.25 dB from one validation to the next. Checked on a 450-step test: all
three are scored and saved, `--resume` restores the copies.

The queue evaluates `best_raw.pt` on the five sets and compares it with `n2_a` song by song; if `best.pt` is an
averaged copy it is scored on MUSDB and MoisesDB as `<name>_ema`. `best.pt` goes to the two hit scans (house
hold-out against the teacher, MoisesDB against real stems, each with `hit_scan2.py --compare` against `n2_a`) and
to int8.

## Pre-flight checks (12 Nov 2026, late evening)

Two 1,000-step runs with one small recipe (MUSDB only, the same amount of audio per step, the same schedule, the
losses of the real runs), each sharing the GPU with the v16 training. Validation is the fast SDR on the 14 MUSDB
validation tracks.

| | Needle 1 (`b20`, frontier) | Needle 2 |
|---|---|---|
| after 500 steps | 1.08 dB | 2.06 dB |
| after 1,000 steps | 1.49 dB (drums 2.94, bass 2.43, other -0.26, vocals 0.84) | 2.32 dB (4.27, 3.05, 0.56, 1.40) |
| training speed | 6.4 steps/s at batch 8 x 1.5 s | 6.3 steps/s at batch 4 x 3 s |

This shows that Needle 2 trains, and learns faster at the start. It does not show where it ends up.

The 1,000-step Needle 2 model in int8 against float, on four MoisesDB test tracks: 2.86 dB average for both; drums
-0.21, bass -0.08, other +0.20, vocals +0.11 dB. So the GRUs do not break in int8; whether the drums lose 0.2 dB
in a fully trained model is checked on all 30 tracks by the queue.

The first of these test runs ran out of GPU memory at its validation: a whole song's input features were built in
one piece on the GPU. `chunked.separate_filtered` now builds them 2048 frames at a time (bit-identical stems).

Two more checks before the night run:

- The data really is v16's. A Needle 2 crop is 3.27 s with speed augmentation (1.64 s for Needle 1), and
  `StemBank` drops every track that is not longer than one crop. No track is that short, so nothing is dropped:
  545 real-stem tracks (16.7 h), 2,329 teacher excerpts (38.8 h), 86 MUSDB tracks (5.3 h), all already in the cache.
- `test_needle2.py --speed 4` reports 6.3 steps/s on the shared GPU, the speed the 1,000-step run really had, with
  1.6 GiB of GPU memory at batch 4. So the queue's speed probe picks a sensible number of steps.

## Open questions, in the order they should be answered

1. ~~Does the family beat Needle 1 at all on the same data (`n2_a`)?~~ Yes: +0.8 dB on MUSDB, +1.2 dB on MoisesDB.
2. ~~Does int8 hold for the GRUs of a trained model?~~ Yes: 0.09 dB lost on MoisesDB.
3. Why does Needle 2 lose more high-frequency attacks? Not the output format (oracle). Does the sub-frame envelope
   input bring them back (`n2_d`), or is it the GRU (`n2_b`'s hit scan)?
4. Does more training help, and do averaged weights (`n2_c`)?
5. Which parts carry the gain: `--rnn 0` (no GRU, `n2_b`), `--tdf 0 0 0 0 0`, `--no-cplx`, and the old STFT
   (`--n-fft 2048 --hop 512 --bins 1024` with a three-band grid) are one flag each.
6. Time resolution at the deepest level: `--tsteps 1` keeps 64 steps instead of 32 (about 1.5x the compute).
7. Per-stem refiners that run ahead of the playhead (blueprint, Candidate C), vocals first.
8. Real hardware: every time here is a Vela estimate with assumed SDRAM bandwidth (350 MB/s).
