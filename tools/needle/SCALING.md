# Needle scaling study: how to reach SDR ≥ 6.5 dB avg / ≥ 9.0 dB vocals on the RA8P1

Date: 2026-11-07. Scope: literature, Needle's own scaling curve, Vela feasibility (random weights), architectural levers, and a recommended ladder. No training was run. All Vela numbers come from `export_vela.py` (Vela 5.2.0, Ethos-U55-256 @ 500 MHz, the `RA8P1_SRAM` / `RA8P1_SDRAM` system configs in the script) and are labelled with their `build/` directory. Every untrained SDR figure is labelled **[extrapolation]**; figures from papers carry a URL. The product gate used throughout: MUSDB18-HQ test SDR (museval median-of-frames, median-of-tracks) ≥ 6.5 avg and ≥ 9.0 vocals, Vela RTF ≤ 0.4, and since two decks may separate at once, **≤ 0.2 per model**. (The README's "section" use case is stricter: 3 s for 16 bars ≈ 30 s of audio at 128 BPM → RTF ≤ 0.1 per deck.)

Hardware facts used (Renesas product page, https://www.renesas.com/en/products/ra8p1): Cortex-M85 1 GHz, Ethos-U55 256 GOPS @ 500 MHz, **0.5/1 MB MRAM**, 4/8 MB flash options, **2 MB SRAM including TCM and 64 KB caches** (so usable arena is below 2 MiB), octal-SPI XIP, 32/16-bit SDRAM interface. Consequence: any model with more than ~0.9 MB of int8 weights cannot live in MRAM; weights go to octal flash (XIP) or SDRAM regardless of where the arena is.

---

## 1. Literature: small and medium music source separation models

SDR in dB on the MUSDB18 test set (50 songs, median SDR; "HQ" where the paper says so). "Vela" = whether the inference graph can be built from operators Vela 5.2.0 maps to the U55 (supported-ops list: https://gitlab.arm.com/artificial-intelligence/ethos-u/ethos-u-vela/-/raw/5.2.0/SUPPORTED_OPS.md). The U55 list has no bidirectional LSTM, no attention/BATCH_MATMUL with dynamic weights beyond int8, no data-dependent normalisation (BatchNorm folds; GroupNorm/LayerNorm/gLN do not); UNIDIRECTIONAL_SEQUENCE_LSTM, FULLY_CONNECTED, TRANSPOSE (NCWH-type permutations for int8/int16), TRANSPOSE_CONV (stride 1×1 or 2×2), DEPTHWISE_CONV_2D, RESIZE_NEAREST_NEIGHBOR (equal H/W factors), ELU/GELU/PRELU, MUL, ADD, CONCATENATION, SLICE/STRIDED_SLICE are on it.

| Model (family) | Train data | Vocals | Drums | Bass | Other | **Avg** | Params | MACs | Context | Vela-mappable? | Source |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **Needle b16 f32** (spectrogram U-Net, magnitude masks) | MUSDB-HQ | 5.21 | 5.07 | 3.63 | 3.34 | **4.31** | 0.50 M | 1.18 GMAC/s | 0.37 s | yes (0 CPU ops) | README (this repo) |
| **Needle b16 f64** | MUSDB-HQ | 5.39 | 5.25 | 4.00 | 3.27 | **4.48** | 0.50 M | 1.18 GMAC/s | 0.74 s | yes | README |
| Needle b16 f32 + real stems | +545 songs | 5.22 | 5.10 | 3.97 | 3.31 | **4.40** | 0.50 M | 1.18 GMAC/s | 0.37 s | yes | README |
| Spleeter 4stems, **trained on MUSDB only**, MWF (U-Net, 6 levels 5×5, Jansson-style) | MUSDB | 5.10 | 5.15 | 4.27 | 3.21 | **4.43** | ≈9.8 M/stem ≈ 39 M [computed from `unet.py` channel list 16-512, 5×5 kernels: INFERENCE] | n/a | 11.9 s (T=512, hop 1024, n_fft 4096, F=1024 bins ≤ 11 kHz) | **yes** (CONV_2D s2, TRANSPOSE_CONV s2, ELU int8, CONCAT, LOGISTIC, MUL) | https://github.com/deezer/spleeter/wiki/Separation-Performances ; https://raw.githubusercontent.com/deezer/spleeter/master/spleeter/model/functions/unet.py ; https://raw.githubusercontent.com/deezer/spleeter/master/spleeter/resources/4stems.json |
| Spleeter 4stems pretrained, mask only | ≈24k songs | 6.55 | 5.93 | 5.10 | 4.24 | **5.46** | same | n/a | 11.9 s | yes | Spleeter wiki (above) |
| Spleeter 4stems pretrained, MWF | ≈24k songs | 6.86 | 6.71 | 5.51 | 4.55 | **5.91** | same | n/a | 11.9 s | yes (MWF on CPU) | Spleeter wiki; Demucs Table 1 https://arxiv.org/pdf/1911.13254 |
| Open-Unmix UMX / UMXHQ (3-layer BLSTM) | MUSDB / HQ | 6.32 / 6.25 | 5.73 / 6.04 | 5.23 / 5.07 | 4.02 / 4.28 | **5.33 / 5.41** | 8.89 M/target ≈ 35.6 M | n/a | whole track | no (BLSTM; causal uni-LSTM costs 0.5–0.7 dB, see X-UMX row) | https://raw.githubusercontent.com/sigsep/open-unmix-pytorch/master/README.md ; params: https://arxiv.org/pdf/1912.02591 Table 3 |
| X-UMX (CrossNet-Open-Unmix) | MUSDB | 6.61 | 6.47 | 5.43 | 4.64 | **5.79** | 36 M | n/a | 8 s | no (BLSTM) |  https://arxiv.org/pdf/2402.17701 Table 2 |
| X-UMX made causal, 23 ms window, no WF | MUSDB | 4.65 | 4.36 | 3.79 | 2.92 | **3.93** | 31 M | n/a | 23 ms | uni-LSTM is on the list | https://arxiv.org/pdf/2402.17701 Table 1–2 |
| Conv-TasNet (music, Défossez) | MUSDB | 6.43 | 6.02 | 6.20 | 4.27 | **5.73** | 9 M (42 MB fp32) | n/a | 2 s | no (global LayerNorm, PReLU ok) | https://arxiv.org/pdf/1911.13254 Table 1, 5; https://arxiv.org/pdf/2402.17701 |
| Demucs v2 (wave U-Net + BiLSTM) 64 ch / 48 ch / 32 ch | MUSDB | 6.84 | 6.86 | 7.01 | 4.42 | **6.28 / 6.20 / 6.08** | 1014 / 570 / 254 MB fp32 (≈ 265 / 150 / 65 M) | n/a | 10 s | no (BiLSTM) | https://arxiv.org/pdf/1911.13254 Table 1, 5 |
| D3Net (multi-dilated DenseNet, MWF) | MUSDB | 7.24 | 7.01 | 5.25 | 4.53 | **6.01** | not stated in arXiv v4 | n/a | 256 frames × hop 1024 = 5.9 s | in principle (dilated CONV_2D, avg-pool, TRANSPOSE_CONV 2×2, CONCAT, band split via SLICE) | https://arxiv.org/pdf/2010.01733 Table 1–2 |
| D3Net + 1,500 songs | +1.5k | 7.80 | 7.36 | 6.20 | 5.37 | **6.68** | — | — | 5.9 s | in principle | same, Table 3 |
| TFC-TDF-U-Net v1 **small** (vocals only, complex in/out, 7 blocks) | MUSDB | **7.07** (Mag-only variant: 6.43) | — | — | — | — | **0.99 M** | n/a | 128 frames × hop 1024 = 3 s | yes except TDF needs TRANSPOSE+FULLY_CONNECTED (on the list; unverified with Vela) | https://arxiv.org/pdf/1912.02591 Tables 2–4 |
| TFC-TDF-U-Net v1 large (n_fft 4096, 9 blocks) | MUSDB | 7.98 (Mag: 7.24) | — | — | — | — | 2.24 M | n/a | 3 s | as above | same |
| TFC-only U-Net v1 (dense 3×3 convs, no TDF) | MUSDB | 6.89 | — | — | — | — | 1.56 M | n/a | 3 s | **yes** | same, Table 2 |
| TFC-TDF-U-Net v2 alone (KUIELab, per-source, multiplicative skips) | MUSDB-HQ | 8.81 | 6.52 | 7.65 | 5.70 | **7.17** | 7.4 M × 4 | n/a | 256 frames × hop 1024 = 5.9 s; n_fft 6144/4096/16384/8192, 2048 bins kept | **yes** (conv + TDF + MUL) | https://arxiv.org/pdf/2111.12203 ; params: https://arxiv.org/pdf/2309.08684 Table 1 |
| v2 + Mixer (1×1 conv over stacked estimates) | MUSDB-HQ | 8.91 | 7.07 | 7.33 | 5.81 | **7.28** | +tiny | n/a | 5.9 s | yes | https://arxiv.org/pdf/2111.12203 |
| KUIELab-MDX-Net (v2 + Mixer + Demucs blend) | MUSDB-HQ | 9.00 | 7.33 | 7.86 | 5.95 | **7.54** | 7.4 M × 4 + Demucs | n/a | 5.9 s | no (Demucs blend) | same; https://arxiv.org/pdf/2211.08553 Table 1 |
| ResUNetDecouple+ (complex mask ResUNet) | MUSDB | 8.98 | 6.62 | 6.04 | 5.29 | **6.73** | 102 M × 4 | n/a | — | conv-only but huge | https://arxiv.org/pdf/2401.13276 Table 2; params https://arxiv.org/pdf/2309.08684 |
| Hybrid Demucs (v3) | MUSDB-HQ | 8.35 | 8.12 | 8.43 | 5.65 | **7.64** | 83.6 M (bag) | n/a | 7.8 s | no (BiLSTM/attention) | https://arxiv.org/pdf/2211.08553 Table 1 |
| HT Demucs | MUSDB-HQ | 7.93 | 7.94 | 8.48 | 5.72 | **7.52** | 42 M | CPU RTF 1.02–1.77 | 7.8 s | no (transformer) | https://arxiv.org/pdf/2211.08553 Tables 1–2 |
| HT Demucs + 800 songs, fine-tuned per source (the teacher class) | +800 | 9.20 | 10.08 | 10.39 | 6.32 | **9.00** | 42 M | — | 7.8 s | no | same |
| BSRNN (band-split, dual-path BLSTM) | MUSDB-HQ | 10.01 | 9.01 | 7.22 | 6.70 | **8.24** | 37.6 M × 4 | n/a | — | no (BLSTM, MLP norms) | https://arxiv.org/pdf/2211.08553 ; params https://arxiv.org/pdf/2309.08684 |
| DTTNet (TFC-TDF v3 enc/dec + dual-path BLSTM latent) | MUSDB-HQ | **10.12** | 7.74 | 7.45 | 6.92 | **8.06** | **5.0 M × 4** | 2.4 s GPU per 3-min song (RTX 3090) | 6 s | no (BLSTM, GroupNorm) | https://arxiv.org/pdf/2309.08684 Tables 1, 3 |
| TFC-TDF-UNet v3 (residual conv + TDF, no explicit time model) | MUSDB-HQ | 9.59 | 8.44 | 8.45 | 6.86 | **8.34** | 70 M | n/a | — | in principle (conv + TDF) but 70 M | https://arxiv.org/pdf/2309.08684 Table 1 |
| SCNet (sparse band compression + dual-path BLSTM) | MUSDB-HQ | 9.89 | 10.51 | 8.82 | 6.76 | **9.00** | 10.08 M | CPU RTF 0.669 s (single Xeon core) | 11 s | no (BiLSTM, GroupNorm) | https://arxiv.org/pdf/2401.13276 Tables 1–2 |
| SCNet ablation: same net **without** sparse compression | MUSDB-HQ | 7.45 | 7.34 | 5.48 | 4.54 | **6.20** (vs 8.52) | — | — | — | — | same, Table 4 |
| HS-TasNet (real-time, 23 ms latency, uni-LSTM hybrid) / -Small | MUSDB-HQ | 5.13 / 5.21 | 5.22 / 4.89 | 4.59 / 4.42 | 3.64 / 3.42 | **4.65 / 4.48** | 42 M / 16 M | 4 ms per 23 ms on 4 cores | 23 ms | uni-LSTM is on the list (unverified) | https://arxiv.org/pdf/2402.17701 Table 2 |
| HS-TasNet / -Small + 150 songs | +150 | 5.97 / 5.51 | 6.34 / 5.75 | 5.62 / 4.93 | 4.28 / 3.86 | **5.55 / 5.01** | 42 / 16 M | — | 23 ms | — | same |
| Wave-U-Net | MUSDB | 3.25 | 4.22 | 3.21 | 2.25 | **3.23** | — | — | — | yes in principle (1-D convs) | https://arxiv.org/pdf/1911.13254 Table 1 |

Reading of the table:

- **Smallest published designs at or above the gate.** ≥ 6.5 avg: TFC-TDF-U-Net v2 (7.17 avg, 7.4 M params per source, four models) is the smallest fully-convolutional one; D3Net reaches 6.68 only with 1,500 extra songs; DTTNet (5 M × 4) and SCNet (10 M, single model) are smaller per dB but rely on bidirectional LSTMs. ≥ 9.0 vocals: DTTNet 10.12 (5 M, BLSTM), SCNet 9.89 (10 M, BLSTM), BSRNN 10.01, TFC-TDF-UNet v3 9.59 (70 M, conv+TDF), KUIELab-MDX-Net 9.00 (needs the Demucs blend). **No published model built only from U55-mappable operators reaches 9.0 dB vocals below ~70 M parameters; the closest is TFC-TDF-U-Net v2 at 8.81 dB with 7.4 M parameters, complex spectrogram I/O, n_fft 6144 and 5.9 s of context.**
- **Int8 inference memory of that class [INFERENCE from the published shapes].** TFC-TDF-U-Net v2 vocals model: 7.4 MB weights; first-level feature maps are 2048 bins × 256 frames × 32 ch = 16 MiB each in int8, so the arena would be some 40–50 MiB; compute, counting 3×3 convs with 32·k channels per level over 11 blocks, is ≈ 95 GMAC per 5.9 s chunk ≈ 16 GMAC/s per source. At the effective U55 throughputs measured below (≈ 60–105 GMAC/s with the arena in SRAM, ≈ 30–75 GMAC/s from SDRAM) that is RTF ≈ 0.15–0.5 for the vocals model alone, i.e. 4 such models are 3–10× over the two-deck budget. The gate-class models therefore have to be re-engineered for compute, not just quantised.
- **The plain magnitude-mask U-Net family plateaus.** Spleeter's 39 M-parameter U-Net trained on MUSDB only scores 4.43 avg — the same as Needle's 0.5 M model (4.31–4.48). With ~24k training songs the identical network reaches 5.91 (MWF) / 5.46 (mask). Data moved it 1.5 dB; width did essentially nothing on 86 songs. This is the single most important fact for Section 2.
- **Complex spectrogram in/out is worth ≈ +0.6–0.7 dB vocals at equal parameters** (TFC-TDF v1 Table 4: 7.07 vs 6.43 at 0.99 M; 7.98 vs 7.24 at 2.24 M). TDF blocks add +0.18 while cutting parameters (7.07 @ 0.99 M vs 6.89 @ 1.56 M).

---

## 2. Needle's own scaling curve (all MUSDB18-HQ test, float, from README)

| Run | Params | Context | Avg | Vocals |
|---|---|---|---|---|
| b12 f32 | 282 k | 0.37 s | 4.06 | 4.83 |
| b16 f32 | 501 k | 0.37 s | 4.31 | 5.21 |
| b16 f64 | 501 k | 0.74 s | 4.48 (paired +0.31 [+0.10, +0.62]) | 5.39 |
| b16 f32 + real stems | 501 k | 0.37 s | 4.40 (paired +0.37 [+0.10, +0.78]) | 5.22 |

Slopes (each from a single interval, so ±50 % is realistic):

- **Width:** params ×1.78 (0.83 doublings) → +0.25 dB avg, +0.38 vocals ⇒ **≈ 0.30 dB avg / 0.45 dB vocals per doubling of parameters**.
- **Context:** 0.37 → 0.74 s (one doubling) → **+0.31 dB avg, +0.18 vocals**; helps pop/rock, not MoisesDB/house (README).
- **Data:** +545 real-stem songs → +0.37 avg on MUSDB, +0.46 MoisesDB, +1.9 house.

Extrapolations **[extrapolation; log-linear in parameters and context, which is optimistic because the Spleeter data point shows saturation]:**

- Width alone, MUSDB-only data, to 6.5 avg from b16 f64: (6.5 − 4.48) / 0.30 ≈ 6.7 doublings ≈ **×100 params ≈ 50 M**. Spleeter at 39 M and 11.9 s context on MUSDB-only sits at 4.43, far below that line, so **this path does not exist**; the curve bends over well before 6.5 on 86 training songs.
- Width + context inside this family, with the data recipe already in the repo (real stems + teacher stems): b16 → b48 (3.2 doublings, assume 0.2–0.3 dB each with saturation: +0.6–1.0), f64 → f128 (+0.15–0.3), data (+0.4–0.9, between Needle's measured +0.37 and D3Net's +0.67 for 1,500 songs) ⇒ **≈ 5.6–6.7 avg**; vocals from 5.39 with 0.3–0.45/doubling ⇒ **≈ 6.6–7.4**. The avg gate is at the edge of this range; **the vocals gate (9.0) is 1.6–2.4 dB beyond it**, consistent with the literature: 9 dB vocals has only been reached with complex I/O, per-source models/heads, band-aware frequency modelling and ≥ 3 s context.
- What base/frames the family would need for 6.5 avg *if* the slopes held: base 48–64 at frames 128 with the full data recipe (≈ 4.5–8 M params). This is where the Vela ladder below concentrates.

### 2b. Measured width series, 9 November 2026 (supersedes the slope above)

The b16/b20/b32 runs at frames 128 with the identical v3 recipe and data (`queue_v6`/`queue_v7`) are the clean width
experiment Section 2 asked for. MUSDB18-HQ test, float (`results/b{16,20,32}_f128_v3_test.json`):

| Run | Params | MUSDB avg | MUSDB vocals | MoisesDB avg | Ibiza v2 avg | int8 avg |
|---|---|---|---|---|---|---|
| b16 f128 v3 | 501 k | 4.40 | 5.16 | 5.76 | 7.72 | 4.39 |
| b20 f128 v3 | 782 k | 4.56 | 5.27 | 5.83 | 8.19 | 4.56 |
| b32 f128 v3 | 1 997 k | 4.61 | 5.34 | 5.89 | 8.48 | 4.60 |

- **The width slope is ≈ 0.10 dB avg / 0.09 dB vocals per doubling of parameters**, not 0.30 / 0.45: two doublings
  (501 k → 2.0 M) bought +0.21 avg and +0.18 vocals, and the second doubling (782 k → 2.0 M, +0.05 avg) is already
  flat. Section 2's extrapolations are therefore optimistic by about 3×, and the "base 48–64 reaches 6.5" line in
  Section 2/Step 3 is dead: at 0.10 dB/doubling, 6.5 avg from b32 would need ≈ 19 doublings.
- Frames 128 did not rescue it either: b16 f64 scored 4.48 and b16 f128 4.40, i.e. the context gain measured at
  f32 → f64 does not continue.
- **In-domain data is where the gain is.** The same three nets on the Ibiza/house test set rise 7.72 → 8.48 (+0.76
  over the same two doublings, 3.6× the MUSDB slope), and the earlier data step was +1.9 dB on house. The product
  metric moves with data, the MUSDB gate barely moves with anything.
- **int8 is free**: ≤ 0.01 dB at every width. Quantisation is not what stands between Needle and the gate.
- Consequence for the ladder: skip further width. The remaining candidates for the 6.5/9.0 gate are the
  architecture levers (complex spectrogram in/out, deep-filter taps — both already implemented in `needle.py`
  as of `2b99f24` — TDF blocks, band compression, vocals-first cascade) and more teacher data; width is spent.

---

## 3. Vela feasibility ladder (random weights; Vela 5.2.0, Ethos-U55-256 @ 500 MHz)

Columns: int8 arena = `sram_memory_used` in the `RA8P1_SRAM` config (feature maps; the same arena lands in SDRAM in the `RA8P1_SDRAM` config); weights = `off_chip_flash_memory_used`; "SRAM" = arena in on-chip SRAM, weights on a 1 GB/s port (MRAM/XIP flash estimate); "SDRAM" = arena in SDRAM at the script's 350 MB/s estimate, weights 1 GB/s; "all-SDRAM" = arena **and** weights at SDRAM speed (extra config run on the same tflite, results in `build/extra_vela_configs.json`). Channel lists are `base × mult`. `bins 512/256` rows keep `n_fft` 2048 but feed only the first 512/256 bins: they are **cost proxies for a band-compressed front end** (Section 4.4), not quality candidates. Every run reports **0 CPU operators**. Rows with the default `mult=(1,2,3,4,6)` were produced by `export_vela.py --base B --frames F`; the `m…` and `bins…` rows by the same script with `NeedleConfig` partially applied (mult / bins), no source changes.

| build dir | channels per level | bins | frames (s) | params | GMAC/s | int8 arena MiB | int8 weights MiB | int8 SRAM ms / RTF | int8 SDRAM ms / RTF | int8 all-SDRAM RTF | int16 arena MiB | int16 SRAM RTF | int16 SDRAM RTF | CPU ops |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `b16_f32` (trained) | 16/32/48/64/96 | 1024 | 32 (0.37) | 0.50 M | 1.18 | 1.5 | 0.44 | 7.1 / 0.019 | 14.2 / 0.038 | 0.039 | 3.0 | 0.035 | 0.091 | 0 |
| `random_f128` | 16/32/48/64/96 | 1024 | 128 (1.49) | 0.50 M | 1.18 | 6.0 | 0.51 | 27.9 / 0.019 | 57.8 / 0.039 | — | 12.5 | 0.037 | 0.102 | 0 |
| `random_b20_f128` | 20/40/60/80/120 | 1024 | 128 (1.49) | 0.78 M | 1.83 | 8.0 | 0.86 | 49.2 / 0.033 | 97.4 / 0.066 | — | 16.0 | 0.065 | 0.156 | 0 |
| `random_b16m1-2-3-4-6-8_f64` (6 levels) | 16/32/48/64/96/128 | 1024 | 64 (0.74) | 1.04 M | 1.21 | 3.0 | 1.02 | 14.3 / 0.019 | 29.0 / 0.039 | — | 6.2 | 0.038 | 0.102 | 0 |
| `random_b24_f32` | 24/48/72/96/144 | 1024 | 32 (0.37) | 1.12 M | 2.62 | 2.0 | 1.20 | 13.5 / 0.036 | 24.0 / 0.065 | — | 4.0 | 0.072 | 0.164 | 0 |
| `random_b24_f64` | 24/48/72/96/144 | 1024 | 64 (0.74) | 1.12 M | 2.62 | 4.0 | 1.20 | 26.7 / 0.036 | 48.2 / 0.065 | 0.074 | 8.0 | 0.072 | 0.167 | 0 |
| `random_b24_f128` | 24/48/72/96/144 | 1024 | 128 (1.49) | 1.12 M | 2.62 | 8.0 | 1.20 | 53.4 / 0.036 | 96.6 / 0.065 | — | 16.0 | 0.072 | 0.170 | 0 |
| `random_b24bins512_f64` (proxy) | 24/48/72/96/144 | 512 | 64 (0.74) | 1.12 M | 1.31 | 2.0 | 1.20 | 13.5 / 0.018 | 24.3 / 0.033 | — | 4.0 | 0.036 | 0.083 | 0 |
| `random_b8m1-4-8-16-32_f32` (thin top) | 8/32/64/128/256 | 1024 | 32 (0.37) | 1.96 M | 1.11 | **1.0** | 1.88 | 5.7 / 0.015 | 14.3 / 0.038 | — | 2.0 | 0.033 | 0.101 | 0 |
| `random_b8m1-4-8-16-32_f64` | 8/32/64/128/256 | 1024 | 64 (0.74) | 1.96 M | 1.11 | 2.0 | 1.88 | 10.9 / 0.015 | 29.5 / 0.040 | 0.045 | 4.0 | 0.031 | 0.103 | 0 |
| `random_b8m1-4-8-16-32_f128` | 8/32/64/128/256 | 1024 | 128 (1.49) | 1.96 M | 1.11 | 4.0 | 1.88 | 21.7 / 0.015 | 61.5 / 0.041 | — | 8.0 | 0.031 | 0.105 | 0 |
| `random_b16m1-2-4-8-16_f32` (deep) | 16/32/64/128/256 | 1024 | 32 (0.37) | 1.97 M | 1.62 | **1.5** | 1.89 | 8.5 / 0.023 | 16.0 / 0.043 | — | 3.1 | 0.046 | 0.112 | 0 |
| `random_b16m1-2-4-8-16_f64` | 16/32/64/128/256 | 1024 | 64 (0.74) | 1.97 M | 1.62 | 3.0 | 1.89 | 16.5 / 0.022 | 32.9 / 0.044 | 0.049 | 6.2 | 0.045 | 0.114 | 0 |
| `random_b32_f32` | 32/64/96/128/192 | 1024 | 32 (0.37) | 2.00 M | 4.63 | 2.5 | 1.92 | 19.1 / 0.051 | 34.8 / 0.094 | — | 5.1 | 0.108 | 0.239 | 0 |
| `random_b32_f64` | 32/64/96/128/192 | 1024 | 64 (0.74) | 2.00 M | 4.63 | 5.0 | 1.92 | 37.9 / 0.051 | 70.0 / 0.094 | 0.104 | 10.2 | 0.107 | 0.247 | 0 |
| `random_b32_f128` | 32/64/96/128/192 | 1024 | 128 (1.49) | 2.00 M | 4.63 | 10.0 | 1.92 | 75.5 / 0.051 | 143.5 / 0.097 | — | 20.5 | 0.107 | 0.251 | 0 |
| `random_b32bins512_f64` (proxy) | 32/64/96/128/192 | 512 | 64 (0.74) | 2.00 M | 2.31 | 2.5 | 1.92 | 19.1 / 0.026 | 35.2 / 0.047 | 0.053 | 5.1 | 0.054 | 0.123 | 0 |
| `random_b32bins256_f64` (proxy) | 32/64/96/128/192 | 256 | 64 (0.74) | 2.00 M | 1.16 | 1.3 | 1.92 | 9.8 / 0.013 | 18.2 / 0.024 | — | 2.6 | 0.027 | 0.062 | 0 |
| `random_b12m1-3-6-12-24_f32` (thin top) | 12/36/72/144/288 | 1024 | 32 (0.37) | 2.48 M | 1.58 | **1.0** | 2.60 | 9.6 / 0.026 | 19.0 / 0.051 | — | 2.0 | 0.051 | 0.118 | 0 |
| `random_b12m1-3-6-12-24_f64` | 12/36/72/144/288 | 1024 | 64 (0.74) | 2.48 M | 1.58 | 2.0 | 2.60 | 18.4 / 0.025 | 37.5 / 0.050 | — | 4.0 | 0.049 | 0.122 | 0 |
| `random_b8m2-6-12-24-48_f32` | 16/48/96/192/384 | 1024 | 32 (0.37) | 4.40 M | 2.52 | **1.5** | 4.24 | 11.4 / 0.031 | 21.0 / 0.057 | — | 3.1 | 0.064 | 0.145 | 0 |
| `random_b24m1-2-4-8-16_f64` | 24/48/96/192/384 | 1024 | 64 (0.74) | 4.43 M | 3.61 | 4.0 | 4.27 | 32.4 / 0.044 | 54.1 / 0.073 | 0.098 | 8.0 | 0.088 | 0.191 | 0 |
| `random_b48_f64` | 48/96/144/192/288 | 1024 | 64 (0.74) | 4.49 M | 10.35 | 7.0 | 4.46 | 79.9 / 0.108 | 104.2 / 0.140 | 0.232 | 14.2 | 0.213 | 0.454 | 0 |
| `random_b48_f128` | 48/96/144/192/288 | 1024 | 128 (1.49) | 4.49 M | 10.35 | 14.0 | 4.47 | 159.5 / 0.107 | 209.4 / 0.141 | — | 28.5 | 0.213 | 0.466 | 0 |
| `random_b48bins512_f64` (proxy) | 48/96/144/192/288 | 512 | 64 (0.74) | 4.49 M | 5.18 | 3.5 | 4.47 | 40.4 / 0.054 | 52.5 / 0.071 | 0.118 | 7.1 | 0.107 | 0.227 | 0 |
| `random_b48bins512_f128` (proxy) | 48/96/144/192/288 | 512 | 128 (1.49) | 4.49 M | 5.18 | 7.0 | 4.46 | 79.9 / 0.054 | 106.1 / 0.071 | — | 14.2 | 0.107 | 0.233 | 0 |
| `random_b32m1-2-4-8-16_f64` | 32/64/128/256/512 | 1024 | 64 (0.74) | 7.87 M | 6.39 | 5.0 | 7.53 | 48.5 / 0.065 | 83.1 / 0.112 | 0.143 | 10.2 | 0.137 | 0.292 | 0 |
| `random_b16m1-2-4-8-16-32_f64` (6 levels) | 16/32/64/128/256/512 | 1024 | 64 (0.74) | 7.87 M | 1.90 | 3.0 | 7.53 | 19.5 / 0.026 | 36.0 / 0.049 | — | 6.2 | 0.051 | 0.121 | 0 |
| `random_b64_f64` | 64/128/192/256/384 | 1024 | 64 (0.74) | 7.98 M | 18.35 | 9.1 | 7.64 | 130.0 / 0.175 | 205.5 / 0.277 | 0.355 | 18.2 | 0.356 | 0.722 | 0 |
| `random_b64_f128` | 64/128/192/256/384 | 1024 | 128 (1.49) | 7.98 M | 18.35 | 18.1 | 7.64 | 259.7 / 0.175 | 417.0 / 0.281 | — | 36.5 | 0.355 | 0.743 | 0 |
| `random_b64bins512_f64` (proxy) | 64/128/192/256/384 | 512 | 64 (0.74) | 7.98 M | 9.17 | 4.6 | 7.64 | 65.7 / 0.088 | 103.3 / 0.139 | 0.180 | 9.1 | 0.178 | 0.361 | 0 |

Weight placement (extra configs, `build/extra_vela_configs.json`): moving the weights from the 1 GB/s port to SDRAM speed (350 MB/s) or octal-SPI speed (400 MB/s assumed) changes inference time by < 1 % for every candidate (e.g. b48 f64: 79.9 → 80.6 ms); Vela streams weights once per chunk. Putting the **arena** in SDRAM costs 1.3–2× (small nets) to 1.6× (b64); arena *and* weights in SDRAM ≈ 2×.

Regularities (useful for designs not in the table):

- **Arena (int8) ≈ bins × frames × (2·c₁ + 16) bytes**, where c₁ is the channel count at full resolution (`base × mult[0]`): the peak is the full-resolution concat (c₁+2) plus the following conv output (c₁). Depth and width of the lower levels add nothing measurable (compare `b16m1-2-4-8-16-32` 7.9 M params, 3.0 MiB vs `b16` 0.5 M, 3.0 MiB at f64). int16 doubles it. So **memory is bought back by thinning the top level or compressing frequency, never by cutting depth**.
- **Effective throughput:** 62 GMAC/s (base 16) → 73 (b24) → 91 (b32) → 96 (b48) → 105 (b64) with the arena in SRAM; 31 → 40 → 49 → 74 → 66 GMAC/s from SDRAM. Peak is 128 GMAC/s (256 MACs × 500 MHz). RTF ≈ model GMAC/s ÷ effective throughput.
- **Weights 1 MB MRAM limit:** only base ≤ 20 (default mult) fits in the 1 MB MRAM option; everything else needs XIP flash or SDRAM for weights, which Vela says is free in time.

Fit classes (int8, two-deck budget RTF ≤ 0.2 per model, RA8P1 2 MB SRAM shared with firmware so ≈ 1.5 MiB arena is the practical ceiling):

| Class | Candidates | Notes |
|---|---|---|
| (a) **2 MB SRAM only** (arena ≤ 1.5 MiB, weights ≤ 1 MB MRAM) | `b16_f32` (1.5 MiB / 0.44 MB) and b12/b8 f32 variants with ≤ 1 MB of weights | nothing wider at 1024 bins |
| (b) **arena in SRAM, weights in XIP flash / SDRAM** | `b8m1-4-8-16-32_f32` (1.0 MiB, 1.96 M, RTF 0.015), `b12m1-3-6-12-24_f32` (1.0 MiB, 2.48 M, 0.026), `b16m1-2-4-8-16_f32` (1.5 MiB, 1.97 M, 0.023), `b8m2-6-12-24-48_f32` (1.5 MiB, 4.40 M, 0.031), `b32bins256_f64` (1.3 MiB) | 4–9× the parameters of today's model at the same arena; all at frames 32 (0.37 s) |
| (c) **everything in SDRAM** (128 MB) | all rows; two-deck RTF ≤ 0.2 holds for every int8 model except b64 at 1024 bins (0.277–0.355) and b48 f64 with weights also in SDRAM (0.232); int16 fails for base ≥ 32 | b48 f128 int8: 14 MiB arena, RTF 0.14 (weights on the fast port) |

---

## 4. Architectural levers beyond width

Each lever: expected gain (source), cost in memory / RTF (Vela table or regularity above), Vela compatibility.

1. **Longer frames / context.** Needle: +0.31 avg for 0.37 → 0.74 s (README, paired CI [+0.10, +0.62]); HT Demucs: +0.53 avg for 3.4 → 7.8 s training segments (https://arxiv.org/pdf/2211.08553 Table 2); the ≥ 7 dB spectrogram models all use 3–12 s (TFC-TDF v1 3 s, KUIELab 5.9 s, SCNet 11 s, Spleeter 11.9 s). Cost: arena ∝ frames (b16: 1.5 → 3.0 → 6.0 MiB for f32/64/128), RTF unchanged, latency = chunk length (fine for the section use case, which already has ~1 s of context on each side). Vela: no change. Expect diminishing returns past ~3 s (frames 256) **[extrapolation]**.
2. **TFC-TDF blocks.** TDF = two time-distributed fully-connected layers along frequency (bottleneck F/bn) after the conv pair, with a residual. +0.18 dB vocals while cutting params from 1.56 M to 0.99 M (https://arxiv.org/pdf/1912.02591 Table 2); it is the ingredient that took KUIELab/DTTNet to 8.8–10 dB vocals. Cost: parameters 2·F_ℓ²/bn per block (level 1 at 1024 bins, bn 8: 262 k; all levels ≈ 350 k), negligible MACs, +1 feature map per level in the arena. Vela: expressible as TRANSPOSE (NHWC→NCWH is in the int8/int16 permitted list) → FULLY_CONNECTED (weights O,1,…,I; constant) → RELU → FULLY_CONNECTED → TRANSPOSE back → ADD; **unverified with Vela 5.2.0 — run `export_vela.py` on a TDF prototype before training** (risk: TRANSPOSE may be executed as a memory copy with low utilisation).
3. **Depthwise-separable convs.** DEPTHWISE_CONV_2D is on the U55 list (stride ≤ 3). No MSS paper reports a gain at equal parameters; it trades 5–8× fewer MACs for the same arena, and **[INFERENCE]** the U55's 256-MAC array is poorly utilised by depthwise kernels, so time drops much less than MACs. For Needle the small models are memory-bound anyway (62 GMAC/s effective at base 16). Low priority; only worth it if a base ≥ 64 design turns out compute-bound.
4. **Band-split / sparse frequency compression at the input.** SCNet's ablation: keeping the low band at full resolution and compressing mid/high bands gives 8.52 vs 6.20 avg for the same global compression (https://arxiv.org/pdf/2401.13276 Table 4); D3Net and Spleeter also cut or band-split the top. Cost: the `bins 512` proxies show exactly half the GMAC/s, arena and RTF of the 1024-bin model at equal parameters (b48 f64: 7.0 → 3.5 MiB, RTF 0.108 → 0.054; b64 f64: 0.175 → 0.088). A front end that keeps bins 0–255 (0–5.5 kHz) at full resolution and pools 256–1023 by 4 yields 448 rows ≈ the bins-512 proxy. Vela: SLICE/STRIDED_SLICE along H, CONV_2D with stride (1..3, or > 3 with VALID padding), CONCATENATION along H — all on the list; the unequal upsampling at the output is **not** (RESIZE_NEAREST_NEIGHBOR needs equal H/W factors, TRANSPOSE_CONV 2×1 only for height-1 maps), so output the high-band masks at reduced frequency resolution and let the Cortex-M85 repeat them when it applies the masks (it already owns that step). Needs a `needle.py` change; worth it because it doubles the affordable width for free.
5. **Multi-stage / cascaded nets.** (i) KUIELab's Mixer (one 1×1 conv over the stacked mixture + four estimates): +0.11 avg (7.17 → 7.28, https://arxiv.org/pdf/2111.12203). Cost ≈ 0; Vela: CONCAT + CONV_2D 1×1. (ii) Blending with a second model family: v2 + Demucs +0.32 avg; not available on the U55. (iii) **Vocals-first cascade**: a dedicated vocals network (complex I/O, its own n_fft, 2–4× the capacity) followed by a 3-stem network on the residual (mixture − vocals). Evidence: per-source models/fine-tuning are standard in every ≥ 9 dB vocals system (KUIELab per-source n_fft 6144 for vocals; HT Demucs per-source fine-tuning +0.25 avg, https://arxiv.org/pdf/2211.08553 §5.5; BSRNN, DTTNet per source). Cost: the two models run sequentially on the one NPU, so RTFs add and the arena is the larger of the two; weights add. Vela: two tflite files, nothing new.
6. **Distillation from HTDemucs_ft on the DJ library (`distill.py`).** Same-architecture data gains in the literature: Spleeter 4.43 → 5.91 with ~24k songs (+1.5), D3Net +0.67 (1,500 songs), Demucs +0.51 (150 songs), HT Demucs +1.28 (800 songs); Needle +0.37 avg on MUSDB and +1.9 on house from 545 real-stem songs. Teacher stems cap the student near the teacher (8.98 avg / 9.21 vocals) and mostly help in-domain (house), less on the MUSDB test gate. Cost: zero at inference. This is the cheapest lever and is already implemented.
7. **QAT.** Not needed today: int8 costs 0.02 dB and every layer quantises at rounding level (README, `quant_debug.py`). Revisit only if complex I/O (signed, wide dynamic range; use a power-law compressed complex input computed on the M85 **[INFERENCE]**) or TDF layers (long FC sums) show > 0.2 dB int8 loss; int16 is the fallback but doubles arena and time (table).
8. **Complex spectrogram in/out (not in the brief, but the largest cheap gain).** +0.64 and +0.74 dB vocals at equal parameters (https://arxiv.org/pdf/1912.02591 Table 4). Cost: input 4 channels instead of 2, output 16 instead of 8, linear head instead of sigmoid; arena +≈ 10 %; the sum-to-mix mask constraint goes (or keep it via a complex mask). Vela: no new operators.
9. **Multichannel Wiener filter on the Cortex-M85.** Spleeter mask → MWF: vocals +0.31, drums +0.78, bass +0.41, other +0.31, **+0.45 avg** (Spleeter wiki); X-UMX with vs without WF +0.66 avg (https://arxiv.org/pdf/2402.17701 Table 1). Cost: CPU only (one EM iteration per TF bin, 4 sources × 2 channels); no training, no NPU change. Testable **now** on the existing b16 checkpoints with `norbert`.
10. **Test-time shifts.** Demucs: +0.23–0.3 dB for 10 shifts (https://arxiv.org/pdf/1911.13254 Table 4); `--shifts 2` already exists (README) at 2× NPU time and no extra memory.

---

## 5. Recommendation: a four-step ladder

Expected SDR ranges are **[extrapolation]** built from Section 2 slopes and Section 4 literature gains, discounted (the gains are not additive). "SRAM/SDRAM split" is for int8.

| Step | Name | Net | Params | Int8 memory | Vela RTF (build dir) | Expected MUSDB avg / vocals | Purpose |
|---|---|---|---|---|---|---|---|
| 1 | **S1 — fits SRAM, deeper** | base 16, mult (1,2,4,8,16), frames 32 (channels 16/32/64/128/256) | 1.97 M | arena 1.5 MiB **SRAM**; weights 1.9 MB **XIP flash or SDRAM** (exceeds 1 MB MRAM) | 0.023 SRAM / 0.043 SDRAM (`random_b16m1-2-4-8-16_f32`) | 4.5–4.8 / 5.5–6.0 | Shows whether depth-only parameters help at all (arena unchanged); drop-in for the current firmware memory map |
| 2 | **S2 — arena to SDRAM, 4× width** | base 32, mult (1,2,3,4,6), frames 64 | 2.00 M | arena 5.0 MiB **SDRAM**; weights 1.9 MB flash/SDRAM | 0.051 SRAM-speed / 0.094 SDRAM / 0.104 all-SDRAM (`random_b32_f64`) | 4.9–5.4 / 5.9–6.5 (with real stems + teacher: +0.4–0.9) | Second point on the width curve (4× params, 2 doublings), needed to confirm or refute the 0.3 dB/doubling slope before paying for b48/b64 |
| 3 | **S3 — band-compressed wide, complex I/O, TDF** | base 48, frames 128, low band 0–255 bins full-res + bins 256–1023 pooled ×4 (cost ≈ `bins 512` proxy), complex in/out, TDF at each level, Mixer, MWF on the M85 | ≈ 5 M (4.5 M + TDF) | arena ≈ 7–8 MiB SDRAM; weights ≈ 5 MB SDRAM | 0.054 SRAM-speed / 0.071 SDRAM / 0.12 all-SDRAM (`random_b48bins512_f128`), + ≈ 10 % for TDF/complex | 5.8–6.6 / 7.0–8.0 | First configuration with a real chance at the 6.5 avg gate; still one model, two-deck safe |
| 4 | **S4 — full SDRAM, vocals-first cascade** | vocals net: S3 topology at base 64 and frames 256 (3 s), own n_fft (4096–6144 on the M85); 3-stem net: S3 topology at base 32–48 on the residual | ≈ 9 + 4 M | arena ≈ 20 MiB SDRAM (max of the two); weights ≈ 13 MB SDRAM | ≈ 0.09 + 0.05 ≈ 0.14 SRAM-speed / ≈ 0.25 all-SDRAM (sum of `random_b64bins512_f64` and `random_b32bins512_f64` scaled) — two decks must alternate or the arena goes to SRAM-cached operation | 6.3–7.0 / 8.0–9.0 | The only ladder step whose published analogue (TFC-TDF-U-Net v2, 8.81 vocals at 7.4 M) approaches 9 dB vocals; 9.0 is not assured on MUSDB test |

Hard conclusions:

- **Memory steps are cheap, compute steps are not.** Vela puts the weight location at < 1 % of inference time and the arena location at ≈ 2×. Everything up to base 48 at 1024 bins (or base 64 band-compressed) stays under the two-deck RTF budget even fully in SDRAM; base 64 at 1024 bins does not.
- **Width alone will not reach the gate on MUSDB-only data** (Spleeter, 39 M params, 4.43). The gate needs, in order of evidence strength: more/distilled data (+0.4–1.5), complex spectrogram I/O (+0.6–0.7 vocals), a dedicated vocals model with its own STFT (+0.25 and the only route to 9 dB in the literature), band-aware frequency modelling (TDF / sparse compression), ≥ 1.5–3 s context, and MWF (+0.45) — all of which are Vela-expressible.
- **9.0 dB vocals on the MUSDB test set is beyond any published U55-mappable model under ~70 M parameters.** If the product metric can be the DJ library scored against the teacher (as `luke_test` already is), the vocals target becomes far more attainable through distillation; that is a product decision, not an engineering one, and should be made before S4 is built.

**First step to train next: S2 (`--base 32 --frames 64`, default mult) with the real-stems + teacher data recipe already planned**, because (i) it is one change in the proven family, so the paired test against b16 f64 cleanly measures the width slope at 2 doublings (the number every later decision hinges on); (ii) it is the smallest model that actually uses the newly allowed SDRAM (5 MiB arena) and whose Vela RTF (0.094–0.104 from SDRAM) clears the two-deck budget with margin; (iii) its 4.6 GMAC/s is 4× today's, so a 3080 Ti run is still an overnight job **[INFERENCE from 3 h for b16]**. In parallel and without training: (a) run MWF (`norbert`) on the existing b16 checkpoints to measure lever 9; (b) export a TDF + complex-I/O prototype of S3 with random weights through `export_vela.py` to confirm 0 CPU operators and the TRANSPOSE/FULLY_CONNECTED cost before any S3 training. If S2 gains < 0.3 dB over b16 f64, skip S3's width and go straight to complex I/O + band compression + vocals-first (the levers that moved the literature), since the family's width curve would then be confirmed flat.

---

### Artefacts produced
New Vela reports (random weights, int8 + int16, both system configs, 0 CPU operators each): `build/random_b24_f32`, `random_b24_f64`, `random_b32_f32`, `random_b32_f64`, `random_b32_f128`, `random_b48_f64`, `random_b48_f128`, `random_b64_f64`, `random_b64_f128`, `random_b8m1-4-8-16-32_f{32,64,128}`, `random_b12m1-3-6-12-24_f{32,64}`, `random_b16m1-2-4-8-16_f{32,64}`, `random_b16m1-2-3-4-6-8_f64`, `random_b16m1-2-4-8-16-32_f64`, `random_b24m1-2-4-8-16_f64`, `random_b32m1-2-4-8-16_f64`, `random_b8m2-6-12-24-48_f32`, `random_b{24,32,48,64}bins512_f64`, `random_b48bins512_f128`, `random_b32bins256_f64` (each `…/needle_b*_f*/report.json`), plus `build/extra_vela_configs.json` (weights-in-SDRAM / weights-in-OSPI / all-SDRAM timings for 12 int8 models) and stray `build/_log_*.txt` from the last batch. Pre-existing reports reused: `build/b16_f32`, `b16_f32_stems`, `random_f128`, `random_b20_f128`, `random_b24_f128`. No tracked file was modified.
