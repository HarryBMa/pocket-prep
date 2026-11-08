# S2: the next training run (base 32, frames 64)

Why this run: `SCALING.md` §5. It measures the width slope at two parameter doublings inside the proven family, with
the data recipe that is known to help on MUSDB (real stems), before any wider or architecturally different model is
paid for. Numbers below are the committed results in `README.md` / `results/`.

## What has moved MUSDB test SDR so far

| Lever | Effect (MUSDB avg, paired where available) |
|---|---|
| real stems, `--stems-share 0.6` | +0.37 dB [+0.10, +0.78], 38/50 songs; MoisesDB +0.46, house +1.9 |
| frames 32 → 64 | +0.31 dB [+0.10, +0.62]; arena 1.5 → 3 MiB (SDRAM) |
| base 12 → 16 | +0.25 dB |
| teacher share 0.35–0.5 (luke) | **−0.2 to −0.4 dB on MUSDB** (b16_f32_kd 4.08, b16_f128_v3 4.11 at 50 k) — better in-domain only |
| width alone on MUSDB-only data | saturates (Spleeter 39 M params: 4.43) |

So: keep the teacher share small when MUSDB is the gate; combine data, context and width.

## Command

```powershell
# tools\needle, GPU free (the 3080 Ti has 12 GB), logs and cache on F: (C: is nearly full)
$stems = @('--stems','F:\needle\data\stems\moisesdb','--stems','F:\needle\data\stems\mshoxx',
           '--stems','F:\needle\data\stems\ni','--stems','F:\needle\data\stems\packs','--stems','F:\needle\data\stems\wavgrind')
$s2 = @('--base','32','--frames','64','--batch','32','--steps','300000',
        '--stems-share','0.5',
        '--teacher','E:\needle-teacher\luke_v2','--teacher-share','0.15','--teacher-seconds','120',
        '--p-speed','0.3','--speed-change','0.06',
        '--patience','40','--val-every','5000','--max-hours','11.5','--gpu-mem','0.8') + $stems
& F:\needle\run_train.ps1 -Name b32_f64_s2 -TrainArgs $s2 *>&1 | Out-File -Append -Encoding utf8 F:\needle\logs\b32_f64_s2.log
# stopped at --max-hours: run again with --resume
```

- `--batch 32`: b32 at frames 64 has about 4× the activations of b16/f32 (b16/f128 needed 16). If PyTorch spills into
  system RAM (train.py warns; ~100× slower) use 16 and keep `--lr 1e-3`.
- `--steps 300000`: b16 early-stopped at 165–190 k; a 4× model gets the longer schedule. Expect 10–14 h
  [INFERENCE from 3 h for b16 and 4× the MACs], hence `--max-hours 11.5` + `--resume`.
- `--teacher-share 0.15`: keeps part of the in-domain gain without the MUSDB loss of the 0.35 runs.
- `--mr-weight` stays off: it is confounded in v3; S2 should be one change so `paired.py` gives a clean slope.
- RAM: a stems + teacher run took 74 GB; if it does not fit, `--cache F:\needle\cache` memory-maps the decoded audio.
- Validation SDR is noisy (README: it picked the wrong winner once). Decide on test sets with `paired.py`, not `val.csv`.

## Evaluate and export (the gate's measurements)

```powershell
.venv\Scripts\python eval.py --checkpoint runs\b32_f64_s2\best.pt --musdb F:\needle\data\musdb18hq --workers 4
.venv\Scripts\python eval.py --checkpoint runs\b32_f64_s2\best.pt --refs F:\needle\data\stems\moisesdb_test
.venv\Scripts\python eval.py --checkpoint runs\b32_f64_s2\best.pt --refs E:\needle-teacher\luke_test_v2 --name b32_f64_s2_lukefair --skip-inaudible-db 40
.venv\Scripts\python paired.py results\b16_f64_test.json results\b32_f64_s2_test.json
.venv-export\Scripts\python export_vela.py --checkpoint runs\b32_f64_s2\best.pt --frames 64 --out build\b32_f64_s2 `
    --audio F:\needle\data\musdb18hq\train --audio-glob "*/mixture.wav"
.venv-export\Scripts\python eval_tflite.py --tflite build\b32_f64_s2\needle_b32_f64\needle_int8.tflite `
    --musdb F:\needle\data\musdb18hq --float results\b32_f64_s2_test.json
```

Expected [extrapolation, SCALING.md §5]: MUSDB avg 4.9–5.4, vocals 5.9–6.5; Vela int8 RTF ≈ 0.05 at SRAM speed /
0.09 from SDRAM (`build/random_b32_f64`), arena 5 MiB, weights 1.9 MB (XIP flash or SDRAM).

Decision rule: `paired.py` ≥ +0.3 dB over b16_f64 → the width slope holds, build S3 (band-compressed base 48,
complex I/O, TDF, MWF). Under +0.15 dB → skip width, go to the architecture levers directly (`SCALING.md` §4).

## Beyond S2 (code changes in `needle.py`, all Vela-expressible)

1. Complex spectrogram in/out: +0.6–0.7 dB vocals at equal parameters.
2. Band-compressed front end (bins 0–255 full resolution, 256–1023 pooled ×4): half the arena and RTF.
3. TDF blocks (TRANSPOSE + FULLY_CONNECTED): check 0 CPU operators with `export_vela.py` on random weights first.
4. Vocals-first cascade: the only published route to ~9 dB vocals under 70 M parameters.
5. Multichannel Wiener filter on the M85 (+0.45 dB, no training): testable now on `b16_f64` with `norbert`.

## Coordination (8 November 2026, Harry's chat session)

The base 32 run is started by `F:\needle\queue_v7.ps1` (copy: `queue_v7.example.ps1`) after the
base 16 / base 20 frames-128 runs and their exports. It runs training (two 11.5 h slices with
`--resume`), the four evaluations, the Vela export and the int8 evaluation itself, and writes
`results\b32_*_test.json` etc. **Do not start a duplicate training, evaluation or export for it**:
the PC's memory and GPU are fully used while it runs (last night two runs died of exactly that).

Recipe chosen: the frames-128 recipe of b16_f128_v3 / b20_f128_v3 (`-Recipe v3`), not the
frames-64 one above, so that base 16 / 20 / 32 form one width series with identical data and
training. Paired results of b16_f128_v3 against b16_f32_stems: MUSDB +0.09 [-0.04, +0.27],
MoisesDB -0.09 [-0.16, -0.02], house +1.06 [+0.26, +1.99] (6/6 songs), Luke Traxx held-out 6.24 vs
4.83. The comparison `b16_f64` vs `b32_f64_s2` proposed above would confound width with data
(b16_f64 was MUSDB-only); use `paired.py results\b16_f128_v3_test.json results\b32_f128_v3_test.json`.

## New music (2026-11-08) and the follow-up queue

`E:\needle-music\jamendo` (2072 Creative Commons tracks curated by genre: deephouse 325, house 700, progressive house
397, techno 650; `manifest.csv` carries the licences — by-nc / by-nc-sa, so private training only), `H:\` (215
house/Ibiza compilation FLACs, commercial, private training only) and the 184 Luke Traxx tracks already distilled
(`E:\needle-teacher\luke_v2`). `queue_s2.example.ps1` runs **after queue_v7** (never alongside it: RAM and GPU):
it distils H:\ and 120 tracks per Jamendo genre with htdemucs_ft (~4.5 h of GPU [estimate]), then trains
`b32_f128_v4` = the v3 recipe of queue_v7 with the new teacher sets added at the same `--teacher-share`, scores it, and
pairs it against `b32_f128_v3` — so the width series (b16/b20/b32 f128 v3) stays clean and this run measures the data
alone. Export and int8 eval follow. `--cache E:\needle-cache` as in queue_v7.
