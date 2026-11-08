# Teacher stems from the new music, then the S2 training (S2_RECIPE.md). Copy, set $WaitPid, run in PowerShell.
# A file F:\needle\STOP3 halts it between jobs. GPU time [estimates]: distill ~15-25 s per track on the 3080 Ti.
#   H:\                      215 house/Ibiza FLACs (in-domain, commercial, private training only)      ~1.5 h
#   E:\needle-music\jamendo  2072 CC tracks curated by genre (deephouse 325, house 700, progressive 397, techno 650,
#                            manifest.csv with licences: by-nc, by-nc-sa ...; private training only)   ~120 per genre ~3 h
# Then S2 twice: the MUSDB-protective mix (gate 4 is MUSDB test) and a DJ-weighted mix for the product metric.
$ErrorActionPreference = 'Continue'
function Log($m) { "$(Get-Date -Format s) $m" }
function Stopped { if (Test-Path F:\needle\STOP3) { Log "STOP3 file, ending"; exit } }
Set-Location F:\needle\pocket-prep\tools\needle
$env:PYTHONUNBUFFERED = '1'
$WaitPid = 0   # PID of a running train.py to wait for (0 = start now)
if ($WaitPid) { Log "waiting for pid $WaitPid"; while (Get-Process -Id $WaitPid -ErrorAction SilentlyContinue) { Start-Sleep 60 } }
Stopped

# 1. htdemucs_ft teacher stems (resumable; failures in <out>\failed.tsv)
Log "distill H:\ -> E:\needle-teacher\ibiza"
& .venv\Scripts\python.exe distill.py --input 'H:\' --out E:\needle-teacher\ibiza --max-minutes 12 2>&1 | Where-Object { $_ -match 'tracks|FAILED|rror|done' } | ForEach-Object { "   $_" }
Stopped
foreach ($g in 'deephouse','house','progressivehouse','techno') {
  Log "distill jamendo/$g (first 120) -> E:\needle-teacher\jamendo_$g"
  & .venv\Scripts\python.exe distill.py --input "E:\needle-music\jamendo\$g" --glob '*' --limit 120 --out "E:\needle-teacher\jamendo_$g" --max-minutes 12 2>&1 | Where-Object { $_ -match 'tracks|FAILED|rror|done' } | ForEach-Object { "   $_" }
  Stopped
}
# optional, stronger vocals teacher (RoFormer + htdemucs on the instrumental), redo of the caches above:
# & F:\needle\.venv-sep\Scripts\python.exe teacher2.py --like E:\needle-teacher\ibiza --out E:\needle-teacher\ibiza_v2

# 2. S2, MUSDB-protective (the gate): real stems 0.5, all teachers together 0.15
$stems = @('--stems','F:\needle\data\stems\moisesdb','--stems','F:\needle\data\stems\mshoxx','--stems','F:\needle\data\stems\ni','--stems','F:\needle\data\stems\packs','--stems','F:\needle\data\stems\wavgrind')
$teachers = @('--teacher','E:\needle-teacher\luke_v2','--teacher','E:\needle-teacher\ibiza',
              '--teacher','E:\needle-teacher\jamendo_deephouse','--teacher','E:\needle-teacher\jamendo_house',
              '--teacher','E:\needle-teacher\jamendo_progressivehouse','--teacher','E:\needle-teacher\jamendo_techno',
              '--teacher-seconds','120')
$s2 = @('--base','32','--frames','64','--batch','32','--steps','300000','--stems-share','0.5','--teacher-share','0.15',
        '--p-speed','0.3','--speed-change','0.06','--patience','40','--val-every','5000','--max-hours','11.5','--gpu-mem','0.8',
        '--cache','F:\needle\cache') + $stems + $teachers
& F:\needle\run_train.ps1 -Name b32_f64_s2 -TrainArgs $s2 *>&1 | Out-File -Append -Encoding utf8 F:\needle\logs\b32_f64_s2.log
Stopped

# 3. S2-dj, the product mix: same net, teachers 0.4 (expect MUSDB a few tenths lower, house/luke higher)
$s2dj = @('--base','32','--frames','64','--batch','32','--steps','300000','--stems-share','0.35','--teacher-share','0.4',
          '--p-speed','0.3','--speed-change','0.06','--patience','40','--val-every','5000','--max-hours','11.5','--gpu-mem','0.8',
          '--cache','F:\needle\cache') + $stems + $teachers
& F:\needle\run_train.ps1 -Name b32_f64_s2dj -TrainArgs $s2dj *>&1 | Out-File -Append -Encoding utf8 F:\needle\logs\b32_f64_s2dj.log
Stopped

# 4. scoring
foreach ($run in 'b32_f64_s2','b32_f64_s2dj') {
  foreach ($set in @(@('--musdb','F:\needle\data\musdb18hq'), @('--refs','F:\needle\data\stems\moisesdb_test'), @('--refs','F:\needle\data\stems\ni_test'))) {
    Log "eval $run $($set[1])"
    & .venv\Scripts\python.exe eval.py --checkpoint "runs\$run\best.pt" @set --workers 4 2>&1 | Where-Object { $_ -match 'SDR|rror' } | ForEach-Object { "   $_" }
  }
  Log "luke_test_v2 fair $run"
  & .venv\Scripts\python.exe eval.py --checkpoint "runs\$run\best.pt" --refs E:\needle-teacher\luke_test_v2 --name "${run}_lukefair" --skip-inaudible-db 40 --workers 3 2>&1 | Where-Object { $_ -match 'SDR|rror' } | ForEach-Object { "   $_" }
  & .venv\Scripts\python.exe paired.py results\b16_f64_test.json "results\${run}_test.json"
}
Log "queue done"
