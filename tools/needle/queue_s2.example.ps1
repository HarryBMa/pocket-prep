# After queue_v7 (base 32 frames-128 v3 + its exports): teacher stems from the new music, then one run that isolates
# the DATA effect at fixed width: b32_f128_v4 = the v3 recipe with the new teacher sets added. Compare with
# paired.py against b32_f128_v3 (same net, same recipe, more/other teacher data). Copy, set $WaitPid, run.
# Halt file F:\needle\STOP4 (queue_v7 uses STOP3). GPU time [estimates, 3080 Ti]: distill ~15-25 s per track.
#   H:\                      215 house/Ibiza FLACs (in-domain, commercial, private training only)      ~1.5 h
#   E:\needle-music\jamendo  2072 CC tracks by genre (deephouse 325, house 700, progressive 397, techno 650;
#                            manifest.csv: by-nc / by-nc-sa licences, private training only), 120 per genre  ~3 h
# Do not run while queue_v7 trains or exports: RAM and GPU are fully used then (two runs died of that on 7 Nov).
$ErrorActionPreference = 'Continue'
function Log($m) { "$(Get-Date -Format s) $m" }
function Stopped { if (Test-Path F:\needle\STOP4) { Log "STOP4 file, ending"; exit } }
Set-Location F:\needle\pocket-prep\tools\needle
$env:PYTHONUNBUFFERED = '1'
$WaitPid = 0   # PID of the running queue_v7.ps1 PowerShell (0 = start now)
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
# optional, stronger vocals teacher (RoFormer + htdemucs on the instrumental), redo of a cache above:
# & F:\needle\.venv-sep\Scripts\python.exe teacher2.py --like E:\needle-teacher\ibiza --out E:\needle-teacher\ibiza_v2

# 2. b32_f128_v4: the v3 recipe (queue_v7) with the new teacher sets. Same --teacher-share, so the paired comparison
#    with b32_f128_v3 measures the data, nothing else. Two 11.5 h slices with --resume, like queue_v7.
$stems = @('--stems','F:\needle\data\stems\moisesdb','--stems','F:\needle\data\stems\mshoxx','--stems','F:\needle\data\stems\ni','--stems','F:\needle\data\stems\packs','--stems','F:\needle\data\stems\wavgrind')
$teachers = @('--teacher','E:\needle-teacher\luke_v2','--teacher','E:\needle-teacher\ibiza',
              '--teacher','E:\needle-teacher\jamendo_deephouse','--teacher','E:\needle-teacher\jamendo_house',
              '--teacher','E:\needle-teacher\jamendo_progressivehouse','--teacher','E:\needle-teacher\jamendo_techno')
$run = 'b32_f128_v4'
$a = @('--base','32','--frames','128','--batch','8','--steps','300000','--stems-share','0.4','--teacher-share','0.35','--teacher-seconds','120',
       '--p-speed','0.3','--speed-change','0.06','--mr-weight','0.5','--cache','E:\needle-cache','--max-hours','11.5') + $stems + $teachers
Log "train $run"
& F:\needle\run_train.ps1 -Name $run -TrainArgs $a *>&1 | Out-File -Append -Encoding utf8 "F:\needle\logs\$run.log"
if (-not (Select-String "F:\needle\logs\$run.log" -Pattern 'finished at step' -Quiet)) {
  Log "resume $run"
  & F:\needle\run_train.ps1 -Name $run -TrainArgs ($a + '--resume') *>&1 | Out-File -Append -Encoding utf8 "F:\needle\logs\$run.log"
}
Stopped

# 3. scoring, paired against the same-width run without the new data
foreach ($set in @(@('--musdb','F:\needle\data\musdb18hq'), @('--refs','F:\needle\data\stems\moisesdb_test'), @('--refs','F:\needle\data\stems\ni_test'))) {
  Log "eval $run $($set[1])"
  & .venv\Scripts\python.exe eval.py --checkpoint "runs\$run\best.pt" @set --workers 4 2>&1 | Where-Object { $_ -match 'SDR|rror' } | ForEach-Object { "   $_" }
}
Log "luke_test_v2 fair $run"
& .venv\Scripts\python.exe eval.py --checkpoint "runs\$run\best.pt" --refs E:\needle-teacher\luke_test_v2 --name "${run}_lukefair" --skip-inaudible-db 40 --workers 3 2>&1 | Where-Object { $_ -match 'SDR|rror' } | ForEach-Object { "   $_" }
& .venv\Scripts\python.exe paired.py results\b32_f128_v3_test.json "results\${run}_test.json"
& .venv\Scripts\python.exe paired.py results\b32_f128_v3_moisesdb_test.json "results\${run}_moisesdb_test.json"
Log "export $run"
& .venv-export\Scripts\python.exe export_vela.py --checkpoint "runs\$run\best.pt" --frames 128 --out "build\$run" --audio F:\needle\data\musdb18hq\train --audio-glob "*/mixture.wav" 2>&1 | Select-String '"params"|inference_ms|arena_kib|weights_kib|cpu_operators|rror' | % { "   $_" }
Log "S2 DATA QUEUE DONE"
