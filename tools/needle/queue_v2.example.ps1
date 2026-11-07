# Stronger teacher, then the combined run. STOP file F:\needle\STOP halts between jobs.
$ErrorActionPreference = 'Continue'
function Log($m) { "$(Get-Date -Format s) $m" }
function Stopped { if (Test-Path F:\needle\STOP) { Log "STOP file, ending"; exit } }
Set-Location F:\needle\pocket-prep\tools\needle
$env:PYTHONUNBUFFERED = '1'
Log "waiting for b16_f32_kd training (pid 38288)"
while (Get-Process -Id 38288 -ErrorAction SilentlyContinue) { Start-Sleep 30 }
Stopped
foreach ($p in @(@('E:\needle-teacher\luke_test','E:\needle-teacher\luke_test_v2'), @('E:\needle-teacher\luke','E:\needle-teacher\luke_v2'))) {
  Log "teacher2 $($p[0]) -> $($p[1])"
  & F:\needle\.venv-sep\Scripts\python.exe teacher2.py --like $p[0] --out $p[1] 2>&1 | Where-Object { $_ -match '^\[|tracks to do|FAILED|rror' } | ForEach-Object { "   $_" }
  Stopped
}
Log "waiting for queue_kd evaluations (pid 43412)"
while (Get-Process -Id 43412 -ErrorAction SilentlyContinue) { Start-Sleep 30 }
Stopped
$stems = @('--stems','F:\needle\data\stems\moisesdb','--stems','F:\needle\data\stems\mshoxx','--stems','F:\needle\data\stems\ni','--stems','F:\needle\data\stems\packs','--stems','F:\needle\data\stems\wavgrind')
& F:\needle\run_train.ps1 -Name b16_f32_kd2 -TrainArgs (@('--base','16','--frames','32','--steps','300000','--stems-share','0.4','--teacher','E:\needle-teacher\luke_v2','--teacher-share','0.35','--teacher-seconds','120') + $stems) *>&1 | Out-File -Append -Encoding utf8 F:\needle\logs\b16_f32_kd2.log
Stopped
foreach ($run in 'b16_f32_kd2','b16_f32_kd') {
  foreach ($set in @(@('--musdb','F:\needle\data\musdb18hq'), @('--refs','F:\needle\data\stems\moisesdb_test'), @('--refs','F:\needle\data\stems\ni_test'))) {
    Log "eval $run $($set[1])"
    & .venv\Scripts\python.exe eval.py --checkpoint "runs\$run\best.pt" @set --workers 6 2>&1 | Where-Object { $_ -match 'SDR|rror' } | ForEach-Object { "   $_" }
  }
}
foreach ($run in 'b16_f32','b16_f32_stems','b16_f32_kd','b16_f32_kd2') {
  Log "luke_test_v2 $run"
  & .venv\Scripts\python.exe eval.py --checkpoint "runs\$run\best.pt" --refs E:\needle-teacher\luke_test_v2 --name "${run}_lukev2" --workers 6 2>&1 | Where-Object { $_ -match 'SDR|rror' } | ForEach-Object { "   $_" }
}
Log "V2 QUEUE DONE"
