# Morning 9 Nov: float evals that OOM'd overnight (b20, b32), the Ibiza test set for every model,
# then paired tables. GPU cap 0.4 (4.8 GB): the Jamendo teacher runs beside it.
$ErrorActionPreference = 'Continue'
function Log($m) { "$(Get-Date -Format s) $m" }
Set-Location F:\needle\pocket-prep\tools\needle
$env:PYTHONUNBUFFERED = '1'; $env:PYTHONIOENCODING = 'utf-8'
foreach ($run in 'b20_f128_v3','b32_f128_v3') {
  foreach ($set in @(@('--musdb','F:\needle\data\musdb18hq'), @('--refs','F:\needle\data\stems\moisesdb_test'), @('--refs','F:\needle\data\stems\ni_test'))) {
    Log "eval $run $($set[1])"
    & .venv\Scripts\python.exe eval.py --checkpoint "runs\$run\best.pt" @set --workers 3 --gpu-mem 0.4 2>&1 | Where-Object { $_ -match 'SDR|rror' } | ForEach-Object { "   $_" }
  }
  Log "luke fair $run"
  & .venv\Scripts\python.exe eval.py --checkpoint "runs\$run\best.pt" --refs E:\needle-teacher\luke_test_v2 --name "${run}_lukefair" --skip-inaudible-db 40 --workers 3 --gpu-mem 0.4 2>&1 | Where-Object { $_ -match 'SDR|rror' } | ForEach-Object { "   $_" }
}
foreach ($run in 'b16_f128_v3','b20_f128_v3','b32_f128_v3','b16_f32_stems','b16_f32','b16_f32_kd') {
  Log "ibiza $run"
  & .venv\Scripts\python.exe eval.py --checkpoint "runs\$run\best.pt" --refs E:\needle-teacher\ibiza_test_v2 --name "${run}_ibiza" --skip-inaudible-db 40 --workers 3 --gpu-mem 0.4 2>&1 | Where-Object { $_ -match 'SDR|rror' } | ForEach-Object { "   $_" }
}
Log "V8 QUEUE DONE"
