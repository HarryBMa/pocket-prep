# After the new teacher (queue_v2, which stops on F:\needle\STOP before its own training):
# the improved run, then fair scoring. A file F:\needle\STOP2 halts it between jobs.
$ErrorActionPreference = 'Continue'
function Log($m) { "$(Get-Date -Format s) $m" }
function Stopped { if (Test-Path F:\needle\STOP2) { Log "STOP2 file, ending"; exit } }
Set-Location F:\needle\pocket-prep\tools\needle
$env:PYTHONUNBUFFERED = '1'
Log "waiting for queue_v2 / teacher2 (pid 39060)"
while (Get-Process -Id 39060 -ErrorAction SilentlyContinue) { Start-Sleep 60 }
Stopped
$stems = @('--stems','F:\needle\data\stems\moisesdb','--stems','F:\needle\data\stems\mshoxx','--stems','F:\needle\data\stems\ni','--stems','F:\needle\data\stems\packs','--stems','F:\needle\data\stems\wavgrind')
$args3 = @('--base','16','--frames','128','--batch','16','--steps','300000','--stems-share','0.4','--teacher','E:\needle-teacher\luke_v2','--teacher-share','0.35','--teacher-seconds','120','--p-speed','0.3','--speed-change','0.06','--mr-weight','0.5') + $stems
& F:\needle\run_train.ps1 -Name b16_f128_v3 -TrainArgs $args3 *>&1 | Out-File -Append -Encoding utf8 F:\needle\logs\b16_f128_v3.log
Stopped
foreach ($set in @(@('--musdb','F:\needle\data\musdb18hq'), @('--refs','F:\needle\data\stems\moisesdb_test'), @('--refs','F:\needle\data\stems\ni_test'))) {
  Log "eval b16_f128_v3 $($set[1])"
  & .venv\Scripts\python.exe eval.py --checkpoint runs\b16_f128_v3\best.pt @set --workers 4 2>&1 | Where-Object { $_ -match 'SDR|rror' } | ForEach-Object { "   $_" }
}
foreach ($run in 'b16_f32','b16_f32_stems','b16_f64','b16_f32_kd','b16_f128_v3') {
  Log "luke_test_v2 fair $run"
  & .venv\Scripts\python.exe eval.py --checkpoint "runs\$run\best.pt" --refs E:\needle-teacher\luke_test_v2 --name "${run}_lukefair" --skip-inaudible-db 40 --workers 3 2>&1 | Where-Object { $_ -match 'SDR|rror' } | ForEach-Object { "   $_" }
}
Log "V3 QUEUE DONE"
