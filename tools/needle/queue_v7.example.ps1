# Base 32 after queue_v6 (b16/b20 f128 v3 + exports). Recipe picked at the top: 'v3' (same as
# b16/b20: frames 128, speed aug, MR loss, teacher 0.35) or 's2' (S2_RECIPE.md: frames 64,
# stems 0.5, teacher 0.15, no MR loss). Memory-mapped cache; 11.5 h slices with resume.
param([string]$Recipe = 'v3')
$ErrorActionPreference = 'Continue'
function Log($m) { "$(Get-Date -Format s) $m" }
Set-Location F:\needle\pocket-prep\tools\needle
$env:PYTHONUNBUFFERED = '1'
$stems = @('--stems','F:\needle\data\stems\moisesdb','--stems','F:\needle\data\stems\mshoxx','--stems','F:\needle\data\stems\ni','--stems','F:\needle\data\stems\packs','--stems','F:\needle\data\stems\wavgrind')
if ($Recipe -eq 'v3') {
  $run = 'b32_f128_v3'; $frames = '128'
  $a = @('--base','32','--frames','128','--batch','8','--steps','300000','--stems-share','0.4','--teacher','E:\needle-teacher\luke_v2','--teacher-share','0.35','--teacher-seconds','120','--p-speed','0.3','--speed-change','0.06','--mr-weight','0.5')
} else {
  $run = 'b32_f64_s2'; $frames = '64'
  $a = @('--base','32','--frames','64','--batch','32','--steps','300000','--stems-share','0.5','--teacher','E:\needle-teacher\luke_v2','--teacher-share','0.15','--teacher-seconds','120','--p-speed','0.3','--speed-change','0.06','--patience','40')
}
$a += @('--cache','E:\needle-cache','--max-hours','11.5') + $stems
Log "waiting for queue_v6 (pid 36804)"
while (Get-Process -Id 36804 -ErrorAction SilentlyContinue) { Start-Sleep 60 }
if (Test-Path F:\needle\STOP3) { Log "STOP3, ending"; exit }
Log "train $run ($Recipe)"
& F:\needle\run_train.ps1 -Name $run -TrainArgs $a *>&1 | Out-File -Append -Encoding utf8 "F:\needle\logs\$run.log"
if (-not (Select-String "F:\needle\logs\$run.log" -Pattern 'finished at step' -Quiet)) {          # hit --max-hours: one more slice
  Log "resume $run"
  & F:\needle\run_train.ps1 -Name $run -TrainArgs ($a + '--resume') *>&1 | Out-File -Append -Encoding utf8 "F:\needle\logs\$run.log"
}
if (Test-Path F:\needle\STOP3) { Log "STOP3, ending"; exit }
foreach ($set in @(@('--musdb','F:\needle\data\musdb18hq'), @('--refs','F:\needle\data\stems\moisesdb_test'), @('--refs','F:\needle\data\stems\ni_test'))) {
  Log "eval $run $($set[1])"
  & .venv\Scripts\python.exe eval.py --checkpoint "runs\$run\best.pt" @set --workers 4 2>&1 | Where-Object { $_ -match 'SDR|rror' } | ForEach-Object { "   $_" }
}
Log "luke fair $run"
& .venv\Scripts\python.exe eval.py --checkpoint "runs\$run\best.pt" --refs E:\needle-teacher\luke_test_v2 --name "${run}_lukefair" --skip-inaudible-db 40 --workers 3 2>&1 | Where-Object { $_ -match 'SDR|rror' } | ForEach-Object { "   $_" }
Log "export $run"
& .venv-export\Scripts\python.exe export_vela.py --checkpoint "runs\$run\best.pt" --frames $frames --out "build\$run" --audio F:\needle\data\musdb18hq\train --audio-glob "*/mixture.wav" 2>&1 | Select-String '"params"|inference_ms|arena_kib|weights_kib|cpu_operators|rror' | % { "   $_" }
$tfl = Get-ChildItem "build\$run" -Recurse -Filter needle_int8.tflite | select -First 1
& .venv-export\Scripts\python.exe eval_tflite.py --tflite $tfl.FullName --musdb F:\needle\data\musdb18hq --float "results\${run}_test.json" --workers 4 2>&1 | Where-Object { $_ -match 'SDR|loss|rror' } | % { "   $_" }
Log "V7 QUEUE DONE"
