# Resume the two improved runs with memory-mapped data, score them, export int8.
# A file F:\needle\STOP3 halts it between jobs.
$ErrorActionPreference = 'Continue'
function Log($m) { "$(Get-Date -Format s) $m" }
function Stopped { if (Test-Path F:\needle\STOP3) { Log "STOP3 file, ending"; exit } }
Set-Location F:\needle\pocket-prep\tools\needle
$env:PYTHONUNBUFFERED = '1'
$stems = @('--stems','F:\needle\data\stems\moisesdb','--stems','F:\needle\data\stems\mshoxx','--stems','F:\needle\data\stems\ni','--stems','F:\needle\data\stems\packs','--stems','F:\needle\data\stems\wavgrind')
foreach ($b in '16','20') {
  $run = "b${b}_f128_v3"
  $args3 = @('--base',$b,'--frames','128','--batch','16','--steps','300000','--stems-share','0.4','--teacher','E:\needle-teacher\luke_v2','--teacher-share','0.35','--teacher-seconds','120','--p-speed','0.3','--speed-change','0.06','--mr-weight','0.5','--cache','E:\needle-cache','--resume') + $stems
  Log "resume $run"
  & F:\needle\run_train.ps1 -Name $run -TrainArgs $args3 *>&1 | Out-File -Append -Encoding utf8 "F:\needle\logs\$run.log"
  Stopped
  foreach ($set in @(@('--musdb','F:\needle\data\musdb18hq'), @('--refs','F:\needle\data\stems\moisesdb_test'), @('--refs','F:\needle\data\stems\ni_test'))) {
    Log "eval $run $($set[1])"
    & .venv\Scripts\python.exe eval.py --checkpoint "runs\$run\best.pt" @set --workers 4 2>&1 | Where-Object { $_ -match 'SDR|rror' } | ForEach-Object { "   $_" }
  }
  Log "luke fair $run"
  & .venv\Scripts\python.exe eval.py --checkpoint "runs\$run\best.pt" --refs E:\needle-teacher\luke_test_v2 --name "${run}_lukefair" --skip-inaudible-db 40 --workers 3 2>&1 | Where-Object { $_ -match 'SDR|rror' } | ForEach-Object { "   $_" }
  Stopped
}
foreach ($run in 'b16_f128_v3','b20_f128_v3') {
  Log "export $run"
  & .venv-export\Scripts\python.exe export_vela.py --checkpoint "runs\$run\best.pt" --frames 128 --out "build\$run" --audio F:\needle\data\musdb18hq\train --audio-glob "*/mixture.wav" 2>&1 | Select-String '"params"|inference_ms|arena_kib|weights_kib|cpu_operators|rror' | % { "   $_" }
  $tfl = Get-ChildItem "build\$run" -Recurse -Filter needle_int8.tflite | select -First 1
  Log "int8 eval $run"
  & .venv-export\Scripts\python.exe eval_tflite.py --tflite $tfl.FullName --musdb F:\needle\data\musdb18hq --float "results\${run}_test.json" --workers 4 2>&1 | Where-Object { $_ -match 'SDR|loss|rror' } | % { "   $_" }
}
Log "V6 QUEUE DONE"
