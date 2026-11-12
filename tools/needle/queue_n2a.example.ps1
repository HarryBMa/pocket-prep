# n2a (12 Nov 2026): the first Needle 2 run (band-split grid, TDF, dual-path GRU: needle2.py, NEEDLE2.md).
# Same data, shares, augmentation and losses as b20_jam3_frontier_env (queue_v16), so the two results differ only
# in the model. The live copy is F:\needle\queue_n2a.ps1; start it with
#   powershell -NoProfile -ExecutionPolicy Bypass -Command "& F:\needle\queue_n2a.ps1 *>&1 | Out-File -Append -Encoding utf8 F:\needle\logs\queue_n2a.log"
# It waits until no other Needle training / evaluation / queue is running (the GPU and the RAM hold one at a time),
# then: speed probe -> training (one slice of at most 11.5 h) -> the five evaluations -> paired comparison with
# b20_jam3_frontier_env -> drum hit scan -> int8 export, Vela, int8 SDR.
# Stop it by creating F:\needle\STOP_N2 (or STOP_TRAIN): checked while waiting and before every step.
$ErrorActionPreference = 'Continue'
Set-Location F:\needle\pocket-prep\tools\needle
$env:PYTHONUNBUFFERED = '1'
$n = 'n2_a'
$base = 'b20_jam3_frontier_env'

function Stopped { (Test-Path F:\needle\STOP_N2) -or (Test-Path F:\needle\STOP_TRAIN) }
function Busy {
  # another Needle job: a training / evaluation process, or one of the numbered queue scripts
  $others = Get-CimInstance Win32_Process | Where-Object {
    ($_.Name -match '^python' -and $_.CommandLine -match 'train\.py|eval\.py|eval_tflite\.py|hit_scan\.py|distill\.py|teacher2\.py') -or
    ($_.Name -match '^(powershell|pwsh)' -and $_.ProcessId -ne $PID -and $_.CommandLine -match 'queue_v\d+')
  }
  [bool]$others
}
function Ev($ck, $name, $set, $extra) {
  if (Stopped) { return }
  "$(Get-Date -Format s) eval $name"
  & .venv\Scripts\python.exe eval.py --checkpoint "runs\$ck\best.pt" @set --name $name --workers 3 --gpu-mem 0.5 @extra 2>&1 | Where-Object { $_ -match 'SDR|rror|Traceback' } | ForEach-Object { "   $_" }
}
function Paired($a, $b) {
  if ((Test-Path "results\$a") -and (Test-Path "results\$b")) {
    & .venv\Scripts\python.exe paired.py "results\$a" "results\$b" 2>&1 | ForEach-Object { "   $_" }
  } else { "   (no paired comparison: results\$a or results\$b is missing)" }
}

"$(Get-Date -Format s) waiting until no other Needle job is running"
$idle = 0
while ($idle -lt 2) {                       # two checks a minute apart: a queue has short gaps between its steps
  if (Stopped) { "STOP file found, ending"; exit }
  if (Busy) { $idle = 0 } else { $idle++ }
  Start-Sleep -Seconds 60
}
$free = (Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory / 1MB
"$(Get-Date -Format s) nothing else is running; free RAM $([math]::Round($free)) GB"
if ($free -lt 90) { "less than 90 GB of RAM is free and the run needs about 87 GB: not starting"; exit }

# Steps: as many as fit one night at the speed this GPU really trains Needle 2 (12 steps on random audio), at most
# 150,000. A full, shorter schedule is better than one cut off by --max-hours. Kept in steps.txt for --resume.
New-Item -ItemType Directory -Force "runs\$n" | Out-Null
$stepsFile = "runs\$n\steps.txt"
if (Test-Path $stepsFile) { $steps = [int](Get-Content $stepsFile) }
else {
  $line = & .venv\Scripts\python.exe test_needle2.py --speed 8 --gpu-mem 0.6 2>&1 | Select-String -Pattern '([\d.]+) steps/s' | Select-Object -Last 1
  $speed = if ($line) { [double]$line.Matches[0].Groups[1].Value } else { 0 }
  $steps = if ($speed -gt 0) { [int][math]::Min(150000, [math]::Max(40000, [math]::Floor($speed * 0.85 * 9.5 * 3600 / 5000) * 5000)) } else { 100000 }
  "$(Get-Date -Format s) speed probe: $speed steps/s at batch 8 -> $steps steps"
  Set-Content $stepsFile $steps
}

if (Stopped) { "STOP file found, ending"; exit }
$stems = @('--stems','F:\needle\data\stems\moisesdb','--stems','F:\needle\data\stems\mshoxx','--stems','F:\needle\data\stems\ni','--stems','F:\needle\data\stems\packs','--stems','F:\needle\data\stems\wavgrind')
$a = @('--arch','bs','--batch','8','--steps',"$steps",'--stems-share','0.4','--teacher','E:\needle-teacher\luke_v2','--teacher','E:\needle-teacher\plainset2\jamendo_v2','--teacher','B:\needle-teacher\jamendo_v3','--teacher-share','0.4','--teacher-seconds','60','--p-speed','0.3','--speed-change','0.06','--mr-weight','0.5','--env-weight','0.005','--cache','H:\needle-cache','--cache','F:\needle-cache2','--gpu-mem','0.75') + $stems
if (Test-Path "runs\$n\last.pt") { $a += '--resume' }
& F:\needle\run_train.ps1 -Name $n -TrainArgs $a *>&1 | Out-File -Append -Encoding utf8 "F:\needle\logs\$n.log"
if (-not (Test-Path "runs\$n\best.pt")) { "no checkpoint: the training failed, see F:\needle\logs\$n.log"; exit }

$hold = @('--refs','E:\needle-teacher\holdout_house\jamendo_v2','--skip-inaudible-db','40')
Ev $n "${n}_holdout" $hold @()
Ev $n "${n}_ibiza" @('--refs','E:\needle-teacher\ibiza_test_v2','--skip-inaudible-db','40') @()
Ev $n $n @('--musdb','F:\needle\data\musdb18hq') @()
Ev $n $n @('--refs','F:\needle\data\stems\moisesdb_test') @()
Ev $n $n @('--refs','F:\needle\data\stems\ni_test') @()
"$(Get-Date -Format s) paired: Needle 2 minus $base, song by song (same training data)"
Paired "${base}_test.json" "${n}_test.json"
Paired "${base}_moisesdb_test.json" "${n}_moisesdb_test.json"
Paired "${base}_ni_test.json" "${n}_ni_test.json"
Paired "${base}_holdout_jamendo_v2.json" "${n}_holdout_jamendo_v2.json"
Paired "${base}_ibiza_ibiza_test_v2.json" "${n}_ibiza_ibiza_test_v2.json"
if (-not (Stopped) -and (Test-Path "runs\$base\best.pt")) {
  "$(Get-Date -Format s) hit_scan"
  & .venv\Scripts\python.exe hit_scan.py "runs\$base\best.pt" "runs\$n\best.pt" 2>&1 | Select-Object -Last 3 | ForEach-Object { "   $_" }
}
if (-not (Stopped)) {
  "$(Get-Date -Format s) int8 export and Vela"
  $env:CUDA_VISIBLE_DEVICES = '-1'
  & .venv-export\Scripts\python.exe export_vela2.py --checkpoint "runs\$n\best.pt" --audio F:\needle\data\musdb18hq\train --audio-glob "*/mixture.wav" --name $n 2>&1 | Where-Object { "$_" -match 'TensorFlow copy|^int8|^params|rror|Traceback' } | ForEach-Object { "   $_" }
  "$(Get-Date -Format s) int8 SDR on the MoisesDB test set"
  & .venv-export\Scripts\python.exe eval_tflite.py --tflite "build\$n\needle_int8.tflite" --refs F:\needle\data\stems\moisesdb_test --float "results\${n}_moisesdb_test.json" --workers 4 2>&1 | Where-Object { "$_" -match 'SDR drums|quantisation|rror|Traceback' } | ForEach-Object { "   $_" }
}
"N2A QUEUE DONE $(Get-Date -Format s)"
