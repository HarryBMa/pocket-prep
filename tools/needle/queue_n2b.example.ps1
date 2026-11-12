# n2b (12 Nov 2026): Needle 2 without the dual-path GRU (--rnn 0), everything else exactly as n2_a (queue_n2a).
# Why: the GRU blocks are 73 ms of n2_a's 250 ms per chunk on the NPU (RTF 0.084 -> 0.060 without them). This run
# says how much quality they buy, whatever n2_a's result is. Same data, same number of steps as n2_a.
# The live copy is F:\needle\queue_n2b.ps1; start it with
#   powershell -NoProfile -ExecutionPolicy Bypass -Command "& F:\needle\queue_n2b.ps1 *>&1 | Out-File -Append -Encoding utf8 F:\needle\logs\queue_n2b.log"
# It waits until queue_n2a and every other Needle job have finished, and starts only if n2_a trained to its last
# step (otherwise n2_a should be resumed first). Stop it by creating F:\needle\STOP_N2 (or STOP_TRAIN): checked
# while waiting and before every step. A stopped training continues later with the same command (--resume).
$ErrorActionPreference = 'Continue'
Set-Location F:\needle\pocket-prep\tools\needle
$env:PYTHONUNBUFFERED = '1'
$n = 'n2_b'
$full = 'n2_a'
$base = 'b20_jam3_frontier_env'

function Stopped { (Test-Path F:\needle\STOP_N2) -or (Test-Path F:\needle\STOP_TRAIN) }
function Busy {
  $others = Get-CimInstance Win32_Process | Where-Object {
    ($_.Name -match '^python' -and $_.CommandLine -match 'train\.py|eval\.py|eval_tflite\.py|hit_scan\.py|distill\.py|teacher2\.py|export_vela2\.py') -or
    ($_.Name -match '^(powershell|pwsh)' -and $_.ProcessId -ne $PID -and $_.CommandLine -match 'queue_(v\d+|n2a)')
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

"$(Get-Date -Format s) waiting until queue_n2a and every other Needle job have finished"
$idle = 0
while ($idle -lt 2) {
  if (Stopped) { "STOP file found, ending"; exit }
  if (Busy) { $idle = 0 } else { $idle++ }
  Start-Sleep -Seconds 60
}
if (-not (Test-Path "runs\$full\best.pt")) { "runs\$full\best.pt does not exist: n2_a did not train, so its ablation is not started"; exit }
$m = Get-CimInstance Win32_PerfFormattedData_PerfOS_Memory
$room = ($m.CommitLimit - $m.CommittedBytes) / 1GB
"$(Get-Date -Format s) nothing else is running; room for $([math]::Round($room)) GB more committed memory"
if ($room -lt 30) { "less than 30 GB of commit room: something else is using the memory, not starting"; exit }

# The same number of steps as n2_a, so the two differ only in the GRU.
$steps = if (Test-Path "runs\$full\steps.txt") { [int](Get-Content "runs\$full\steps.txt") } else { 100000 }
# n2_a must have reached its last step (train.py prints "finished at step N"); a run cut short by the time limit
# or a crash should be resumed before the GPU goes to its ablation.
$fin = Select-String -Path "F:\needle\logs\$full.log" -Pattern 'finished at step ([\d,]+)' -ErrorAction SilentlyContinue | Select-Object -Last 1
$reached = if ($fin) { [int]($fin.Matches[0].Groups[1].Value -replace ',', '') } else { 0 }
if ($reached -lt $steps) { "n2_a reached step $reached of ${steps}: resume it first, its ablation is not started"; exit }
"$(Get-Date -Format s) $steps steps, as n2_a"

if (Stopped) { "STOP file found, ending"; exit }
$stems = @('--stems','F:\needle\data\stems\moisesdb','--stems','F:\needle\data\stems\mshoxx','--stems','F:\needle\data\stems\ni','--stems','F:\needle\data\stems\packs','--stems','F:\needle\data\stems\wavgrind')
$a = @('--arch','bs','--rnn','0','--batch','8','--steps',"$steps",'--stems-share','0.4','--teacher','E:\needle-teacher\luke_v2','--teacher','E:\needle-teacher\plainset2\jamendo_v2','--teacher','B:\needle-teacher\jamendo_v3','--teacher-share','0.4','--teacher-seconds','60','--p-speed','0.3','--speed-change','0.06','--mr-weight','0.5','--env-weight','0.005','--cache','H:\needle-cache','--cache','F:\needle-cache2','--gpu-mem','0.75') + $stems
if (Test-Path "runs\$n\last.pt") { $a += '--resume' }
& F:\needle\run_train.ps1 -Name $n -TrainArgs $a *>&1 | Out-File -Append -Encoding utf8 "F:\needle\logs\$n.log"
if (-not (Test-Path "runs\$n\best.pt")) { "no checkpoint: the training failed, see F:\needle\logs\$n.log"; exit }

$hold = @('--refs','E:\needle-teacher\holdout_house\jamendo_v2','--skip-inaudible-db','40')
Ev $n "${n}_holdout" $hold @()
Ev $n "${n}_ibiza" @('--refs','E:\needle-teacher\ibiza_test_v2','--skip-inaudible-db','40') @()
Ev $n $n @('--musdb','F:\needle\data\musdb18hq') @()
Ev $n $n @('--refs','F:\needle\data\stems\moisesdb_test') @()
Ev $n $n @('--refs','F:\needle\data\stems\ni_test') @()
"$(Get-Date -Format s) paired: what the GRU buys ($full minus $n), song by song"
Paired "${n}_test.json" "${full}_test.json"
Paired "${n}_moisesdb_test.json" "${full}_moisesdb_test.json"
Paired "${n}_ni_test.json" "${full}_ni_test.json"
Paired "${n}_holdout_jamendo_v2.json" "${full}_holdout_jamendo_v2.json"
Paired "${n}_ibiza_ibiza_test_v2.json" "${full}_ibiza_ibiza_test_v2.json"
"$(Get-Date -Format s) paired: Needle 2 without GRU minus $base"
Paired "${base}_test.json" "${n}_test.json"
Paired "${base}_moisesdb_test.json" "${n}_moisesdb_test.json"
Paired "${base}_holdout_jamendo_v2.json" "${n}_holdout_jamendo_v2.json"
if (-not (Stopped)) {
  "$(Get-Date -Format s) int8 export and Vela"
  $env:CUDA_VISIBLE_DEVICES = '-1'
  & .venv-export\Scripts\python.exe export_vela2.py --checkpoint "runs\$n\best.pt" --audio F:\needle\data\musdb18hq\train --audio-glob "*/mixture.wav" --name $n 2>&1 | Where-Object { "$_" -match 'TensorFlow copy|^int8|^params|rror|Traceback' } | ForEach-Object { "   $_" }
  "$(Get-Date -Format s) int8 SDR on the MoisesDB test set"
  & .venv-export\Scripts\python.exe eval_tflite.py --tflite "build\$n\needle_int8.tflite" --refs F:\needle\data\stems\moisesdb_test --float "results\${n}_moisesdb_test.json" --workers 4 2>&1 | Where-Object { "$_" -match 'SDR drums|quantisation|rror|Traceback' } | ForEach-Object { "   $_" }
}
"N2B QUEUE DONE $(Get-Date -Format s)"
