# One Needle 2 run from start to finish: wait for the GPU, train, evaluate, compare song by song, scan the drum
# hits, export int8. The recipe (data, shares, augmentation, losses, batch 8) is n2_a's; -Extra adds or changes
# train.py arguments. The live copy is F:\needle\queue_n2x.ps1. Examples (13 Nov 2026):
#
#   n2_d: the sub-frame envelope input, from scratch, with one averaged copy of the weights
#     powershell -NoProfile -ExecutionPolicy Bypass -Command "& F:\needle\queue_n2x.ps1 -Name n2_d -Extra '--sub','4','--ema','0.999' *>&1 | Out-File -Append -Encoding utf8 F:\needle\logs\queue_n2d.log"
#   n2_c: 150,000 more steps on n2_a with a new schedule and two averaged copies; waits for n2_d
#     powershell ... -Command "& F:\needle\queue_n2x.ps1 -Name n2_c -Init n2_a -Extra '--lr','5e-4','--ema','0.999','0.9999' -After n2_d *>&1 | Out-File ... F:\needle\logs\queue_n2c.log"
#
# -Name must come first: a waiting queue recognises the queues it has to wait for (-After) by "queue_n2x.ps1 -Name
# <name>" in their command lines. The older queues (queue_v*, queue_n2a, queue_n2b) are always waited for.
#
# With --ema the run keeps best_raw.pt (the trained weights' best) and best.pt (the best of the trained weights and
# the averaged copies). The five evaluations and the comparison with -Against use best_raw.pt, so runs with and
# without --ema compare like for like; best.pt is scored on the two real-stem sets as <Name>_ema when it is an
# averaged copy, and it is the checkpoint that goes to the hit scans and to int8 (the one that would ship).
# Stop: create F:\needle\STOP_N2 (or STOP_TRAIN), checked while waiting and before every step; then end the running
# python train.py if there is one. Starting the same command again continues the training from its last validation.
param(
  [Parameter(Mandatory = $true)][string]$Name,
  [string[]]$Extra = @(),
  [int]$Steps = 150000,
  [string]$Init = '',                                # train on from runs\<Init>\best.pt (train.py --init)
  [string]$Against = 'n2_a',                         # the run to compare with, song by song
  [string[]]$After = @()                             # names of other queue_n2x runs that must have finished first
)
$ErrorActionPreference = 'Continue'
Set-Location F:\needle\pocket-prep\tools\needle
$env:PYTHONUNBUFFERED = '1'
$n = $Name
$base = 'b20_jam3_frontier_env'                      # Needle 1 on the same data
# Queues to wait for: the script path (not just the word, which other queues carry in their arguments).
$waitFor = (@('\\queue_(v\d+|n2a|n2b)\.ps1') + ($After | ForEach-Object { 'queue_n2x\.ps1 -Name ' + [regex]::Escape($_) + '\b' })) -join '|'

function Stopped { (Test-Path F:\needle\STOP_N2) -or (Test-Path F:\needle\STOP_TRAIN) }
function Busy {
  $others = Get-CimInstance Win32_Process | Where-Object {
    ($_.Name -match '^python' -and $_.CommandLine -match 'train\.py|eval\.py|eval_tflite\.py|hit_scan\.py|distill\.py|teacher2\.py|export_vela2\.py') -or
    ($_.Name -match '^(powershell|pwsh)' -and $_.ProcessId -ne $PID -and $_.CommandLine -match $waitFor)
  }
  [bool]$others
}
function Ev($ck, $name, $set) {
  if (Stopped) { return }
  "$(Get-Date -Format s) eval $name ($ck)"
  & .venv\Scripts\python.exe eval.py --checkpoint $ck @set --name $name --workers 3 --gpu-mem 0.5 2>&1 | Where-Object { $_ -match 'SDR|rror|Traceback' } | ForEach-Object { "   $_" }
}
function Paired($a, $b) {
  if ((Test-Path "results\$a") -and (Test-Path "results\$b")) {
    & .venv\Scripts\python.exe paired.py "results\$a" "results\$b" 2>&1 | ForEach-Object { "   $_" }
  } else { "   (no paired comparison: results\$a or results\$b is missing)" }
}

"$(Get-Date -Format s) $n : extra arguments [$($Extra -join ' ')], $Steps steps$(if ($Init) { ", on from $Init" }), compared with $Against; waiting for the older queues$(if ($After) { ', ' + ($After -join ', ') }) and every other Needle job"
$idle = 0
while ($idle -lt 2) {
  if (Stopped) { "STOP file found, ending"; exit }
  if (Busy) { $idle = 0 } else { $idle++ }
  Start-Sleep -Seconds 60
}
if ($Init -and -not (Test-Path "runs\$Init\best.pt")) { "runs\$Init\best.pt does not exist: nothing to train on"; exit }
$m = Get-CimInstance Win32_PerfFormattedData_PerfOS_Memory
$room = ($m.CommitLimit - $m.CommittedBytes) / 1GB
"$(Get-Date -Format s) nothing else is running; room for $([math]::Round($room)) GB more committed memory"
if ($room -lt 30) { "less than 30 GB of commit room: something else is using the memory, not starting"; exit }

if (Stopped) { "STOP file found, ending"; exit }
$stems = @('--stems','F:\needle\data\stems\moisesdb','--stems','F:\needle\data\stems\mshoxx','--stems','F:\needle\data\stems\ni','--stems','F:\needle\data\stems\packs','--stems','F:\needle\data\stems\wavgrind')
$a = @('--arch','bs') + $Extra + @('--batch','8','--steps',"$Steps",'--patience','100','--stems-share','0.4','--teacher','E:\needle-teacher\luke_v2','--teacher','E:\needle-teacher\plainset2\jamendo_v2','--teacher','B:\needle-teacher\jamendo_v3','--teacher-share','0.4','--teacher-seconds','60','--p-speed','0.3','--speed-change','0.06','--mr-weight','0.5','--env-weight','0.005','--cache','H:\needle-cache','--cache','F:\needle-cache2','--gpu-mem','0.75') + $stems
if (Test-Path "runs\$n\last.pt") { $a += '--resume' } elseif ($Init) { $a += @('--init',"runs\$Init\best.pt") }
& F:\needle\run_train.ps1 -Name $n -TrainArgs $a *>&1 | Out-File -Append -Encoding utf8 "F:\needle\logs\$n.log"
if (-not (Test-Path "runs\$n\best.pt")) { "no checkpoint: the training failed, see F:\needle\logs\$n.log"; exit }
$fin = Select-String -Path "F:\needle\logs\$n.log" -Pattern 'finished at step ([\d,]+)' -ErrorAction SilentlyContinue | Select-Object -Last 1
$reached = if ($fin) { [int]($fin.Matches[0].Groups[1].Value -replace ',', '') } else { 0 }
if ($reached -lt $Steps) { "$n reached step $reached of ${Steps}: start this queue again to continue it; nothing is evaluated yet"; exit }
"$(Get-Date -Format s) trained. Validation (step, hours, drums, bass, other, vocals, avg, best so far, then the averaged copies):"
Get-Content "runs\$n\val.csv" | Select-Object -Last 3 | ForEach-Object { "   $_" }

# Which checkpoints: the trained weights' own best for the comparisons; best.pt (may be an averaged copy) ships.
$raw = if (Test-Path "runs\$n\best_raw.pt") { "runs\$n\best_raw.pt" } else { "runs\$n\best.pt" }
$ship = "runs\$n\best.pt"
$kind = ((& .venv\Scripts\python.exe ckinfo.py $ship 2>&1 | Select-Object -Last 1) -split ' ')[0]
$ema = $kind -like 'ema*'
"$(Get-Date -Format s) best.pt holds: $kind"
Ev $raw "${n}_holdout" @('--refs','E:\needle-teacher\holdout_house\jamendo_v2','--skip-inaudible-db','40')
Ev $raw "${n}_ibiza" @('--refs','E:\needle-teacher\ibiza_test_v2','--skip-inaudible-db','40')
Ev $raw $n @('--musdb','F:\needle\data\musdb18hq')
Ev $raw $n @('--refs','F:\needle\data\stems\moisesdb_test')
Ev $raw $n @('--refs','F:\needle\data\stems\ni_test')
if ($ema) {
  Ev $ship "${n}_ema" @('--musdb','F:\needle\data\musdb18hq')
  Ev $ship "${n}_ema" @('--refs','F:\needle\data\stems\moisesdb_test')
}
"$(Get-Date -Format s) paired: $n minus $Against, song by song (trained weights)"
Paired "${Against}_test.json" "${n}_test.json"
Paired "${Against}_moisesdb_test.json" "${n}_moisesdb_test.json"
Paired "${Against}_ni_test.json" "${n}_ni_test.json"
Paired "${Against}_holdout_jamendo_v2.json" "${n}_holdout_jamendo_v2.json"
Paired "${Against}_ibiza_ibiza_test_v2.json" "${n}_ibiza_ibiza_test_v2.json"
if ($ema) {
  "$(Get-Date -Format s) paired: what averaging buys ($kind minus the trained weights of $n)"
  Paired "${n}_test.json" "${n}_ema_test.json"
  Paired "${n}_moisesdb_test.json" "${n}_ema_moisesdb_test.json"
}
if ($Against -ne $base) {
  "$(Get-Date -Format s) paired: $n minus $base (Needle 1)"
  Paired "${base}_test.json" "${n}_test.json"
  Paired "${base}_moisesdb_test.json" "${n}_moisesdb_test.json"
}
if (-not (Stopped)) {
  "$(Get-Date -Format s) drum hits of best.pt: house hold-out against the teacher, then MoisesDB against real stems"
  & .venv\Scripts\python.exe hit_scan.py "runs\$base\best.pt" "runs\$Against\best.pt" $ship 2>&1 | Select-Object -Last 3 | ForEach-Object { "   $_" }
  & .venv\Scripts\python.exe hit_scan2.py --compare F:\needle\stem_loops\hit_scan.json $Against $n 2>&1 | Select-Object -First 5 | ForEach-Object { "   $_" }
  & .venv\Scripts\python.exe hit_scan2.py --refs F:\needle\data\stems\moisesdb_test --out results\hits_moisesdb_test.json "runs\$base\best.pt" "runs\$Against\best.pt" $ship 2>&1 | Select-Object -Last 3 | ForEach-Object { "   $_" }
  & .venv\Scripts\python.exe hit_scan2.py --compare results\hits_moisesdb_test.json $Against $n 2>&1 | Select-Object -First 5 | ForEach-Object { "   $_" }
}
if (-not (Stopped)) {
  "$(Get-Date -Format s) int8 export of best.pt and Vela"
  $env:CUDA_VISIBLE_DEVICES = '-1'
  & .venv-export\Scripts\python.exe export_vela2.py --checkpoint $ship --audio F:\needle\data\musdb18hq\train --audio-glob "*/mixture.wav" --name $n 2>&1 | Where-Object { "$_" -match 'TensorFlow copy|^int8|^params|rror|Traceback' } | ForEach-Object { "   $_" }
  "$(Get-Date -Format s) int8 SDR on the MoisesDB test set"
  $float = if ($ema) { "results\${n}_ema_moisesdb_test.json" } else { "results\${n}_moisesdb_test.json" }
  & .venv-export\Scripts\python.exe eval_tflite.py --tflite "build\$n\needle_int8.tflite" --refs F:\needle\data\stems\moisesdb_test --float $float --workers 4 2>&1 | Where-Object { "$_" -match 'SDR drums|quantisation|rror|Traceback' } | ForEach-Object { "   $_" }
  Remove-Item Env:CUDA_VISIBLE_DEVICES
}
"N2X QUEUE DONE ($n) $(Get-Date -Format s)"
