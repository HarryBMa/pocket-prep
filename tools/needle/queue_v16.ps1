# v16 (12 Nov 2026): frontier + drum hit-envelope loss, now with teacher pass 3 (913 more Jamendo tracks on B:).
# Same recipe as b20_jam2_frontier_env (v15); only the teacher data grows (1,416 -> 2,329 tracks).
# New decoded arrays go to F:\needle-cache2 (H: is full); existing ones are read from H:\needle-cache.
$ErrorActionPreference = 'Continue'
Set-Location F:\needle\pocket-prep\tools\needle
$env:PYTHONUNBUFFERED = '1'
function Ev($ck, $name, $set, $extra) {
  "$(Get-Date -Format s) eval $name"
  & .venv\Scripts\python.exe eval.py --checkpoint "runs\$ck\best.pt" @set --name $name --workers 3 --gpu-mem 0.5 @extra 2>&1 | Where-Object { $_ -match 'SDR|rror|Traceback' } | ForEach-Object { "   $_" }
}
New-Item -ItemType Directory -Force F:\needle-cache2 | Out-Null
if (Test-Path F:\needle\STOP_TRAIN) { "STOP_TRAIN, ending"; exit }
$stems = @('--stems','F:\needle\data\stems\moisesdb','--stems','F:\needle\data\stems\mshoxx','--stems','F:\needle\data\stems\ni','--stems','F:\needle\data\stems\packs','--stems','F:\needle\data\stems\wavgrind')
$a = @('--base','20','--frames','128','--batch','16','--steps','150000','--stems-share','0.4','--teacher','E:\needle-teacher\luke_v2','--teacher','E:\needle-teacher\plainset2\jamendo_v2','--teacher','B:\needle-teacher\jamendo_v3','--teacher-share','0.4','--teacher-seconds','60','--p-speed','0.3','--speed-change','0.06','--mr-weight','0.5','--cache','H:\needle-cache','--cache','F:\needle-cache2','--gpu-mem','0.6') + $stems + @('--coord','--harmonics','2','3','4','--taps','3','--env-weight','0.005')
$n = 'b20_jam3_frontier_env'
if (Test-Path "runs\$n\last.pt") { $a += '--resume' }
& F:\needle\run_train.ps1 -Name $n -TrainArgs $a *>&1 | Out-File -Append -Encoding utf8 "F:\needle\logs\$n.log"
$hold = @('--refs','E:\needle-teacher\holdout_house\jamendo_v2','--skip-inaudible-db','40')
Ev $n "${n}_holdout" $hold @()
Ev $n "${n}_ibiza" @('--refs','E:\needle-teacher\ibiza_test_v2','--skip-inaudible-db','40') @()
Ev $n $n @('--musdb','F:\needle\data\musdb18hq') @()
Ev $n $n @('--refs','F:\needle\data\stems\moisesdb_test') @()
Ev $n $n @('--refs','F:\needle\data\stems\ni_test') @()
"$(Get-Date -Format s) hit_scan $n"
& .venv\Scripts\python.exe hit_scan.py runs\b20_jam2_frontier\best.pt runs\b20_jam2_frontier_env\best.pt "runs\$n\best.pt" 2>&1 | Select-Object -Last 3 | ForEach-Object { "   $_" }
"V16 QUEUE DONE $(Get-Date -Format s)"
