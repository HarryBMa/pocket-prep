# Pulse round 3: ensemble teacher (final0-2) + bought music, synthetic loop tracks, from-scratch w24 with
# 6 levels (~12 s context) vs 5 levels on the same data; Raveform test-set grid eval; int8 + Vela of the best.
$ErrorActionPreference = "Continue"
$env:PYTHONIOENCODING = "utf-8"
Set-Location F:\needle\pocket-prep\tools\needle
$py = ".venv\Scripts\python.exe"
$log = "F:\needle\logs\pulse_q3.log"
"$(Get-Date -Format s) ensemble teacher" | Out-File $log -Append
& $py pulse_teacher_ens.py *> F:\needle\logs\pulse_teacher_ens.log
"$(Get-Date -Format s) loop tracks" | Out-File $log -Append
& $py pulse_loops.py --n 1500 *> F:\needle\logs\pulse_loops.log
foreach ($lv in 6, 5) {
    "$(Get-Date -Format s) train p3_w24_l$lv" | Out-File $log -Append
    & $py pulse_rave.py train --init "" --width 24 --levels $lv --teacher F:\needle\beat_teacher_ens --loop-frac 0.2 `
        --steps 60000 --lr 2e-3 --name "p3_w24_l$lv" *> "F:\needle\logs\pulse_p3_w24_l$lv.log"
}
foreach ($m in "p2_w24", "p3_w24_l5", "p3_w24_l6") {
    "$(Get-Date -Format s) grid eval $m" | Out-File $log -Append
    & $py pulse_grid.py --model $m *>> $log
}
"$(Get-Date -Format s) int8 + vela p3_w24_l6" | Out-File $log -Append
$env:CUDA_VISIBLE_DEVICES = "-1"
& .venv-export\Scripts\python.exe pulse_export.py --model p3_w24_l6 --modes int8 *> F:\needle\logs\pulse_export_p3.log
Select-String F:\needle\logs\pulse_export_p3.log -Pattern '^int8' | ForEach-Object Line | Out-File $log -Append
"PULSE Q3 DONE $(Get-Date -Format s)" | Out-File $log -Append
