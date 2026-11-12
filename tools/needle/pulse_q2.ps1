# Pulse round 2: Raveform human labels (prep -> fine-tune w24 -> test-set eval incl. bar-fit downbeats)
$ErrorActionPreference = "Continue"
$env:PYTHONIOENCODING = "utf-8"
Set-Location F:\needle\pocket-prep\tools\needle
$py = ".venv\Scripts\python.exe"
$log = "F:\needle\logs\pulse_q2.log"
"$(Get-Date -Format s) prep" | Out-File $log -Append
& $py pulse_rave.py prep *>> $log
"$(Get-Date -Format s) eval before fine-tune (p1 only)" | Out-File $log -Append
& $py pulse_rave.py eval --models p1_w24 p1_w32 *>> $log
"$(Get-Date -Format s) train p2_w24" | Out-File $log -Append
& $py pulse_rave.py train --name p2_w24 *> F:\needle\logs\pulse_p2_w24.log
"$(Get-Date -Format s) eval" | Out-File $log -Append
& $py pulse_rave.py eval --models p1_w24 p2_w24 *>> $log
"PULSE Q2 DONE $(Get-Date -Format s)" | Out-File $log -Append
