# RoFormer vocals + htdemucs_ft (1 shift) on the Jamendo download, house first. Resumable.
$env:PYTHONIOENCODING = 'utf-8'
Set-Location F:\needle\pocket-prep\tools\needle
foreach ($tag in 'house','deephouse','techno','progressivehouse','electronic','techhouse') {
  if (Test-Path F:\needle\STOP_TEACHER) { "STOP_TEACHER, ending"; exit }
  "$(Get-Date -Format s) teacher jamendo/$tag"
  & F:\needle\.venv-sep\Scripts\python.exe teacher2.py --input "E:\needle-music\jamendo\$tag" --glob '*.*' --shifts 1 --out E:\needle-teacher\jamendo_v2 2>&1 | Where-Object { $_ -match '^\[\d+/\d+\].*(00|25|50|75)\]|tracks to do|FAILED|rror' } | ForEach-Object { "$_" }
}
"=== JAMENDO TEACHER DONE $(Get-Date -Format s)"
