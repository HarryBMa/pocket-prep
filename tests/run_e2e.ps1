cd C:\Users\Harry\source\pocket-prep; Remove-Item -Recurse -Force data\e2e, data\export_test -ErrorAction SilentlyContinue; .\.venv\Scripts\python.exe -u tests\e2e.py data\sorry.flac data\export_test --stems *> data\e2e.log; "EXIT $LASTEXITCODE" | Out-File -Append data\e2e.log


