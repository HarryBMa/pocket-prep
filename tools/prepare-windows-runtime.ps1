# Copy the DLLs the Windows build needs next to the installed exe into target\redist\,
# where tools\windows-installer-hooks.nsh adds them to the installer. Tauri runs this as
# beforeBundleCommand (tauri.windows.conf.json), after the release build.
#
#   DirectML.dll                          ONNX Runtime's DirectML (GPU stems); the copy in
#                                         System32 is older than ONNX Runtime needs
#   msvcp140.dll, msvcp140_1.dll,         Visual C++ runtime for ONNX Runtime; app-local
#   vcruntime140.dll, vcruntime140_1.dll  deployment is allowed by Microsoft's redist terms
$ErrorActionPreference = 'Stop'
Set-Location (Join-Path $PSScriptRoot '..')
$out = 'target\redist'
New-Item -ItemType Directory -Force $out | Out-Null

# DirectML: the file ort downloaded with ONNX Runtime (target\release\DirectML.dll links to it).
$dml = Get-Item 'target\release\DirectML.dll'
if ($dml.LinkType) { $dml = Get-Item $dml.Target }
Copy-Item $dml.FullName (Join-Path $out 'DirectML.dll') -Force

# Visual C++ runtime, from the newest Visual Studio / Build Tools install.
$vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
$vs = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
$crt = Get-ChildItem "$vs\VC\Redist\MSVC\*\x64\Microsoft.VC14*.CRT" -Directory |
    Sort-Object { [version]($_.Parent.Parent.Name) } | Select-Object -Last 1
foreach ($f in 'msvcp140.dll', 'msvcp140_1.dll', 'vcruntime140.dll', 'vcruntime140_1.dll') {
    Copy-Item (Join-Path $crt.FullName $f) (Join-Path $out $f) -Force
}
Get-ChildItem $out | Format-Table Name, Length
