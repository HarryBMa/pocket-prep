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
# With `tauri build --target <triple>` the build lands in target\<triple>\release.
$release = 'target\release'
if ($env:TAURI_ENV_TARGET_TRIPLE -and (Test-Path "target\$env:TAURI_ENV_TARGET_TRIPLE\release\DirectML.dll")) {
    $release = "target\$env:TAURI_ENV_TARGET_TRIPLE\release"
}
$dml = Get-Item "$release\DirectML.dll" -ErrorAction SilentlyContinue
if (-not $dml) {
    # ort's rc.9 Windows build links DirectML statically-by-import and does not always leave a DLL behind (a CI build with
    # --target did not: nothing under target\ or in ort's cache). Say what is there, then fetch Microsoft's redistributable.
    Write-Host "DirectML.dll is not in $release. DLLs outside deps\ under target\:"
    Get-ChildItem 'target' -Recurse -Filter '*.dll' -ErrorAction SilentlyContinue | Where-Object { $_.FullName -notmatch '\\deps\\' } |
        Select-Object -First 30 -ExpandProperty FullName | ForEach-Object { Write-Host "  $_" }
    $dml = Get-ChildItem 'target' -Recurse -Filter 'DirectML.dll' -ErrorAction SilentlyContinue | Select-Object -First 1
    if (-not $dml) {
        # The version ONNX Runtime 1.20's DirectML provider is built against (a guess from its documentation, not checked here).
        $v = '1.15.4'
        try {
            $zip = Join-Path $env:TEMP "directml-$v.zip"
            Invoke-WebRequest "https://www.nuget.org/api/v2/package/Microsoft.AI.DirectML/$v" -OutFile $zip -UseBasicParsing
            Expand-Archive $zip (Join-Path $env:TEMP "directml-$v") -Force
            $dml = Get-Item (Join-Path $env:TEMP "directml-$v\bin\x64-win\DirectML.dll")
            Write-Host "Using DirectML $v from NuGet: $($dml.FullName)"
        } catch { Write-Warning "Could not fetch DirectML $v from NuGet: $_" }
    }
}
if ($dml) {
    if ($dml.LinkType) { $dml = Get-Item $dml.Target }
    Copy-Item $dml.FullName (Join-Path $out 'DirectML.dll') -Force
} else {
    # The installer is still made (windows-installer-hooks.nsh adds the file only if it is there); the app then uses the
    # system's DirectML.dll, which may be too old for GPU stems, and says so when it falls back.
    Write-Warning "No DirectML.dll: building the installer without it."
    Remove-Item (Join-Path $out 'DirectML.dll') -ErrorAction SilentlyContinue
}

# Visual C++ runtime, from the newest Visual Studio / Build Tools install.
$vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
$vs = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
$crt = Get-ChildItem "$vs\VC\Redist\MSVC\*\x64\Microsoft.VC14*.CRT" -Directory |
    Sort-Object { [version]($_.Parent.Parent.Name) } | Select-Object -Last 1
foreach ($f in 'msvcp140.dll', 'msvcp140_1.dll', 'vcruntime140.dll', 'vcruntime140_1.dll') {
    Copy-Item (Join-Path $crt.FullName $f) (Join-Path $out $f) -Force
}
Get-ChildItem $out | Format-Table Name, Length
