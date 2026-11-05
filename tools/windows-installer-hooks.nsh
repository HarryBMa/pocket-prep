; Tauri NSIS hooks: put the runtime DLLs next to pocket-prep.exe.
; tools/prepare-windows-runtime.ps1 copies them into target/redist/ before bundling.
; (Listed here rather than as Tauri resources, which are checked before that script runs.)

; Resolved now, while this file is being included: inside a macro, __FILEDIR__ would be
; the folder of the generated installer script instead.
!define PP_REDIST "${__FILEDIR__}\..\target\redist"

!macro NSIS_HOOK_POSTINSTALL
  SetOutPath "$INSTDIR"
  File "${PP_REDIST}\DirectML.dll"
  File "${PP_REDIST}\msvcp140.dll"
  File "${PP_REDIST}\msvcp140_1.dll"
  File "${PP_REDIST}\vcruntime140.dll"
  File "${PP_REDIST}\vcruntime140_1.dll"
!macroend

!macro NSIS_HOOK_POSTUNINSTALL
  Delete "$INSTDIR\DirectML.dll"
  Delete "$INSTDIR\msvcp140.dll"
  Delete "$INSTDIR\msvcp140_1.dll"
  Delete "$INSTDIR\vcruntime140.dll"
  Delete "$INSTDIR\vcruntime140_1.dll"
!macroend
