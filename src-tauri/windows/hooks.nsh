; ModelGym replaces RackTop while keeping its local/remote data identity.
; These hooks execute only when the user runs the delivered installer.
Var ModelGymLegacyDisplayName
Var ModelGymLegacyInstallDir

!macro NSIS_HOOK_PREINSTALL
  ; The old binary shares ModelGym's application identity even when it was
  ; unpacked manually and has no uninstall record. Do not launch into that copy.
  !insertmacro CheckIfAppIsRunning "RackTop.exe" "RackTop"
!macroend

!macro ModelGymMigrateShortcut OLDLINK NEWLINK NEWDIRECTORY
  ; Never delete a same-named shortcut pointing to an unrelated application.
  !insertmacro IsShortcutTarget "${OLDLINK}" "$ModelGymLegacyInstallDir\RackTop.exe"
  Pop $0
  ${If} $0 = 1
    !insertmacro UnpinShortcut "${OLDLINK}"
    Delete "${OLDLINK}"
    CreateDirectory "${NEWDIRECTORY}"
    ${IfNot} ${FileExists} "${NEWLINK}"
      CreateShortcut "${NEWLINK}" "$INSTDIR\${MAINBINARYNAME}.exe"
      !insertmacro SetLnkAppUserModelId "${NEWLINK}"
    ${EndIf}
  ${EndIf}
!macroend

!macro NSIS_HOOK_POSTINSTALL
  ${If} $ModelGymLegacyDisplayName == "RackTop"
  ${AndIf} $ModelGymLegacyInstallDir != ""
    !insertmacro ModelGymMigrateShortcut "$DESKTOP\RackTop.lnk" "$DESKTOP\${PRODUCTNAME}.lnk" "$DESKTOP"
    !insertmacro ModelGymMigrateShortcut "$SMPROGRAMS\RackTop\RackTop.lnk" "$SMPROGRAMS\$AppStartMenuFolder\${PRODUCTNAME}.lnk" "$SMPROGRAMS\$AppStartMenuFolder"
    !insertmacro ModelGymMigrateShortcut "$SMPROGRAMS\RackTop.lnk" "$SMPROGRAMS\$AppStartMenuFolder\${PRODUCTNAME}.lnk" "$SMPROGRAMS\$AppStartMenuFolder"
    RMDir "$SMPROGRAMS\RackTop"
    ; Only the legacy executable is removed. SQLite, WebView data, SSH keys and
    ; user-created files are outside this operation and remain untouched.
    Delete "$ModelGymLegacyInstallDir\RackTop.exe"
  ${EndIf}
!macroend
