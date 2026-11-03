# ModelGym Windows packaging and RackTop upgrades

ModelGym replaces RackTop while preserving its existing data. `com.racktop.desktop`,
`racktop.sqlite`, the keyring service, browser storage keys, SSH key names and remote
`~/.racktop` paths deliberately remain compatible. They are internal identities, not
product branding. Do not rename them without an explicit migration.

Build with the project environment active:

```powershell
npm.cmd run bundle:windows
```

This produces an NSIS setup EXE and an MSI for Windows x64, with `ModelGym.exe` as
the application binary. Bundler tools are cached under `src-tauri/target/.tauri/`.
Release signing and upstream updater artifacts are disabled; ModelGym never checks
or installs updates from RackTop's publishing feed.

## Upgrade compatibility

- NSIS keeps the original `...\Uninstall\RackTop` and `Software\racktop\RackTop`
  registry identities. Its visible product name, binary and new shortcuts are
  ModelGym. This allows the existing directory and version to be discovered.
- Upgrade uninstalls always receive `/UPDATE`, which preserves application data
  even if the old uninstaller's data-removal checkbox were selected.
- The installer's legacy process check covers `RackTop.exe` as well as the normal
  `ModelGym.exe` check. Users must close the old app before replacing it.
- Legacy NSIS shortcuts are migrated only when their targets match the detected
  RackTop installation; same-named shortcuts targeting other apps are preserved.
- MSI pins the original RackTop upgrade code
  `22bdaaa4-514a-5723-ba40-e8a2c0881d81`, derived by the original Tauri bundler from
  `UUIDv5(DNS, "RackTop.exe.app.x64")`. Changing only the product name would otherwise
  generate a different upgrade code and leave two registered products.
- The MSI template retains the original install-location registry keys. Its renamed
  key executable uses a new stable component GUID, and exact legacy shortcut names
  are removed when Windows Installer detects an existing related MSI product.

Use the same installer family for upgrades: NSIS setup for an existing NSIS
installation, MSI for an existing MSI installation. A manually unpacked EXE has no
installer registration and its directory cannot be discovered automatically. It can
still open the preserved data after the old process is closed; do not delete the
application-data directories when replacing such a copy.

The two templates derive from the official Tauri CLI v2.11.4 templates. Source URLs
and the compatibility changes are documented in their headers. The files use the
upstream MIT license, reproduced in `TAURI_LICENSE_MIT`.

Tests in `src/utils/windowsUpgrade.test.js` guard data identity, installer upgrade
identity and update-source isolation. Installer compilation and actual upgrade
behavior must also be checked against an isolated fixture before delivery. Do not
run an upgrade test against a user's real running installation or database.
