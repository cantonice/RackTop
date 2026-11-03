#!/usr/bin/env bash
set -euo pipefail

[[ $(uname -s) == Linux ]] || { printf 'This installer is for Linux.\n' >&2; exit 1; }

source_image=$(realpath "${1:?Usage: bash scripts/install-appimage.sh /path/to/ModelGym.AppImage}")
project_root=$(cd "$(dirname "$0")/.." && pwd)
install_root="$HOME/.local/lib/racktop"
data_root="${XDG_DATA_HOME:-$HOME/.local/share}"
install -d "$install_root" "$data_root/applications" "$data_root/icons/hicolor/256x256/apps"
if [[ "$source_image" != "$install_root/RackTop.AppImage" ]]; then
  staged_image=$(mktemp "$install_root/.RackTop.AppImage.XXXXXX")
  trap 'rm -f "$staged_image"' EXIT
  install -m 755 "$source_image" "$staged_image"
  mv "$staged_image" "$install_root/RackTop.AppImage"
fi
install -m 644 "$project_root/src-tauri/icons/128x128@2x-linux.png" "$data_root/icons/hicolor/256x256/apps/racktop-appimage.png"
# Desktop Entry Exec has its own escaping rules, independent of the shell.
escaped_executable=${install_root//\\/\\\\}
escaped_executable=${escaped_executable//\"/\\\"}
escaped_executable=${escaped_executable//\$/\\\$}
escaped_executable=${escaped_executable//\`/\\\`}
escaped_executable=${escaped_executable//%/%%}
# Keep the existing install/desktop/icon identity and GTK class for upgrades.
desktop_file="$data_root/applications/racktop.desktop"
{
  printf '[Desktop Entry]\nType=Application\nName=ModelGym\n'
  printf 'Exec="%s/RackTop.AppImage"\n' "$escaped_executable"
  printf 'Icon=racktop-appimage\nStartupWMClass=Racktop\nTerminal=false\nCategories=Development;\n'
} > "$desktop_file"
rm -f "$data_root/applications/com.racktop.appimage.desktop"
if command -v update-desktop-database >/dev/null; then update-desktop-database "$data_root/applications"; fi
if command -v gtk-update-icon-cache >/dev/null; then gtk-update-icon-cache -f -t "$data_root/icons/hicolor" >/dev/null 2>&1 || true; fi
# Nautilus does not use an AppImage's embedded icon for the executable file.
# Apply per-file metadata rather than branding every AppImage MIME type.
if command -v gio >/dev/null; then
  for image in "$source_image" "$install_root/RackTop.AppImage"; do
    if ! gio set -t string "$image" metadata::custom-icon-name racktop-appimage; then
      printf 'Warning: file icon metadata could not be registered for %s\n' "$image" >&2
    fi
  done
fi
printf 'Installed AppImage: %s\nDesktop entry: %s\n' "$install_root/RackTop.AppImage" "$desktop_file"
