#!/bin/sh
set -eu

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
project_dir=$(CDPATH= cd -- "$script_dir/.." && pwd)
source_unit="$project_dir/systemd/heiwa.service"
source_overlay_dir="$project_dir/gnome_extension/heiwa-cursor-overlay@local"
user_unit_dir="$HOME/.config/systemd/user"
target_unit="$user_unit_dir/heiwa.service"
target_overlay_dir="$HOME/.local/share/gnome-shell/extensions/heiwa-cursor-overlay@local"

mkdir -p "$user_unit_dir"
install -m 0644 "$source_unit" "$target_unit"
mkdir -p "$target_overlay_dir"
install -m 0644 "$source_overlay_dir/metadata.json" "$target_overlay_dir/metadata.json"
install -m 0644 "$source_overlay_dir/extension.js" "$target_overlay_dir/extension.js"
install -m 0644 "$source_overlay_dir/stylesheet.css" "$target_overlay_dir/stylesheet.css"

session_names=""
for name in WAYLAND_DISPLAY DISPLAY XDG_RUNTIME_DIR DBUS_SESSION_BUS_ADDRESS AT_SPI_BUS_ADDRESS XAUTHORITY; do
    eval "value=\${$name-}"
    if [ -n "$value" ]; then session_names="$session_names $name"; fi
done
if [ -n "$session_names" ]; then
    # shellcheck disable=SC2086
    systemctl --user import-environment $session_names
fi
systemctl --user daemon-reload
systemctl --user enable heiwa.service

printf '%s\n' "Installed and enabled: $target_unit"
printf '%s\n' "Installed GNOME overlay: $target_overlay_dir"
printf '%s\n' "Start: systemctl --user start heiwa"
printf '%s\n' "Enable patterns after GNOME login: gnome-extensions enable heiwa-cursor-overlay@local"
printf '%s\n' "Hide patterns: gnome-extensions disable heiwa-cursor-overlay@local"
printf '%s\n' "Stop/emergency stop: systemctl --user stop heiwa"
