#!/bin/sh
set -eu

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
project_dir=$(CDPATH= cd -- "$script_dir/.." && pwd)
source_unit="$project_dir/systemd/heiwa-language.service"
user_unit_dir="$HOME/.config/systemd/user"
target_unit="$user_unit_dir/heiwa-language.service"

mkdir -p "$user_unit_dir"
install -m 0644 "$source_unit" "$target_unit"

session_names=""
for name in WAYLAND_DISPLAY DISPLAY XDG_RUNTIME_DIR DBUS_SESSION_BUS_ADDRESS AT_SPI_BUS_ADDRESS; do
    eval "value=\${$name-}"
    if [ -n "$value" ]; then
        session_names="$session_names $name"
    fi
done
if [ -n "$session_names" ]; then
    # Import only values already present in this graphical user session.
    # shellcheck disable=SC2086
    systemctl --user import-environment $session_names
fi
systemctl --user daemon-reload

printf '%s\n' "Installed: $target_unit"
printf '%s\n' "Start manually: systemctl --user start heiwa-language"
