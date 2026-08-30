#!/bin/sh
set -eu

# Persistent GNOME user-session settings. They disable automatic idle power
# transitions without simulating input and without disabling manual shutdown.
gsettings set org.gnome.desktop.session idle-delay 'uint32 0'
gsettings set org.gnome.settings-daemon.plugins.power idle-dim false
gsettings set org.gnome.settings-daemon.plugins.power sleep-inactive-ac-type 'nothing'
gsettings set org.gnome.settings-daemon.plugins.power sleep-inactive-ac-timeout 0
gsettings set org.gnome.settings-daemon.plugins.power sleep-inactive-battery-type 'nothing'
gsettings set org.gnome.settings-daemon.plugins.power sleep-inactive-battery-timeout 0

printf '%s\n' "GNOME automatic blank/dim/suspend disabled for the current user."
printf 'idle-delay='; gsettings get org.gnome.desktop.session idle-delay
printf 'idle-dim='; gsettings get org.gnome.settings-daemon.plugins.power idle-dim
printf 'ac-action='; gsettings get org.gnome.settings-daemon.plugins.power sleep-inactive-ac-type
printf 'battery-action='; gsettings get org.gnome.settings-daemon.plugins.power sleep-inactive-battery-type
