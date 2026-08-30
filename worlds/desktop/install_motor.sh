#!/usr/bin/env bash
set -euo pipefail

UID_NUM="$(id -u)"
GID_NUM="$(id -g)"
SOCKET="/run/heiwa-ydotool.sock"
SERVICE="/etc/systemd/system/heiwa-ydotoold.service"

if ! command -v ydotoold >/dev/null 2>&1; then
  echo "ERROR: ydotoold is missing" >&2
  exit 2
fi

# Stop the packaged user daemon. On this machine it cannot open /dev/uinput,
# so it stays "active" without producing a usable socket.
systemctl --user disable --now ydotool.service >/dev/null 2>&1 || true
systemctl --user disable --now ydotoold.service >/dev/null 2>&1 || true

sudo rm -f \
  /run/user/"${UID_NUM}"/.ydotool_socket \
  /tmp/.ydotool_socket \
  "${SOCKET}"

sudo tee "${SERVICE}" >/dev/null <<EOF
[Unit]
Description=Heiwa ydotoold motor daemon
After=multi-user.target

[Service]
Type=simple
ExecStart=/usr/bin/ydotoold --socket-path=${SOCKET} --socket-perm=0660 --socket-own=${UID_NUM}:${GID_NUM}
Restart=on-failure
RestartSec=1

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable heiwa-ydotoold.service
sudo systemctl restart heiwa-ydotoold.service

for _ in $(seq 1 30); do
  if [[ -S "${SOCKET}" ]]; then
    break
  fi
  sleep 0.1
done

echo "HEIWA YDOTOOL MOTOR INSTALL"
echo "service_state: $(sudo systemctl is-active heiwa-ydotoold.service || true)"
echo "socket: ${SOCKET}"
echo "socket_exists: $([[ -S "${SOCKET}" ]] && echo true || echo false)"
if [[ -S "${SOCKET}" ]]; then
  stat -c 'socket_owner: %U:%G' "${SOCKET}"
  stat -c 'socket_mode: %a' "${SOCKET}"
else
  echo "journal:"
  sudo journalctl -u heiwa-ydotoold.service -n 30 --no-pager || true
  exit 3
fi

echo "HEIWA YDOTOOL MOTOR INSTALL: OK"
