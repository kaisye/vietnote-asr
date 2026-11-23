#!/usr/bin/env bash
# Runs on the VM as root (upload.sh calls it): installs Python deps and Caddy, installs the code, restarts.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"; source "$HERE/config.sh"
export DEBIAN_FRONTEND=noninteractive

if ! command -v caddy >/dev/null; then
  apt-get update -q
  apt-get install -y -q python3-venv python3-pip debian-keyring debian-archive-keyring apt-transport-https curl gpg
  curl -1sLf https://dl.cloudsmith.io/public/caddy/stable/gpg.key | gpg --dearmor -o /usr/share/keyrings/caddy-stable-archive-keyring.gpg
  curl -1sLf https://dl.cloudsmith.io/public/caddy/stable/debian.deb.txt > /etc/apt/sources.list.d/caddy-stable.list
  apt-get update -q && apt-get install -y -q caddy
fi
id asr >/dev/null 2>&1 || useradd --system --home "$REMOTE_DIR" --shell /usr/sbin/nologin asr

mkdir -p "$REMOTE_DIR"
rsync -a --delete --exclude venv "$HERE/" "$REMOTE_DIR/app/" 2>/dev/null || { rm -rf "$REMOTE_DIR/app"; cp -R "$HERE" "$REMOTE_DIR/app"; }
[ -d "$REMOTE_DIR/venv" ] || python3 -m venv "$REMOTE_DIR/venv"
"$REMOTE_DIR/venv/bin/pip" install -q --upgrade pip
"$REMOTE_DIR/venv/bin/pip" install -q -r "$REMOTE_DIR/app/requirements.txt"
chown -R asr:asr "$REMOTE_DIR"

{ [ -n "$EMAIL_FOR_TLS" ] && printf '{\n\temail %s\n}\n\n' "$EMAIL_FOR_TLS"; sed "s/__DOMAIN__/$DOMAIN/" "$REMOTE_DIR/app/Caddyfile"; } > /etc/caddy/Caddyfile
cp "$REMOTE_DIR/app/asr-demo.service" /etc/systemd/system/asr-demo.service
systemctl daemon-reload
systemctl enable --now asr-demo && systemctl restart asr-demo
systemctl reload caddy || systemctl restart caddy
for i in $(seq 1 60); do curl -sf localhost:8765/info >/dev/null && break; sleep 2; done
curl -s localhost:8765/info | head -c 300; echo
