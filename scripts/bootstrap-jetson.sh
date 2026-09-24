#!/usr/bin/env bash
#
# bootstrap-jetson.sh — ONE-TIME Jetson setup for the cold store forecasting
# service (see docs/DEPLOY-JETSON.md §3). It is the only step that uses sudo,
# so it asks for the Jetson password; run it in an interactive terminal.
# Afterwards, deploy-edge.sh and rollback-edge.sh never ask for it.
#
# It:
#   1. creates /opt/aioros-bodega and hands it to the Jetson user
#   2. installs edge/aioros-bodega-forecast.service and enables it
#   3. writes /etc/sudoers.d/aioros-bodega: allows ONLY start/stop/restart/
#      is-active of that service without a password (what deploy-edge.sh needs)
#   4. removes the previous unit name (aioros-bodega-pronostico), if present
#
# Usage:
#   ./scripts/bootstrap-jetson.sh user@jetson-ip
#
# Safe to re-run (idempotent); required whenever the .service file changes.

set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$REPO_ROOT/scripts/lib.sh"

HOST="${1:-${AIOROS_BODEGA_JETSON_HOST:-}}"
[ -n "$HOST" ] && [ "$HOST" != "-h" ] && [ "$HOST" != "--help" ] || { usage_from_header "$0"; exit 1; }
BASE=/opt/aioros-bodega
SERVICE=aioros-bodega-forecast.service
LEGACY_SERVICE=aioros-bodega-pronostico.service

log "copying the unit to $HOST"
scp -q "$REPO_ROOT/edge/$SERVICE" "$HOST:/tmp/$SERVICE"

log "setting up $BASE, systemd and sudoers on $HOST (asks for the Jetson password)"
# shellcheck disable=SC2087
ssh -t "$HOST" "bash -c '
set -euo pipefail
user=\$(id -un); group=\$(id -gn)
sudo mkdir -p $BASE/releases
sudo chown -R \$user:\$group $BASE
if [ -f /etc/systemd/system/$LEGACY_SERVICE ]; then
  sudo systemctl disable --now $LEGACY_SERVICE >/dev/null 2>&1 || true
  sudo rm -f /etc/systemd/system/$LEGACY_SERVICE
fi
sed \"s/__USER__/\$user/\" /tmp/$SERVICE | sudo tee /etc/systemd/system/$SERVICE >/dev/null
rm -f /tmp/$SERVICE
rule=\"\$user ALL=(root) NOPASSWD: /usr/bin/systemctl start $SERVICE, /usr/bin/systemctl stop $SERVICE, /usr/bin/systemctl restart $SERVICE, /usr/bin/systemctl is-active $SERVICE\"
echo \"\$rule\" | sudo tee /etc/sudoers.d/aioros-bodega >/dev/null
sudo chmod 440 /etc/sudoers.d/aioros-bodega
sudo visudo -cf /etc/sudoers.d/aioros-bodega >/dev/null
sudo systemctl daemon-reload
sudo systemctl enable $SERVICE >/dev/null 2>&1
'"
ok "Jetson ready: $BASE owned by the user, $SERVICE enabled (starts after the first deploy)"
echo "  Next step: ./scripts/deploy-edge.sh $HOST"
