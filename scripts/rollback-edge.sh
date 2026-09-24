#!/usr/bin/env bash
#
# rollback-edge.sh — points /opt/aioros-bodega/current at another already
# verified release and restarts the service. Without a release, lists them.
#
# Usage:
#   ./scripts/rollback-edge.sh user@jetson-ip            # list
#   ./scripts/rollback-edge.sh user@jetson-ip <release>  # activate

set -uo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$REPO_ROOT/scripts/lib.sh"

HOST="${1:-${AIOROS_BODEGA_JETSON_HOST:-}}"
RELEASE="${2:-}"
BASE=/opt/aioros-bodega
SERVICE=aioros-bodega-forecast.service
[[ "$HOST" == *@* ]] || { usage_from_header "$0"; exit 1; }

if [ -z "$RELEASE" ]; then
  ssh "$HOST" "cd '$BASE/releases' && cur=\$(readlink '$BASE/current' | xargs -r basename); \
    for r in \$(ls -1 | sort); do m=' '; [ \"\$r\" = \"\$cur\" ] && m='*'; \
    v=\$([ -f \"\$r/.aioros-release-ready\" ] && echo verified || echo 'NOT VERIFIED'); echo \"\$m \$r (\$v)\"; done"
  exit 0
fi

ssh "$HOST" "test -s '$BASE/releases/$RELEASE/.aioros-release-ready'" \
  || { err "releases/$RELEASE does not exist or did not pass deploy verification"; exit 1; }
PREVIOUS="$(current_release "$HOST" "$BASE")"
log "current: ${PREVIOUS:-none} -> $RELEASE"
ssh "$HOST" "ln -sfn '$BASE/releases/$RELEASE' '$BASE/current' \
  && printf '%s\t%s\t%s\n' \"\$(date -u +%FT%TZ)\" '$RELEASE' '$PREVIOUS' >> '$BASE/activation-history.tsv' \
  && sudo -n systemctl restart $SERVICE && sleep 20 && systemctl is-active --quiet $SERVICE" \
  || { err "rollback failed; see: ssh $HOST journalctl -u $SERVICE -n 50"; exit 1; }
ok "$SERVICE active on releases/$RELEASE"
