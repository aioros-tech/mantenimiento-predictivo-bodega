#!/usr/bin/env bash
#
# deploy-edge.sh — deploys the forecasting service (TimesFM 2.5 + SLM) to the
# Jetson Orin Nano, as described in docs/DEPLOY-JETSON.md. Same layout as
# aioros-beta-ai/scripts/deploy-edge.sh:
#
#   <base>/releases/<release>/   immutable: edge/ + src/ + data/
#   <base>/current -> releases/<release>   only moves once everything verified
#   <base>/venv/                 shared, exact versions from requirements-edge.txt
#   <base>/models/<repo>@<rev>/  weights pinned by revision (edge/models.yaml)
#
# Usage:
#   ./scripts/deploy-edge.sh user@jetson-ip [options]
#
# Options:
#   --release NAME    unique release name (default: UTC timestamp)
#   --no-activate     copy, install and verify, but don't move `current`
#
# Environment:
#   AIOROS_BODEGA_JETSON_HOST   user@ip (alternative to the first argument)
#
# Prerequisite: run ./scripts/bootstrap-jetson.sh ONCE (sudo).

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$REPO_ROOT/scripts/lib.sh"

HOST="${AIOROS_BODEGA_JETSON_HOST:-}"
NO_ACTIVATE=0
RELEASE="$(date -u +%Y%m%dT%H%M%SZ)-$$"
while [ $# -gt 0 ]; do
  case "$1" in
    --release)     RELEASE="${2:?--release needs a name}"; shift ;;
    --no-activate) NO_ACTIVATE=1 ;;
    -h|--help)     usage_from_header "$0"; exit 0 ;;
    *@*)           HOST="$1" ;;
    *)             err "unknown argument: $1"; exit 1 ;;
  esac
  shift
done
[ -n "$HOST" ] || { err "missing Jetson host (user@ip)"; usage_from_header "$0"; exit 1; }
[[ "$RELEASE" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$ ]] || { err "invalid release name: $RELEASE"; exit 1; }

BASE=/opt/aioros-bodega
DEST="$BASE/releases/$RELEASE"
SERVICE=aioros-bodega-forecast.service
EDGE_FILES=(service.py download_models.py units.yaml models.yaml requirements-edge.txt "$SERVICE")
SRC_FILES=(reporte_slm.py generar_datos_dashboard.py construir_dashboard.py plantilla_dashboard.html)
DATA_FILES=(anomaly-free.csv)
WAS_ACTIVE=0

check_local_files() {
  log "checking local files"
  local f
  for f in "${EDGE_FILES[@]/#/edge/}" "${SRC_FILES[@]/#/src/}" "${DATA_FILES[@]/#/data/}"; do
    [ -s "$REPO_ROOT/$f" ] || { err "missing $f"; exit 1; }
  done
  ok "edge/ + src/ + data/ ($(($(wc -l < "$REPO_ROOT/data/anomaly-free.csv") - 1)) replay readings)"
}

copy_release() {
  log "release '$RELEASE' -> $HOST:$DEST"
  ssh "$HOST" "test -w '$BASE/releases'" || {
    err "$BASE/releases missing or not writable: run ./scripts/bootstrap-jetson.sh $HOST first"
    exit 1
  }
  ssh "$HOST" "mkdir '$DEST' && mkdir '$DEST/edge' '$DEST/src' '$DEST/data'" || {
    err "release '$RELEASE' already exists; releases are immutable"; exit 1; }
  (cd "$REPO_ROOT" \
    && rsync -a --partial "${EDGE_FILES[@]/#/edge/}" "$HOST:$DEST/edge/" \
    && rsync -a --partial "${SRC_FILES[@]/#/src/}" "$HOST:$DEST/src/" \
    && rsync -a --partial "${DATA_FILES[@]/#/data/}" "$HOST:$DEST/data/") \
    || { err "rsync failed"; exit 1; }
  ssh "$HOST" "cd '$DEST' && find edge src data -type f -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS" \
    || { err "could not write SHA256SUMS"; exit 1; }
  ok "copied (git $(git -C "$REPO_ROOT" rev-parse --short HEAD 2>/dev/null || echo '?'))"
}

install_venv() {
  log "venv $BASE/venv with the exact versions in requirements-edge.txt (first time downloads torch, ~3 min)"
  ssh "$HOST" bash -s -- "$BASE" "$DEST" <<'REMOTE' || { err "could not set up the venv"; exit 1; }
set -euo pipefail
base="$1"; dest="$2"
# Rebuild the venv from scratch when requirements-edge.txt changes: pip does not
# reinstall a package of the same version even if the wheel URL changed.
req_sha=$(sha256sum "$dest/edge/requirements-edge.txt" | cut -d' ' -f1)
if [ "$(cat "$base/venv/.requirements.sha256" 2>/dev/null)" != "$req_sha" ]; then
  rm -rf "$base/venv"
  python3 -m venv "$base/venv"
  "$base/venv/bin/pip" install -q --upgrade pip
  "$base/venv/bin/pip" install -q -r "$dest/edge/requirements-edge.txt"
  echo "$req_sha" > "$base/venv/.requirements.sha256"
fi
"$base/venv/bin/python" -W ignore -c "import sys, torch, timesfm, transformers; print('  python', sys.version.split()[0], '| torch', torch.__version__, '| cuda', torch.cuda.is_available(), '| transformers', transformers.__version__)"
REMOTE
  ok "venv ready"
}

fetch_models() {
  log "weights pinned by revision in $BASE/models (first time ~3.8 GB)"
  ssh "$HOST" "cd '$DEST' && '$BASE/venv/bin/python' edge/download_models.py edge/models.yaml '$BASE/models'" \
    || { err "could not download the weights"; exit 1; }
  ok "weights ready"
}

systemctl_remote() { ssh "$HOST" "systemctl $1 --quiet $SERVICE"; }

verify_release() {
  # Smoke test BEFORE activating: one full cycle (TimesFM + SLM + dashboard)
  # for one unit, with local weights and no network, requiring the GPU. The
  # running service is stopped during the test: two copies of the models don't
  # fit in 8 GB next to Alpha/Beta. If the test fails, it is started again.
  if systemctl_remote is-active; then WAS_ACTIVE=1; fi
  log "verifying: one service cycle on the Jetson's GPU"
  if [ "$WAS_ACTIVE" -eq 1 ]; then
    warn "stopping $SERVICE during the test (memory)"
    ssh "$HOST" "sudo -n systemctl stop $SERVICE" || { err "could not stop $SERVICE"; exit 1; }
  fi
  if ! ssh "$HOST" bash -s -- "$BASE" "$DEST" <<'REMOTE'; then
set -euo pipefail
base="$1"; dest="$2"
cd "$dest"
HF_HUB_OFFLINE=1 "$base/venv/bin/python" edge/service.py --models-dir "$base/models" \
  --output smoke --cycles 1 --max-units 1 --interval 0 --require-cuda
"$base/venv/bin/python" - <<'EOF'
import json
d = json.load(open("smoke/datos_dashboard.json"))
assert d["dispositivo"] == "cuda", d["dispositivo"]
e = d["equipos"][0]
assert e["reporte"].strip(), "empty report"
print(f"  {e['nombre']}: {e['estado']}, guardrail {'OK' if not e['guardarrail'] else e['guardarrail']}")
EOF
test -s smoke/dashboard/index.html
printf 'release_ready_at=%s\n' "$(date -u +%FT%TZ)" > .aioros-release-ready
REMOTE
    err "the service failed on the Jetson; the release is not activated"
    [ "$WAS_ACTIVE" -eq 1 ] && ssh "$HOST" "sudo -n systemctl start $SERVICE" && warn "$SERVICE resumed on the previous release"
    exit 1
  fi
  ok "release verified"
}

# Wait for the service to load the models (is-active only says the process
# started; loading takes ~15 s to 1 min and is where it would fail on memory).
wait_models_loaded() {
  local since="$1" i
  for i in $(seq 1 36); do
    sleep 5
    systemctl_remote is-active || return 1
    ssh "$HOST" "journalctl -u $SERVICE --since '@$since' -o cat 2>/dev/null | grep -q 'models ready'" && return 0
  done
  return 1
}

activate_release() {
  if [ "$NO_ACTIVATE" -eq 1 ]; then
    warn "--no-activate: release '$RELEASE' is ready but not active"
    [ "$WAS_ACTIVE" -eq 1 ] && ssh "$HOST" "sudo -n systemctl start $SERVICE" && warn "$SERVICE resumed on the previous release"
    echo "  Activate it later: ./scripts/rollback-edge.sh $HOST $RELEASE"
    return
  fi
  local previous since
  previous="$(current_release "$HOST" "$BASE")"
  log "activating '$RELEASE' (previous: ${previous:-none})"
  ssh "$HOST" "ln -sfn '$DEST' '$BASE/current' \
    && printf '%s\t%s\t%s\n' \"\$(date -u +%FT%TZ)\" '$RELEASE' '$previous' >> '$BASE/activation-history.tsv'" \
    || { err "could not move current"; exit 1; }
  since="$(ssh "$HOST" date +%s)"
  if ssh "$HOST" "sudo -n systemctl restart $SERVICE" && wait_models_loaded "$since"; then
    ok "$SERVICE active on releases/$RELEASE, models loaded"
    return
  fi
  err "$SERVICE did not start with the new release; restoring the previous one"
  if [ -n "$previous" ]; then
    ssh "$HOST" "ln -sfn '$BASE/releases/$previous' '$BASE/current' && sudo -n systemctl restart $SERVICE" || true
  fi
  echo "  See why: ssh $HOST journalctl -u $SERVICE -n 50"
  exit 1
}

check_local_files
copy_release
install_venv
fetch_models
verify_release
activate_release
echo
ok "deployment complete on $HOST:$DEST"
echo "  dashboard: http://${HOST#*@}:8095/"
echo "  logs:      ssh $HOST journalctl -u $SERVICE -f"
echo "  reports:   ssh $HOST tail -f $BASE/output/reports.jsonl"
echo "  rollback:  ./scripts/rollback-edge.sh $HOST <previous-release>"
