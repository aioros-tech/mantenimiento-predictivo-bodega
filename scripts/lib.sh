#!/usr/bin/env bash
# lib.sh — output helpers shared by scripts/*.sh (same format as
# aioros-beta-ai/scripts/lib.sh). Sourced, not executed.

log()  { printf '\033[1;34m==>\033[0m %s\n' "$*"; }
ok()   { printf '\033[1;32m  ok\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m  !!\033[0m %s\n' "$*" >&2; }
err()  { printf '\033[1;31m  xx\033[0m %s\n' "$*" >&2; }

# Print the comment block after the shebang as usage help.
usage_from_header() {
  awk 'NR==1{next} !/^#/{exit} {sub(/^# ?/,""); print}' "$1"
}

# current_release <host> <base> — name of the release <base>/current points to,
# or empty if there is no symlink or it points to something that doesn't exist.
current_release() {
  ssh "$1" "t=\$(readlink '$2/current' 2>/dev/null) && [ -d \"\$t\" ] && basename \"\$t\"" || true
}
