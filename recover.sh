#!/usr/bin/env bash
# Restore source files from the latest backup (or a given backup directory).
# Usage: ./recover.sh [path/to/.fr_backup_YYYYMMDD_HHMMSS]
set -euo pipefail
WS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

SRC="${1:-$WS/.fr_backup_latest}"
if [[ ! -d "$SRC" ]]; then
  echo "No backup found at: $SRC" >&2
  echo "Run ./backup.sh first, or pass a backup directory." >&2
  exit 1
fi

echo "Restoring from $SRC"

restore() {
  local from=$1 to=$2
  if [[ -f "$from" ]]; then
    mkdir -p "$(dirname "$to")"
    cp -a "$from" "$to"
    echo "  restored $(basename "$to")"
  fi
}

for f in start.sh stop.sh lib.sh install.sh verify.sh backup.sh recover.sh context.md README.md; do
  restore "$SRC/$f" "$WS/$f"
done

for f in dashboard.py face_pipeline.py enroll.py recognize.py check_camera.py \
         benchmark.py compare_models.py g1_locomotion.py evaluate.py fetch_impostors.py \
         go2.py selftest.py tune_threshold.py verify_deploy.py; do
  restore "$SRC/scripts/$f" "$WS/scripts/$f"
done

chmod +x "$WS"/*.sh 2>/dev/null || true
echo "Done. Verify with: python3 -m py_compile scripts/dashboard.py && ./verify.sh --local"
