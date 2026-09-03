#!/usr/bin/env bash
# Restore source files from the latest backup (or a given backup directory).
# Usage: ./recover.sh [path/to/.fr_backup_YYYYMMDD_HHMMSS]
set -euo pipefail
WS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Backups live outside the workspace -- see the note in backup.sh. Override
# with FR_ARCHIVE, matching backup.sh.
ARCHIVE="${FR_ARCHIVE:-$HOME/fr_history_archive}"

SRC="${1:-}"
if [[ -z "$SRC" ]]; then
  # Prefer the symlink, but fall back to the newest snapshot by mtime. The
  # symlink is the one part of this that can go stale -- it broke once when the
  # directory it named was moved -- and "restore the newest" is what someone
  # running this in a hurry means anyway.
  if [[ -d "$ARCHIVE/.fr_backup_latest" ]]; then
    SRC="$ARCHIVE/.fr_backup_latest"
  else
    SRC="$(ls -dt "$ARCHIVE"/.fr_backup_20* 2>/dev/null | head -1 || true)"
    [[ -n "$SRC" ]] && echo "No 'latest' link; using newest snapshot instead."
  fi
fi

if [[ -z "$SRC" || ! -d "$SRC" ]]; then
  echo "No backup found in $ARCHIVE" >&2
  echo "Run ./backup.sh first, or pass a snapshot directory:" >&2
  echo "    ./recover.sh $ARCHIVE/.fr_backup_YYYYMMDD_HHMMSS" >&2
  ls -1dt "$ARCHIVE"/.fr_backup_20* 2>/dev/null | head -5 | sed 's|^|  available: |' >&2 || true
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
