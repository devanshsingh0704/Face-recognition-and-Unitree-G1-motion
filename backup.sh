#!/usr/bin/env bash
# Snapshot the workspace source files so they can be restored if deleted again.
# Run manually after changes, or from cron: 0 */6 * * * ~/face_recognition_ws/backup.sh -q
set -euo pipefail
WS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
QUIET=0
[[ "${1:-}" == "-q" ]] && QUIET=1

# Snapshots live OUTSIDE the workspace, in $HOME.
#
# They contain context.md, which carries the robot addresses and SSH password.
# Keeping copies of that inside the workspace put credentials inside the
# directory we publish from, guarded only by a .gitignore line -- and on
# 2026-09-03 a fresh `git init` did sweep a saved .git directory into a commit
# (66 of 93 tracked files) before a content scan caught it. One boundary that
# does not depend on getting an ignore pattern right is worth more than five
# copies that do.
#
# Override with FR_ARCHIVE if you want them somewhere else.
ARCHIVE="${FR_ARCHIVE:-$HOME/fr_history_archive}"

STAMP="$(date +%Y%m%d_%H%M%S)"
DEST="$ARCHIVE/.fr_backup_${STAMP}"
LATEST="$ARCHIVE/.fr_backup_latest"

# 700: these snapshots hold a password, so keep them to this account.
mkdir -p "$ARCHIVE"
chmod 700 "$ARCHIVE"
mkdir -p "$DEST/scripts"

copy() {
  local src=$1 dest=$2
  if [[ -f "$src" ]]; then
    cp -a "$src" "$dest"
  fi
}

for f in start.sh stop.sh lib.sh install.sh verify.sh backup.sh recover.sh context.md README.md; do
  copy "$WS/$f" "$DEST/$f"
done

for f in dashboard.py face_pipeline.py enroll.py recognize.py check_camera.py \
         benchmark.py compare_models.py g1_locomotion.py evaluate.py fetch_impostors.py \
         go2.py selftest.py tune_threshold.py verify_deploy.py; do
  copy "$WS/scripts/$f" "$DEST/scripts/$f"
done

# Keep compiled bytecode as a last-resort source recovery aid.
if compgen -G "$WS/scripts/__pycache__/dashboard.cpython-*.pyc" >/dev/null; then
  mkdir -p "$DEST/scripts/__pycache__"
  cp -a "$WS/scripts/__pycache__/dashboard.cpython-"*.pyc "$DEST/scripts/__pycache__/" 2>/dev/null || true
fi

ln -sfn "$DEST" "$LATEST"

# Prune old backups (keep last 10). Scoped to the archive so a stray
# .fr_backup_* left in the workspace is never a deletion target -- this only
# removes what this script created.
mapfile -t OLD < <(ls -dt "$ARCHIVE"/.fr_backup_20* 2>/dev/null | tail -n +11 || true)
for d in "${OLD[@]}"; do
  [[ -d "$d" && "$d" != "$DEST" ]] && rm -rf "$d"
done

if [[ "$QUIET" -eq 0 ]]; then
  echo "Backed up to $DEST"
  echo "Latest symlink: $LATEST -> $DEST"
  echo "Archive is outside the workspace, so snapshots of context.md (which"
  echo "holds the robot password) are never inside the published tree."
fi
