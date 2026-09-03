#!/usr/bin/env bash
#
# Check everything without starting anything.
#
#   ./verify.sh                  check this machine
#   ./verify.sh --wifi           check the Go2 over wifi
#   ./verify.sh --ethernet       check the Go2 over the cable
#
# Options:
#   --full      also re-run the accuracy evaluation (slow, ~2 min)
#   --port N    which port to test for availability (default 5050)
#
# Runs the same pre-flight that start.sh does -- venv, packages, models,
# gallery, camera pixels, port -- and then confirms the machine reproduces the
# reference scores recorded on the development laptop.

source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

show_help() { sed -n '3,17p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; }

FULL=0
ARGS=()
for a in "$@"; do
  case "$a" in
    --full) FULL=1 ;;
    *)      ARGS+=("$a") ;;
  esac
done
parse_common "${ARGS[@]}"
cd "$WS" || exit 1
printf "${B}Face recognition — verify (%s)${N}\n" "$MODE"

# --------------------------------------------------------------- remote modes
if [[ "$MODE" != "local" ]]; then
  require_link
  hdr "Robot pre-flight"
  rrun "cd $REMOTE && ./verify.sh --local --port $PORT $([[ $FULL == 1 ]] && echo --full) 2>&1 | sed 's/\x1b\[[0-9;]*m//g'" 900
  exit $?
fi

# ---------------------------------------------------------------- local checks
# strict=0: report a missing camera rather than aborting, since the rest of the
# verification is still worth seeing.
preflight_local 0

hdr "Port $PORT"
if ss -ltn 2>/dev/null | grep -q ":$PORT "; then
  warn "port $PORT is in use (a dashboard may already be running)"
else
  ok "port $PORT free"
fi

hdr "Reproducibility"
if [[ -f "$WS/db/baseline.json" ]]; then
  "$PY" "$WS/scripts/verify_deploy.py" 2>&1 \
    | grep -viE "WARN|Applied providers|UserWarning|warnings.warn|FutureWarning|tform" \
    | sed 's/^/  /'
else
  warn "no db/baseline.json -- generate it on the development machine:"
  printf "  ${D}      ./venv/bin/python scripts/verify_deploy.py --baseline${N}\n"
fi

if [[ "$FULL" == "1" ]]; then
  hdr "Accuracy (full evaluation)"
  "$PY" "$WS/scripts/evaluate.py" 2>&1 \
    | grep -viE "WARN|Applied providers|UserWarning|warnings.warn|FutureWarning|tform" \
    | sed -n '/SUMMARY/,$p' | sed 's/^/  /'
fi

printf "\n"
if [[ -z "${CAMKIND:-}" ]]; then
  warn "everything checks out except the camera — fix that before starting"
  exit 1
fi
printf "${G}All checks passed.${N} Camera: %s\n\n" "$CAMKIND"
