#!/usr/bin/env bash
#
# Stop the dashboard and release the camera.
#
#   ./stop.sh                  stop it on this machine
#   ./stop.sh --wifi             stop it on the Go2 over wifi
#   ./stop.sh --ethernet         stop it on the Go2 over the cable
#
# Options:
#   --status    report whether it is running instead of stopping it
#   --logs      show the robot's dashboard log (remote modes)
#   --port N    default 5050

source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

show_help() { sed -n '3,15p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; }

WANT=stop
ARGS=()
for a in "$@"; do
  case "$a" in
    --status) WANT=status ;;
    --logs)   WANT=logs ;;
    *)        ARGS+=("$a") ;;
  esac
done
parse_common "${ARGS[@]}"
cd "$WS" || exit 1
printf "${B}Face recognition — %s (%s)${N}\n" "$WANT" "$MODE"

report_status() {   # $1 = host or 127.0.0.1
  if curl -s --max-time 5 "http://$1:$PORT/api/status" -o /tmp/fr_status.json 2>/dev/null; then
    "$PY" - <<'PYEOF'
import json
d = json.load(open("/tmp/fr_status.json"))
print(f"  running    {d['source']}")
print(f"  video      {d['fps']:.1f} fps")
print(f"  inference  {d['infer_fps']:.1f} fps   detect {d['detect_ms']:.0f} ms")
print(f"  threshold  {d['threshold']}  (tuned {d['tuned_threshold']})")
print(f"  enrolled   {', '.join(d['enrolled'])}")
if d["faces"]:
    for f in d["faces"]:
        print(f"  in frame   {f['label']} {f['score']:.3f} ({f['px']}px)")
else:
    print("  in frame   nobody")
PYEOF
  else
    warn "not running (nothing answering on port $PORT)"
    return 1
  fi
}

if [[ "$MODE" == "local" ]]; then
  case "$WANT" in
    status) hdr "Status"; report_status 127.0.0.1 || true ;;
    logs)   warn "--logs only applies to the robot; local runs print to your terminal" ;;
    stop)
      hdr "Stopping"
      pids=$(pgrep -f "venv/bin/python.*[s]cripts/dashboard\.py" 2>/dev/null || true)
      if [[ -z "$pids" ]]; then
        ok "nothing was running"
      else
        kill -9 $pids 2>/dev/null || true
        sleep 2
        pgrep -f "venv/bin/python.*[s]cripts/dashboard\.py" >/dev/null 2>&1 \
          && warn "a process survived — check manually" \
          || ok "stopped (pid $(tr '\n' ' ' <<<"$pids"))"
      fi
      ss -ltn 2>/dev/null | grep -q ":$PORT " && warn "port $PORT still bound" || ok "port $PORT free"
      ;;
  esac
  exit 0
fi

HOST="$(host_for_mode)"
require_link
case "$WANT" in
  status)
    hdr "Status"
    report_status "$HOST" || printf "  ${D}start it with:  ./start.sh --%s${N}\n" "$MODE"
    ;;
  logs)
    hdr "Robot dashboard log"
    rrun "sed 's/\x1b\[[0-9;]*m//g' $REMOTE_LOG 2>/dev/null | grep -viE 'GET /|Applied providers|UserWarning|warnings.warn|FutureWarning|tform' | tail -30" \
      || warn "no log — has it been started?"
    ;;
  stop)
    hdr "Stopping on the robot"
    # The bracket in [s]cripts stops the pattern matching the shell that is
    # running it -- otherwise pkill kills itself before the checks below run,
    # and reports a survivor that does not exist.
    rrun "pkill -9 -f '[s]cripts/dashboard.py' 2>/dev/null; sleep 2; \
          ps -eo cmd | grep -q '[s]cripts/dashboard.py' && echo ALIVE || echo GONE; \
          ss -ltn | grep -q ':$PORT ' && echo BOUND || echo FREE" > /tmp/fr_stop.txt 2>&1
    grep -q GONE /tmp/fr_stop.txt && ok "dashboard stopped" || warn "a process may have survived"
    grep -q FREE /tmp/fr_stop.txt && ok "port $PORT free, camera released" || warn "port $PORT still bound"
    ;;
esac
