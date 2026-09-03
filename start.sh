#!/usr/bin/env bash
#
# Verify everything, then start the face recognition dashboard.
#
#   ./start.sh                  run on the G1 locally (or this machine)
#   ./start.sh --g1              sync gallery and run on the G1 over wifi
#   ./start.sh --wifi            sync gallery and run on the Go2 over wifi
#   ./start.sh --ethernet        run on the Go2 over the ethernet cable
#
# Options:
#   --range near|mid|far   default mid (1280x720) on G1 USB 3; use near on Go2 USB 2
#   --port N               default 5050
#   --realsense            force the D435i (default auto-probe tries RealSense first)
#   --webcam [N]           force a V4L2 camera
#   --sync                 push code as well as the gallery (remote modes)
#   --token T              pin the control token instead of generating one
#
# Go-to-person tuning (forwarded to dashboard.py, which validates them):
#   --stop-distance M      standoff in metres, camera to face (floor 0.20)
#   --move-vx M/S          forward speed        --move-omega RAD/S  max yaw rate
#   --obstacle-distance M  stop-and-wait range  --move-timeout S    give up after
#   --move-iface IFACE     DDS interface for LocoClient (eth0 on the G1)
#
# The remote modes do not duplicate any logic: they copy the gallery across and
# then run this same script with --local on the robot.

source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

show_help() { sed -n '3,25p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; }

parse_common "$@"
cd "$WS" || exit 1
printf "${B}Face recognition — start (%s)${N}\n" "$MODE"

# Generated here rather than in dashboard.py so that both paths can print a URL
# that actually works -- in remote mode this script never sees the robot's
# stdout, so a token minted on the robot would be invisible from the laptop.
if [[ -z "$TOKEN" ]]; then
  TOKEN=$(head -c 32 /dev/urandom | base64 | tr -dc 'A-Za-z0-9' | cut -c1-16)
fi

# --------------------------------------------------------------- remote modes
if [[ "$MODE" != "local" ]]; then
  HOST="$(host_for_mode)"
  require_link
  export REMOTE
  sync_to_robot "$([[ $DO_SYNC == 1 ]] && echo all || echo db)"

  hdr "Starting on the robot"
  rrun "pkill -9 -f '[s]cripts/dashboard.py' 2>/dev/null; rm -f $REMOTE_LOG; echo cleared" >/dev/null 2>&1

  # Detached deliberately: a backgrounded process inherits the ssh channel's
  # stdout, so reading it would block until the server exits, i.e. never.
  REMOTE_ARGS="--local --port $PORT --range $RANGE --token $TOKEN"
  [[ -n "$CAMERA" ]] && REMOTE_ARGS="$REMOTE_ARGS --$CAMERA"
  # Guarded expansion: an empty array under `set -u` is an error on older bash.
  [[ ${#MOVE_ARGS[@]} -gt 0 ]] && REMOTE_ARGS="$REMOTE_ARGS ${MOVE_ARGS[*]}"
  "$PY" - <<PYEOF
import sys; sys.path.insert(0, "scripts")
from go2 import launch
launch("cd $REMOTE && setsid nohup ./start.sh $REMOTE_ARGS "
       "</dev/null >$REMOTE_LOG 2>&1 &", host="$HOST")
PYEOF

  printf "  ${D}waiting for the camera to warm up...${N}\n"
  for _ in $(seq 1 45); do
    if curl -s --max-time 3 "http://$HOST:$PORT/api/status" -o /tmp/fr_status.json 2>/dev/null; then
      ok "dashboard live on the robot"
      "$PY" - <<'PYEOF'
import json
d = json.load(open("/tmp/fr_status.json"))
print(f"  {d['source']}  ·  threshold {d['threshold']}  ·  {', '.join(d['enrolled'])}")
PYEOF
      printf "\n  open  ${B}http://%s:%s/?token=%s${N}\n" "$HOST" "$PORT" "$TOKEN"
      printf "  ${D}the token is required for the page, the video and go-to;${N}\n"
      printf "  ${D}Stop works without it${N}\n\n"
      printf "  ${D}stop:    ./stop.sh --%s${N}\n" "$MODE"
      printf "  ${D}logs:    ./stop.sh --%s --logs${N}\n\n" "$MODE"
      exit 0
    fi
    sleep 2
  done
  printf "\n"
  bad "did not come up within 90s — the robot's log follows"
  rrun "sed 's/\x1b\[[0-9;]*m//g' $REMOTE_LOG 2>/dev/null | tail -25"
  exit 1
fi

# ---------------------------------------------------------------- local mode
preflight_local 1
free_port_local

hdr "Starting dashboard"
ARGS=(--host 0.0.0.0 --port "$PORT" --range "$RANGE" --token "$TOKEN")
if [[ "$CAMKIND" == "realsense" ]]; then
  ARGS+=(--realsense)
else
  ARGS+=(--camera "$CAMERA_INDEX")
fi
[[ ${#MOVE_ARGS[@]} -gt 0 ]] && ARGS+=("${MOVE_ARGS[@]}")

# Every address, labelled, wifi first.
#
# `hostname -I | awk '{print $1}'` returned the eth0 address here -- the G1's
# internal control-board network, which a laptop on wifi cannot reach.
# That was survivable when you could just type the IP you knew, but the URL now
# carries the token, so the printed line is the one people copy and it has to
# be one that works. Printing all of them beats guessing which network the
# person reading this is on.
print_urls() {
  local pass iface addr
  for pass in wlan other; do
    while read -r iface addr; do
      case "$iface" in lo|docker*|br-*|veth*) continue ;; esac
      # wl* not wlan*: the robot has wlan0 but the laptop has wlp0s20f3.
      [[ "$pass" == wlan  && "$iface" != wl* ]] && continue
      [[ "$pass" == other && "$iface" == wl* ]] && continue
      printf "  open  ${B}http://%s:%s/?token=%s${N}   ${D}(%s)${N}\n" \
             "$addr" "$PORT" "$TOKEN" "$iface"
    done < <(ip -4 -o addr show scope global 2>/dev/null |
             awk '{split($4,a,"/"); print $2, a[1]}')
  done
}
URLS="$(print_urls)"
if [[ -n "$URLS" ]]; then
  printf '%s\n' "$URLS"
else
  printf "  open  ${B}http://127.0.0.1:%s/?token=%s${N}\n" "$PORT" "$TOKEN"
fi
printf "  ${D}the token is required for the page, the video and go-to;${N}\n"
printf "  ${D}Stop works without it${N}\n"
printf "  ${D}threshold %s — do not lower it below the tuned value${N}\n" "${THRESHOLD:-?}"
printf "  ${D}Ctrl-C to stop${N}\n\n"

# exec so Ctrl-C reaches Python directly and the camera is released cleanly
exec "$PY" -u "$WS/scripts/dashboard.py" "${ARGS[@]}"
