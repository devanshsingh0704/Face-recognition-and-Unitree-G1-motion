# Shared helpers for start.sh / stop.sh / verify.sh / install.sh
#
# Sourced, never executed. Keeps the four entry points thin and stops the
# pre-flight checks from being copy-pasted into each of them.

set -uo pipefail

WS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Prefer a virtualenv, but fall back to the system interpreter. On the G1,
# `python3 -m venv` produces a venv without pip -- a sitecustomize.py and some
# coverage .pth files break ensurepip when it runs isolated -- and the platform
# already ships numpy/opencv/flask/pyrealsense2 anyway. So a venv is a nicety
# there, not a requirement.
if [[ -x "$WS/venv/bin/python" ]] && "$WS/venv/bin/python" -c "import numpy" >/dev/null 2>&1; then
  PY="$WS/venv/bin/python"
else
  PY="$(command -v python3)"
fi
REMOTE=/home/unitree/face_recognition_ws
REMOTE_LOG=/home/unitree/fr_dash.log

# Known robots. Addresses live OUTSIDE the repo so nothing published carries
# this network's layout: ~/.fr_robots.env, or FR_* in the environment. See
# .fr_robots.env.example. Only the remote modes need them -- the usual
# `./start.sh --local` on the robot itself never reads any of this.
[[ -f "$HOME/.fr_robots.env" ]] && source "$HOME/.fr_robots.env"
HOST_G1="${FR_HOST_G1:-}"          # G1 over wifi
HOST_GO2="${FR_HOST_GO2:-}"        # Go2 over wifi
HOST_GO2_ETH="${FR_HOST_GO2_ETH:-}"  # Go2 over the ethernet cable
HOST_OVERRIDE=""                   # set by --host

R=$'\e[31m'; G=$'\e[32m'; Y=$'\e[33m'; B=$'\e[36m'; D=$'\e[2m'; N=$'\e[0m'
ok()   { printf "  ${G}ok${N}    %s\n" "$1"; }
warn() { printf "  ${Y}warn${N}  %s\n" "$1"; }
bad()  { printf "  ${R}FAIL${N}  %s\n" "$1"; }
hdr()  { printf "\n${B}%s${N}\n" "$1"; }
die()  { printf "\n${R}Cannot continue.${N} %s\n\n" "$1"; exit 1; }

# ------------------------------------------------------------------ arguments
# Default to "local" -- meaning "this machine". On the robot (where these
# scripts normally run) that is what you want, so ./start.sh needs no flag.
# --g1 / --wifi / --ethernet are only for driving a robot from the laptop.
MODE=local         # local | g1 | go2 | ethernet
PORT=5050
RANGE=mid          # G1 on USB 3: mid (1280x720). Go2 on USB 2: use --range near.
CAMERA=""          # realsense | webcam | "" = auto (RealSense first)
CAMERA_INDEX=0
DO_SYNC=0
TOKEN=""           # shared token for the control endpoints; generated if empty
SET_LOGIN=0        # --set-login: write the login account and exit

# Go-to-person tuning, forwarded verbatim to dashboard.py rather than validated
# here. One place decides what a legal standoff is, and it is the code that
# would act on an illegal one -- dashboard.py refuses anything below its floor.
# Without this passthrough the approach could only ever run on its defaults,
# which is not what you want the first time you walk the robot at a person.
MOVE_ARGS=()

parse_common() {
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --local)          MODE=local; shift ;;
      --go2|--wifi)     MODE=go2; shift ;;
      --g1)             MODE=g1; shift ;;
      --ethernet|--eth) MODE=ethernet; shift ;;
      --host)           MODE=custom; HOST_OVERRIDE="${2:?--host needs an address}"; shift 2 ;;
      --port)           PORT="${2:?--port needs a number}"; shift 2 ;;
      --range)          RANGE="${2:?--range needs near|mid|far}"; shift 2
                        [[ "$RANGE" =~ ^(near|mid|far)$ ]] || die "--range must be near, mid or far" ;;
      --realsense)      CAMERA=realsense; shift ;;
      --webcam)         CAMERA=webcam; shift
                        [[ "${1:-}" =~ ^[0-9]+$ ]] && { CAMERA_INDEX="$1"; shift; } ;;
      --sync)           DO_SYNC=1; shift ;;
      --token)          TOKEN="${2:?--token needs a value}"; shift 2 ;;
      --set-login)      SET_LOGIN=1; shift ;;
      --stop-distance|--stop-px|--move-vx|--move-omega|--obstacle-distance|\
      --move-timeout|--move-iface)
                        MOVE_ARGS+=("$1" "${2:?$1 needs a value}"); shift 2 ;;
      -h|--help)        show_help; exit 0 ;;
      *)                die "Unknown option: $1  (try --help)" ;;
    esac
  done
}

host_for_mode() {
  case "$MODE" in
    go2)      echo "$HOST_GO2" ;;
    g1)       echo "$HOST_G1" ;;
    ethernet) echo "$HOST_GO2_ETH" ;;
    custom)   echo "$HOST_OVERRIDE" ;;
    *)        echo "" ;;
  esac
}

# --------------------------------------------------------------- remote access
rrun() {  # rrun "<cmd>" [timeout]
  "$PY" "$WS/scripts/go2.py" --host "$(host_for_mode)" --timeout "${2:-120}" "$1"
}

require_link() {
  local host; host="$(host_for_mode)"
  # Say which knob is missing rather than letting ping fail on an empty string.
  [[ -n "$host" ]] || die "No address configured for --$MODE.
  Robot addresses are kept out of the repo. Create ~/.fr_robots.env from
  .fr_robots.env.example, or export FR_HOST_G1 / FR_HOST_GO2 / FR_HOST_GO2_ETH.
  Running on the robot itself needs none of this -- use --local."
  hdr "Link ($MODE — $host)"
  ping -c1 -W2 "$host" >/dev/null 2>&1 || die "$host does not respond.
  wifi:     the robot must be on this network -- check the Unitree app
  ethernet: plug the cable in and give this laptop a 192.168.123.x address
  Other options:  --go2  --g1  --ethernet  --host IP"
  rrun 'echo up' >/dev/null 2>&1 || die "$host pings but SSH is refused.
  On wifi this usually means another device has taken the robot's IP.
  Use the cable instead:  --ethernet"
  ok "reachable, ssh working"
}

sync_to_robot() {   # $1 = "db" for gallery only, "all" for code too
  hdr "Syncing to robot"
  local what="${1:-db}"
  "$PY" - "$what" <<'PYEOF'
import sys
sys.path.insert(0, "scripts")
from go2 import put
from pathlib import Path
import os
what = sys.argv[1]
remote = os.environ.get("REMOTE", "/home/unitree/face_recognition_ws")
files = ["db/embeddings.npz", "db/config.json", "db/baseline.json"]
if what == "all":
    files += ["start.sh", "stop.sh", "verify.sh", "install.sh", "lib.sh"] + \
             [str(p) for p in sorted(Path("scripts").glob("*.py"))]
sent = 0
for f in files:
    p = Path(f)
    if p.exists():
        put(p, f"{remote}/{f}")
        print(f"  sent {f}")
        sent += 1
print(f"  {sent} file(s)")
PYEOF
  [[ "$what" == "all" ]] && rrun "chmod +x $REMOTE/*.sh" >/dev/null 2>&1
  ok "synced"
}

# ------------------------------------------------------------- local preflight
# Every check here exists because that exact thing broke during development.
preflight_local() {
  local strict="${1:-1}"   # 0 = report only, do not exit

  hdr "Environment"
  [[ -x "$PY" ]] || die "No venv at $WS/venv
  Run:  ./install.sh"
  ok "venv  $("$PY" --version 2>&1)"

  local missing
  missing=$("$PY" - <<'PYEOF' 2>/dev/null
import importlib.util as u
need = {"cv2": "opencv-python", "numpy": "numpy",
        "onnxruntime": "onnxruntime", "flask": "flask",
        "insightface": "insightface"}
print(" ".join(p for m, p in need.items() if u.find_spec(m) is None))
PYEOF
)
  [[ -z "$missing" ]] || die "Missing packages: $missing
  Run:  ./install.sh"
  ok "packages  opencv, numpy, onnxruntime, flask, insightface"

  hdr "Models"
  local m
  for m in models/candidates/buffalo_s/det_500m.onnx \
           models/candidates/buffalo_s/w600k_mbf.onnx; do
    if [[ -s "$WS/$m" ]]; then
      ok "$(basename "$m")  ($(du -h "$WS/$m" | cut -f1))"
    else
      die "Missing model: $m
  Run:  ./install.sh"
    fi
  done

  hdr "Identity database"
  if [[ ! -f "$WS/db/embeddings.npz" ]]; then
    die "Nobody is enrolled (db/embeddings.npz missing)
  Put photos in dataset/enroll/<name>/ then run:
    ./venv/bin/python scripts/enroll.py --report"
  fi
  local dbinfo
  dbinfo=$("$PY" - <<'PYEOF' 2>&1
import sys, json
sys.path.insert(0, "scripts")
from face_pipeline import IdentityDB, CONFIG_PATH, load_threshold
db = IdentityDB.load()
print("PEOPLE|" + ", ".join(f"{n} ({len(db.samples[n])})" for n in db.names))
print("SUBS|" + ", ".join(f"{n}:{len(db.subcentroids.get(n, []))}" for n in db.names))
print(f"THR|{load_threshold():.2f}")
if CONFIG_PATH.exists():
    m = json.loads(CONFIG_PATH.read_text()).get("metrics", {})
    if m.get("impostor_max") is not None:
        print(f"GAP|impostor max {m['impostor_max']:.3f}, genuine min {m['genuine_min']:.3f}, "
              f"gap {m['genuine_min'] - m['impostor_max']:+.3f} "
              f"(n={m.get('n_impostor', '?')} strangers)")
else:
    print("GAP|not tuned -- run scripts/tune_threshold.py --save")
PYEOF
)
  grep -q Traceback <<<"$dbinfo" && die "Could not read the database:
$dbinfo"
  ok "enrolled: $(sed -n 's/^PEOPLE|//p' <<<"$dbinfo")"
  ok "pose clusters: $(sed -n 's/^SUBS|//p' <<<"$dbinfo")"
  THRESHOLD=$(sed -n 's/^THR|//p' <<<"$dbinfo")
  ok "threshold: $THRESHOLD"
  printf "  ${D}      %s${N}\n" "$(sed -n 's/^GAP|//p' <<<"$dbinfo")"
  local npeople
  npeople=$(sed -n 's/^PEOPLE|//p' <<<"$dbinfo" | awk -F',' '{print NF}')
  [[ "$npeople" -ge 2 ]] || warn "only one person enrolled -- cross-person confusion is unmeasurable"

  hdr "Camera"
  # A camera that opens can still hand back black frames: this laptop's webcam
  # stays black for ~8s after opening, which once looked exactly like a dead
  # device. Probe actual pixels, never just open().
  local caminfo
  caminfo=$("$PY" - "$CAMERA" "$CAMERA_INDEX" <<'PYEOF' 2>/dev/null
import sys, time
force, index = sys.argv[1], int(sys.argv[2])

def realsense():
    try:
        import pyrealsense2 as rs
    except ImportError:
        return None, "pyrealsense2 not installed"
    try:
        devs = rs.context().query_devices()
        if len(devs) == 0:
            return None, "no RealSense on the USB bus"
        d = devs[0]
        name = d.get_info(rs.camera_info.name)
        serial = d.get_info(rs.camera_info.serial_number)
        try:
            usb = d.get_info(rs.camera_info.usb_type_descriptor)
        except Exception:
            usb = "?"
        import numpy as np
        pipe, cfg = rs.pipeline(), rs.config()
        cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        pipe.start(cfg)
        try:
            good = sum(
                1 for _ in range(30)
                if (f := pipe.wait_for_frames(timeout_ms=3000).get_color_frame())
                and np.asanyarray(f.get_data()).std() > 3
            )
        finally:
            pipe.stop()
        if good < 10:
            return None, f"{name} opened but gave {good}/30 usable frames"
        return "realsense", f"{name} serial {serial}, USB {usb}, {good}/30 frames ok"
    except Exception as e:
        return None, f"RealSense error: {type(e).__name__}: {e}"

def webcam(i):
    import cv2
    cap = cv2.VideoCapture(i, cv2.CAP_V4L2)
    if not cap.isOpened():
        return None, f"/dev/video{i} will not open"
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    start = time.time()
    try:
        while time.time() - start < 25:
            okf, fr = cap.read()
            if okf and fr is not None and fr.std() > 3:
                return "webcam", (f"/dev/video{i} {fr.shape[1]}x{fr.shape[0]}, "
                                  f"ready in {time.time() - start:.1f}s")
            time.sleep(0.05)
    finally:
        cap.release()
    return None, f"/dev/video{i} stayed blank for 25s (privacy shutter? in use?)"

notes = []
if force in ("", "realsense"):
    kind, msg = realsense(); notes.append(msg)
    if kind: print(f"OK|{kind}|{msg}"); raise SystemExit
    if force == "realsense": print(f"ERR|{msg}"); raise SystemExit
if force in ("", "webcam"):
    kind, msg = webcam(index); notes.append(msg)
    if kind: print(f"OK|{kind}|{msg}"); raise SystemExit
print("ERR|" + " ; ".join(notes))
PYEOF
)
  if [[ "$caminfo" == OK\|* ]]; then
    CAMKIND=$(cut -d'|' -f2 <<<"$caminfo")
    ok "$(cut -d'|' -f3 <<<"$caminfo")"
  else
    bad "$(cut -d'|' -f2- <<<"$caminfo")"
    [[ "$strict" == "1" ]] && die "No usable camera.
  - RealSense: check it is plugged in; a USB 3 port gives more range
  - Webcam:    check the privacy shutter and the camera-mute key
  - Either:    close anything else holding the camera
  Diagnose:  ./venv/bin/python scripts/check_camera.py"
    CAMKIND=""
  fi
}

free_port_local() {
  hdr "Port $PORT"
  local stale
  stale=$(pgrep -f "venv/bin/python.*[s]cripts/dashboard\.py" 2>/dev/null || true)
  if [[ -n "$stale" ]]; then
    warn "stopping a dashboard already running (pid $(tr '\n' ' ' <<<"$stale"))"
    kill -9 $stale 2>/dev/null || true
    sleep 2
  fi
  local i
  for i in {1..10}; do
    ss -ltn 2>/dev/null | grep -q ":$PORT " || break
    sleep 1
  done
  ss -ltn 2>/dev/null | grep -q ":$PORT " \
    && die "Port $PORT is held by something else.
  Look:  ss -ltnp | grep :$PORT
  Or:    --port 5051"
  ok "port $PORT free"
}
