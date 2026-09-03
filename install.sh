#!/usr/bin/env bash
#
# Set up everything needed to run: virtual environment, packages, models.
#
#   ./install.sh                  set up this machine
#   ./install.sh --wifi          set up the Go2 over wifi (copies the workspace)
#   ./install.sh --ethernet      set up the Go2 over the cable
#
# Options:
#   --models-only    just fetch the ONNX models, skip the venv
#   --impostors      also download the FairFace impostor set (~2.7 GB) used for
#                    threshold tuning; not needed simply to run
#
# Safe to re-run: it skips what is already present.

source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

show_help() { sed -n '3,17p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; }

MODELS_ONLY=0
IMPOSTORS=0
ARGS=()
for a in "$@"; do
  case "$a" in
    --models-only) MODELS_ONLY=1 ;;
    --impostors)   IMPOSTORS=1 ;;
    *)             ARGS+=("$a") ;;
  esac
done
parse_common "${ARGS[@]}"
cd "$WS" || exit 1
printf "${B}Face recognition — install (%s)${N}\n" "$MODE"

BUFFALO_URL=https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_s.zip
YUNET_URL=https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx

# --------------------------------------------------------------- remote modes
if [[ "$MODE" != "local" ]]; then
  require_link
  hdr "Copying the workspace to the robot"
  rrun "mkdir -p $REMOTE/scripts $REMOTE/db $REMOTE/models/candidates/buffalo_s"
  export REMOTE
  sync_to_robot all

  hdr "Copying models"
  "$PY" - <<'PYEOF'
import sys, os
sys.path.insert(0, "scripts")
from go2 import put
from pathlib import Path
remote = os.environ.get("REMOTE", "/home/unitree/face_recognition_ws")
for f in ["models/candidates/buffalo_s/det_500m.onnx",
          "models/candidates/buffalo_s/w600k_mbf.onnx",
          "models/face_detection_yunet_2023mar.onnx",
          "models/face_recognition_sface_2021dec.onnx"]:
    p = Path(f)
    if p.exists():
        put(p, f"{remote}/{f}")
        print(f"  sent {f}  ({p.stat().st_size/1e6:.0f} MB)")
PYEOF
  ok "models copied"

  hdr "Building the environment on the robot"
  printf "  ${D}this can take several minutes on ARM${N}\n"
  # --system-site-packages reuses the JetPack numpy/opencv rather than
  # rebuilding them, which on aarch64 is the difference between minutes and
  # an hour.
  rrun "cd $REMOTE && python3 -m venv --system-site-packages venv 2>&1 | tail -2; \
        ./venv/bin/pip install --quiet --upgrade pip 2>&1 | tail -2; \
        ./venv/bin/pip install --quiet onnxruntime flask insightface pyrealsense2 2>&1 | tail -4; \
        ./venv/bin/python -c 'import cv2,numpy,onnxruntime,flask,insightface,pyrealsense2; print(\"  all imports ok\")'" 1800
  ok "environment ready"

  hdr "Verifying"
  rrun "cd $REMOTE && ./verify.sh --local 2>&1 | sed 's/\x1b\[[0-9;]*m//g' | tail -20" 900
  printf "\n${G}Robot ready.${N} Start it with (on the robot):  ./start.sh\n\n" "$MODE"
  exit 0
fi

# ---------------------------------------------------------------- local mode
if [[ "$MODELS_ONLY" == "0" ]]; then
  hdr "Virtual environment"
  if [[ -x "$PY" ]]; then
    ok "already exists  ($("$PY" --version 2>&1))"
  else
    python3 -m venv venv || die "Could not create a venv.
  On Debian/Ubuntu you may need:  sudo apt install python3-venv"
    ok "created  ($("$PY" --version 2>&1))"
  fi

  hdr "Packages"
  "$PY" -m pip install --quiet --upgrade pip 2>&1 | tail -2
  local_missing=$("$PY" - <<'PYEOF' 2>/dev/null
import importlib.util as u
need = {"cv2": "opencv-python", "numpy": "numpy", "onnxruntime": "onnxruntime",
        "flask": "flask", "insightface": "insightface", "paramiko": "paramiko"}
print(" ".join(p for m, p in need.items() if u.find_spec(m) is None))
PYEOF
)
  if [[ -z "$local_missing" ]]; then
    ok "all present"
  else
    printf "  ${D}installing: %s${N}\n" "$local_missing"
    "$PY" -m pip install --quiet $local_missing 2>&1 | tail -3
    ok "installed"
  fi
  # pyrealsense2 is optional: no wheel on some platforms, and a webcam works fine
  "$PY" -c "import pyrealsense2" >/dev/null 2>&1 \
    && ok "pyrealsense2 present" \
    || { "$PY" -m pip install --quiet pyrealsense2 >/dev/null 2>&1 \
         && ok "pyrealsense2 installed" \
         || warn "pyrealsense2 unavailable here — the RealSense will not work, webcam still will"; }
fi

hdr "Models"
mkdir -p models/candidates
if [[ -s models/candidates/buffalo_s/det_500m.onnx && -s models/candidates/buffalo_s/w600k_mbf.onnx ]]; then
  ok "buffalo_s already present (SCRFD detector + ArcFace recogniser)"
else
  printf "  ${D}downloading buffalo_s...${N}\n"
  curl -fsSL --retry 3 -o models/candidates/buffalo_s.zip "$BUFFALO_URL" \
    || die "Download failed: $BUFFALO_URL"
  mkdir -p models/candidates/buffalo_s
  ( cd models/candidates/buffalo_s && unzip -o -q ../buffalo_s.zip )
  rm -f models/candidates/buffalo_s.zip
  # 1k3d68 and 2d106det are landmark models we do not use; genderage likewise
  rm -f models/candidates/buffalo_s/1k3d68.onnx \
        models/candidates/buffalo_s/2d106det.onnx \
        models/candidates/buffalo_s/genderage.onnx
  ok "buffalo_s ready  ($(du -sh models/candidates/buffalo_s | cut -f1))"
fi
if [[ -s models/face_detection_yunet_2023mar.onnx ]]; then
  ok "yunet present (fallback detector)"
else
  curl -fsSL --retry 3 -o models/face_detection_yunet_2023mar.onnx "$YUNET_URL" \
    && ok "yunet downloaded" || warn "yunet download failed (only needed for --backend sface)"
fi

if [[ "$IMPOSTORS" == "1" ]]; then
  hdr "Impostor set (FairFace, CC BY 4.0)"
  if [[ -d dataset/impostors && $(ls dataset/impostors/*/*.jpg 2>/dev/null | wc -l) -gt 100 ]]; then
    ok "already present ($(ls dataset/impostors/*/*.jpg | wc -l) images)"
  else
    printf "  ${D}this downloads ~2.7 GB and takes a while${N}\n"
    "$PY" -m pip install --quiet datasets pillow 2>&1 | tail -2
    "$PY" scripts/fetch_impostors.py --split validation --count 2000 --world-count 300 \
      2>&1 | grep -viE "^Downloading|^Resolving|%\|" | tail -6
  fi
fi

hdr "Result"
if [[ -f db/embeddings.npz ]]; then
  ok "gallery present — run ./verify.sh"
else
  warn "nobody enrolled yet"
  printf "  ${D}put photos in dataset/enroll/<name>/ then run:${N}\n"
  printf "  ${D}  ./venv/bin/python scripts/enroll.py --report${N}\n"
  printf "  ${D}  ./venv/bin/python scripts/tune_threshold.py --save${N}\n"
fi
printf "\n${G}Install complete.${N}  Next:  ./verify.sh\n\n"
