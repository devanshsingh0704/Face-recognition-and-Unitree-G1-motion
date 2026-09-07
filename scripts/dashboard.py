"""Local browser dashboard for live recognition testing.

Runs a small Flask server that streams annotated frames as MJPEG plus a live
stats panel. Nothing is sent anywhere -- it binds locally by default and the
page is served from this process.

This is also how the system runs on the G1: the robot has no display, so
a headless server streaming to a browser on your laptop or phone is the natural
deployment shape, not an OpenCV window.

    python scripts/dashboard.py                 # webcam, http://127.0.0.1:5000
    python scripts/dashboard.py --realsense
    python scripts/dashboard.py --host 0.0.0.0  # reachable from other devices

The threshold slider is live: drag it and watch identities flip between named
and unknown. That is the fastest way to feel where the decision boundary sits.

The "go to person" panel walks the G1 to whichever enrolled person you name: it
yaws to put their face on the centre line, walks in, and stops at the standoff.
One click runs the whole mission -- if the face goes out of view the robot
searches for it and resumes rather than ending, so the button is pressed once
and not once per stride. It is a dry run until "Enable walk" is ticked, and the
Stop button is live throughout. See move_loop for what ends a mission.

Because that panel can move a 35 kg humanoid, the page, the video and the
control endpoints are gated. Two ways in, for two kinds of caller: people sign
in at /login and get a cookie that lasts 12 hours and survives restarts
(`./start.sh --set-login` creates the account); scripts and curl send the
shared token and never fill in a form. Stop is the one exception -- it needs
neither, because anyone who can reach the robot should be able to halt it.
An approach also needs the browser to keep polling:
lose the page or the wifi and the robot stops by itself within about a second
and a half. See MOVE_HEARTBEAT_TIMEOUT_S for why that is not redundant with the
command-duration backstop in g1_locomotion.
"""

from __future__ import annotations

import argparse
import functools
import getpass
import json
import secrets
import sys
import threading
import time
from collections import deque
from datetime import timedelta
from pathlib import Path
from typing import NamedTuple

import cv2
import numpy as np
from flask import (Flask, Response, jsonify, redirect, render_template_string,
                   request, session)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from g1_locomotion import G1Locomotion
from face_pipeline import (
    CONFIG_PATH,
    DET_THREADS,
    EMB_THREADS,
    MIN_FACE_WIDTH_PX,
    FacePipeline,
    FaceTracker,
    IdentityDB,
    configure_runtime_threads,
    load_threshold,
    open_camera,
    threads_label,
)

# Detection stays at full resolution. Downscaling to 0.5 was 3x faster but cost
# ~0.06 of similarity -- measured on the held-out photos it pushed Parth's worst
# frame to 0.443, below the operating threshold. Landmark precision drives crop
# alignment, and alignment drives the embedding, so cheap detection is a false
# economy here.
#
# The real saving is recognition: it costs ~28 ms per face against detection's
# ~11 ms, and a face does not change identity between frames.
# NOTE: this has no effect on the default 'arcface' backend. SCRFD letterboxes
# internally to its prepared 640x640 input, so detection already costs the same
# regardless of frame size and FacePipeline.detect() returns before `scale` is
# read (see the early return in face_pipeline.detect). It is honoured only by the
# legacy 'sface' backend. Left in place for that path -- do not spend time
# tuning it expecting a speedup here.
DETECT_SCALE = 1.0
RECOGNISE_EVERY = 3
SMOOTH_FRAMES = 9

# Hysteresis: a confirmed identity is kept while its score stays above
# threshold * HOLD_RATIO, but a new one still needs the full threshold. Stops a
# walking person strobing between their name and "unknown" without letting a
# stranger in through the back door.
HOLD_RATIO = 0.85

# SCRFD scores on a lower scale than YuNet did. Measured across 88 genuine
# faces: min 0.603, median 0.774. A 0.70 gate rejected 10 of them; 0.55 rejects
# none while still discarding the weak detections whose embeddings drift.
MIN_DETECT_CONFIDENCE = 0.55

JPEG_QUALITY = 80

# The browser stream is downscaled to this width before encoding when the
# capture frame is wider. Processing still runs at full capture resolution --
# only the picture sent to the browser shrinks -- so range and accuracy are
# untouched while the JPEG encode and the wifi both get cheaper. 0 disables it.
# Matters here because the operator deadman fires on dashboard silence and this
# robot's wifi is unreliable: a lighter stream is a steadier heartbeat.
STREAM_MAX_WIDTH = 960

# Capture resolution sets the working range, because range is pure optics.
# Measured D435i focal lengths, and the distance at which a face is 32px wide
# (the smallest size that still identified 20/20 in testing):
#
#   640x480    fx=604    32px face at 2.9 m     30 fps on USB 2.0
#   1280x720   fx=906    32px face at 4.4 m     15 fps
#   1920x1080  fx=1358   32px face at 6.6 m      8 fps
#
# The frame rates are USB 2.0 limits. On a USB 3 cable the D435i does 1080p at
# 30 fps, which would give the long range and a smooth picture together.
#
# Never resize a frame to a different aspect ratio -- squashing a face
# horizontally cost ~0.3 similarity in testing. Letterbox instead.
#   --range near     640x480    30 fps   to ~2.9 m   smoothest
#   --range mid      1280x720   15 fps   to ~4.4 m   default
#   --range far      1920x1080   8 fps   to ~6.6 m   choppy on USB 2.0
# Reach assumes a 20px face, the smallest SCRFD+ArcFace still identified 20/20
# in testing. (The old figures here assumed YuNet's 32px floor and understated
# every mode.) D435i focal lengths: 604 / 906 / 1358 px.
# Frame rates below are what we ask for; open_source() negotiates down if the
# USB link cannot sustain them. On USB 3 (the G1) every mode runs at 30 fps, so
# `far` gives long range AND smooth video. On USB 2 (the Go2, whose USB 3 port
# is blocked by the lidar) mid falls to 15 and far to 8.
RANGE_MODES = {
    "near": (640, 480, 30, 4.7),
    "mid": (1280, 720, 30, 7.0),
    "far": (1920, 1080, 30, 10.5),
}
CAPTURE_WIDTH, CAPTURE_HEIGHT, CAPTURE_FPS = 1280, 720, 15

# Depth is requested at 640x480 whatever the colour resolution is, because
# rs.align resamples it to the colour frame anyway and the smaller stream costs
# less USB bandwidth. 640x480 z16 is also the mode the D435i offers at every
# frame rate.
DEPTH_WIDTH, DEPTH_HEIGHT = 640, 480

# How long to wait for a frame before giving up on it and going round again.
#
# This doubles as the shutdown latency: the capture thread can only notice that
# state["running"] went false between attempts, and the main thread waits for
# it so that pipe.stop() runs on a live interpreter. It was 5000 ms, which made
# Ctrl-C take five seconds -- long enough that the process was being torn down
# with a wait_for_frames still in flight. At 30 fps a frame is 33 ms, so two
# seconds is still a very generous hiccup allowance.
FRAME_WAIT_MS = 2000

# ---------------------------------------------------------------- approach
# Go to a named person: yaw to put their face on the centre line, walk forward,
# and stop at a standoff. Only the target's face steers -- everyone else in
# frame is ignored for both aiming and stopping.
#
# One click starts a mission, and the mission owns the robot until it arrives,
# the operator stops it, or a budget runs out. Losing sight of the face is an
# ordinary event in the middle of a walk -- a head turning, someone crossing in
# front, motion blur at 0.1 m/s -- and it used to END the mission, which is why
# the button had to be pressed over and over to cross a room. Losing the face
# is now a phase, not an ending:
#
#   searching   stand still, then sweep toward where the face last was
#   confirming  the face is back: hold still until the name has held
#   approaching walk and yaw toward the face
#
# Every reacquisition goes through confirming again, so the robot never lunges
# at a name that flickered for a single frame.
MOVE_PHASE_SEARCH = "searching"
MOVE_PHASE_CONFIRM = "confirming"
MOVE_PHASE_APPROACH = "approaching"

# How long the face must keep its name, standing still, before the first walk.
MOVE_CONFIRM_S = 5.0
# After the robot has already walked once, a reacquisition only needs this long
# -- motion blur from the robot's own gait was forcing a full 5 s halt after
# every brief dropout, which looked like "two steps then nothing".
MOVE_RESUME_CONFIRM_S = 1.5
# A dropout inside confirm holds the countdown where it is rather than
# restarting it; only a sustained one sends the mission back to searching.
MOVE_CONFIRM_LOST_S = 1.0
# Mid-stride misses are normal while the robot is moving -- blur, a head turn,
# someone crossing the frame. Coast on the last heading this long before
# searching; 0.6 s was short enough that two steps then a halt was typical.
MOVE_LOST_GRACE_S = 2.5
# Forward speed while the face is briefly off-screen inside that grace window.
SEEK_COAST_VX_RATIO = 0.6

# Searching. Most losses resolve on their own within a second or two, so stand
# still first: turning immediately walks the camera away from a face that was
# about to come back. The sweep then leads toward the side the face was last
# seen on and reverses periodically, so the robot scans rather than spinning on
# the spot indefinitely.
SEARCH_HOLD_S = 1.5
SEARCH_OMEGA = 0.25
SEARCH_SWEEP_S = 3.0
DEFAULT_SEARCH_TIMEOUT = 90.0

# Zero-velocity commands while waiting go out at this interval: often enough
# that the robot is definitely still, rarely enough not to flood DDS at the
# loop rate.
HALT_INTERVAL_S = 0.2

MOVE_STEP_INTERVAL = 0.20

# The standoff, measured camera-to-face. Anything under STOP_DISTANCE_FLOOR_M
# is refused: that is inside arm's reach of a 35 kg robot and there is no
# margin left for depth noise or a person leaning in.
DEFAULT_STOP_DISTANCE_M = 0.55
STOP_DISTANCE_FLOOR_M = 0.20
# D435 depth lies and goes invalid below ~0.40 m; face-size is used instead.
DEPTH_MIN_RELIABLE_M = 0.40
# No forward creep once this close — MIN_MOVE_VX was overshooting the standoff.
STOP_CREEP_MARGIN_M = 0.05

# A detection box is a bit wider than the face inside it. 0.16 m is the width
# that made SCRFD's boxes agree with a tape measure on this camera, and it only
# feeds the pixel-size fallback -- depth is used whenever it is available.
FACE_WIDTH_M = 0.16

# Below this the face fills the frame, detection starts dropping it, and losing
# the target means we are on top of them rather than that they walked off.
NEAR_ARRIVAL_M = 0.70

# Ease off over the last stride so the stop is not a lurch.
SLOW_RADIUS_M = 1.20
MIN_MOVE_VX = 0.05

# SetVelocity acks even when the robot is not in a walking FSM. Measured on
# this G1: FSM 501 is regular walking mode (remote R1+Y). Stop early when
# forward commands run but distance does not shrink -- do not burn the full
# move-timeout budget against a robot that is standing still.
FSM_WALKING = 501
STALL_CHECK_WALKED_S = 6.0
STALL_MIN_PROGRESS_M = 0.12
STALL_MIN_PX_GAIN = 0.08
# Forward chunks must overlap generously; 0.35 s validity at 0.20 s intervals
# was enough for the SDK to ack but the gait still died after a stride or two.
FORWARD_CHUNK_MIN_S = 1.0

# Yaw control. err is the face's horizontal offset from centre, as a fraction
# of half the frame width, so it runs -1 (hard left) to +1 (hard right).
TURN_GAIN = 0.9
TURN_DEADBAND = 0.06
# Beyond this, forward speed scales down rather than going to zero -- a robot
# that will only turn in place never closes distance when the face sits near
# the frame edge, which is where it lands while walking.
TURN_ONLY_ERR = 0.45
TURN_MIN_VX_RATIO = 0.35      # slowest forward fraction at full offset

DEFAULT_MOVE_VX = 0.12
DEFAULT_MAX_OMEGA = 0.40

# 0 = no cap: follow until the standoff is reached, the operator stops it, or a
# real failure is detected. A mission is meant to end because it succeeded, not
# because a clock ran out on a chase that was still closing.
#
# Was 30 s, which could never work for a target who moves. Effective ground
# speed is ~0.04-0.08 m/s -- commanded 0.12, scaled down by TURN_MIN_VX_RATIO
# whenever there is yaw to correct and again inside SLOW_RADIUS_M, then only
# 50-70% of that is achieved -- so 30 s bought 1.2-2.5 m of travel per click.
# Measured: it gave up 1.56 m short of a walking target (tape-measured; the
# depth reading agreed to 10 cm once camera-to-face slant is accounted for).
#
# What still ends a mission, so removing this leaves the robot bounded:
#   - arrival at cfg.stop_distance_m, over MOVE_REACHED_STABLE_FRAMES frames
#   - the operator deadman (now the ONLY bound on total walking time)
#   - stall detection: commands sent, no closest-approach progress -- the check
#     that catches a robot which is not in walking FSM 501 at all
#   - search_timeout: no sighting of the target for that long
#   - Stop movement, which needs no token
# An obstacle does NOT end the mission: it halts and waits, then resumes.
DEFAULT_MOVE_TIMEOUT = 0.0

# Operator deadman.
#
# move_loop runs ON THE ROBOT and drives the control board over eth0, so it is
# entirely unaffected by the wifi link to whoever is watching. Without this, a
# dropped wifi link leaves the robot walking toward the target with the Stop
# button unreachable and the video frozen -- the operator has lost contact and
# the robot has not noticed. The half-second SetVelocity duration in
# g1_locomotion does NOT cover this: it expires commands when the publisher
# stops, and here the publisher is perfectly healthy.
#
# So an approach requires the browser to keep checking in. The dashboard polls
# /api/status every 400 ms; the video stream also refreshes the heartbeat.
#
# 1.5 s was too tight: a mission stands still for MOVE_CONFIRM_S before its
# first walk command, so the robot often took one step and stopped with "lost
# contact with the dashboard". Background tabs and wifi jitter need headroom
# too. The grace at the start covers confirming and the first commands even if
# the browser's first poll is late.
#
# This is enforced in dry-run too, deliberately.
MOVE_HEARTBEAT_TIMEOUT_S = 8.0
MOVE_HEARTBEAT_GRACE_S = 3.0

# Require several consecutive "close enough" readings before declaring arrival.
# One noisy depth frame at 1.00 m standoff was enough to end the approach after
# a single step.
MOVE_REACHED_STABLE_FRAMES = 4

# Obstacle guard floor: below this the person you are walking toward IS the
# nearest thing in the corridor, so blocking would mean stopping on them.
OBSTACLE_MIN_LIMIT_M = 0.45

# Forward corridor obstacle guard, from the same aligned depth frame. This is a
# coarse stop-and-wait check on what is directly ahead, not the lidar guard in
# g1_navigation_ws -- it sees only the camera's field of view and nothing at
# knee height outside it.
# Measured on the G1 2026-08-29, not guessed. With a person at ~2.2 m the
# corridor reads 2.13-2.87 m (their body and the background); a chair placed
# between robot and person reads 1.30-1.41 m. 1.8 sits in that gap with 0.33 m
# of headroom against a false stop and 0.39 m of margin to catch the chair.
#
# It was 0.55, which never fired on anything real: the standoff is 0.30, so the
# guard had a 25 cm band to act in, on a robot whose feet are ahead of its
# camera. A chair a metre away was ignored. The corridor sensing was always
# correct -- only this ceiling was wrong.
DEFAULT_OBSTACLE_M = 1.80
OBSTACLE_MARGIN_M = 0.30      # how much nearer than the target something must
                              # be to count as an obstacle rather than as them.
                              # This cap is what stops the person you are
                              # walking toward from reading as the obstacle,
                              # and it keeps working as the robot closes in:
                              # at 1 m out the limit becomes 0.70, so their own
                              # body stays clear of it while a second person
                              # stepping between still trips it.
CORRIDOR_WIDTH_FRAC = 0.35
CORRIDOR_TOP_FRAC = 0.25
CORRIDOR_BOTTOM_FRAC = 0.70
# How much of the corridor something has to fill before it counts. A plain
# minimum trips on single noisy pixels; the 5th percentile needed an object to
# cover a twentieth of the view, which a chair leg at two metres does not. Of
# the two ways to be wrong, stopping for nothing is the cheap one.
CORRIDOR_PERCENTILE = 2.0
# How much of the corridor something has to fill before it counts. A plain
# minimum trips on single noisy pixels; the 5th percentile needed an object to
# cover a twentieth of the view, which a chair leg at two metres does not. Of
# the two ways to be wrong, stopping for nothing is the cheap one.
CORRIDOR_PERCENTILE = 2.0

app = Flask(__name__)

state = {
    "threshold": 0.45,
    "fps": 0.0,          # rate the browser is being fed
    "infer_fps": 0.0,    # rate detection+recognition manages
    "faces": [],
    "raw": None,         # newest frame from the camera, unannotated
    "raw_version": 0,
    "frame": None,       # newest annotated + JPEG-encoded frame
    "frame_version": 0,
    "viewers": 0,        # open /video_feed streams; 0 means skip annotate+encode
    "running": True,
    "detect_ms": 0.0,
    "embed_ms": 0.0,
    "history": deque(maxlen=40),
    "depth": None,       # newest depth frame, aligned to colour, raw uint16
    "depth_scale": 0.0,  # multiply by this for metres
    "depth_ok": False,   # whether the source has a depth stream at all
    "fx": 0.0,           # colour focal length in pixels
}
move_state = {
    "active": False,
    "target": "",
    "execute": False,
    "status": "idle",
    "message": "idle",
    "target_px": 0,
    "target_dist": None,
    "dist_source": "",
    "offset": 0.0,
    "omega": 0.0,
    "obstacle_dist": None,
    "blocked": False,
    "started_at": 0.0,
    "last_beat": 0.0,    # last time the controlling browser checked in
}
lock = threading.Lock()


def _note_operator_alive() -> None:
    """Record that the controlling browser is still there. Takes the lock."""
    with lock:
        move_state["last_beat"] = time.perf_counter()


# ------------------------------------------------------------------- accounts
# Credentials live outside the workspace so no password or hash is ever in the
# tree that gets rsynced between machines.
AUTH_PATH = Path.home() / "fr_auth.json"


def load_auth() -> dict | None:
    """The stored account, or None if nobody has run --set-login yet."""
    if not AUTH_PATH.exists():
        return None
    try:
        data = json.loads(AUTH_PATH.read_text())
    except (ValueError, OSError):
        return None
    if not all(data.get(k) for k in ("username", "hash", "secret")):
        return None
    return data


def save_auth(username: str, password: str) -> None:
    from werkzeug.security import generate_password_hash

    existing = load_auth()
    data = {
        "username": username,
        "hash": generate_password_hash(password),
        # The cookie-signing key is kept WITH the credentials, and reused if one
        # already exists. A fresh key per start would silently log everyone out
        # on every restart -- which is exactly the friction the login page is
        # here to remove.
        "secret": existing["secret"] if existing else secrets.token_hex(32),
    }
    AUTH_PATH.write_text(json.dumps(data, indent=2))
    AUTH_PATH.chmod(0o600)


def _authed() -> bool:
    """A logged-in session, or the shared token. Either is enough.

    Two mechanisms because they serve different callers: people get a cookie
    that outlives a restart, scripts and curl keep using the token and never
    have to fill in a form.
    """
    if session.get("user"):
        return True
    want = app.config.get("TOKEN") or ""
    got = (request.headers.get("X-FR-Token")
           or request.args.get("token") or "")
    # compare_digest rather than == so a wrong token cannot be recovered a
    # character at a time from response timings.
    return bool(want) and secrets.compare_digest(got, want)


def require_auth(view):
    """Gate everything that can move the robot, retune it, or show the camera.

    Deliberately NOT applied to /api/stop_move: the stop path must work for
    someone who never logged in, and the worst an unauthorised stop can do is
    halt a robot.
    """
    @functools.wraps(view)
    def wrapped(*args, **kwargs):
        if _authed():
            return view(*args, **kwargs)
        # A person gets sent somewhere they can act on; an XHR gets JSON it can
        # parse. Handing raw JSON to someone who reopened a stale bookmarked
        # token was the whole problem -- it looked like a crash, not a login.
        if request.path.startswith("/api/"):
            return jsonify(ok=False, error="not authenticated"), 403
        return redirect("/login")
    return wrapped


def _reset_move_readings() -> None:
    """Clear the per-frame approach readings. Call with the lock held."""
    move_state["target_px"] = 0
    move_state["target_dist"] = None
    move_state["dist_source"] = ""
    move_state["offset"] = 0.0
    move_state["omega"] = 0.0
    move_state["obstacle_dist"] = None
    move_state["blocked"] = False

LOGIN_PAGE = """<!doctype html>
<title>Face Recognition — Sign in</title>
<style>
  :root { color-scheme: dark; }
  * { box-sizing: border-box; }
  body { margin:0; min-height:100vh; display:flex; align-items:center;
         justify-content:center; background:#0d1117; color:#e6edf3;
         font:14px/1.5 ui-monospace,SFMono-Regular,Menlo,monospace; }
  form { background:#161b22; border:1px solid #30363d; border-radius:8px;
         padding:26px; width:320px; }
  h1 { font-size:15px; margin:0 0 4px; font-weight:600; }
  p.sub { color:#8b949e; font-size:12px; margin:0 0 18px; }
  label { display:block; font-size:11px; text-transform:uppercase;
          letter-spacing:.08em; color:#8b949e; margin:12px 0 4px; }
  input { width:100%; padding:9px 10px; border-radius:6px; background:#0d1117;
          border:1px solid #30363d; color:#e6edf3; font:inherit; }
  input:focus { outline:none; border-color:#58a6ff; }
  button { width:100%; margin-top:18px; padding:10px; border:0; cursor:pointer;
           border-radius:6px; background:#238636; color:#fff; font:inherit;
           font-weight:600; }
  button:hover { background:#2ea043; }
  .err { margin-top:14px; padding:8px 10px; border-radius:6px; font-size:12px;
         background:rgba(248,81,73,.15); border-left:3px solid #f85149; }
  .note { margin-top:16px; color:#8b949e; font-size:11px; line-height:1.5; }
</style>
<form method="post">
  <h1>Face Recognition</h1>
  <p class="sub">Unitree G1 &middot; sign in to view the dashboard</p>
  <label for="u">Username</label>
  <input id="u" name="username" autocomplete="username" autofocus required>
  <label for="p">Password</label>
  <input id="p" name="password" type="password"
         autocomplete="current-password" required>
  <button type="submit">Sign in</button>
  {% if error %}<div class="err">{{ error }}</div>{% endif %}
  <p class="note">Stays signed in for 12 hours, across dashboard restarts.
  Stopping the robot never requires signing in.</p>
</form>
"""

NO_ACCOUNT_PAGE = """<!doctype html>
<title>Face Recognition — No account set</title>
<style>
  :root { color-scheme: dark; }
  body { margin:0; min-height:100vh; display:flex; align-items:center;
         justify-content:center; background:#0d1117; color:#e6edf3;
         font:14px/1.6 ui-monospace,SFMono-Regular,Menlo,monospace; }
  div { max-width:460px; background:#161b22; border:1px solid #30363d;
        border-radius:8px; padding:26px; }
  h1 { font-size:15px; margin:0 0 12px; }
  code { background:#0d1117; border:1px solid #30363d; border-radius:4px;
         padding:2px 6px; display:inline-block; margin-top:6px; }
  p { color:#8b949e; font-size:12px; }
</style>
<div>
  <h1>No login account has been set up yet</h1>
  <p>Create one on the robot, once. It is stored outside the workspace and
     survives restarts and redeploys:</p>
  <code>./start.sh --set-login</code>
  <p>Until then the dashboard is reachable with the startup token only:
     <code>?token=...</code> from the line <code>start.sh</code> prints.</p>
</div>
"""

PAGE = """<!doctype html>
<title>Face Recognition — Live</title>
<style>
  :root { color-scheme: dark; }
  * { box-sizing: border-box; }
  body { margin:0; background:#0d1117; color:#e6edf3;
         font:14px/1.5 ui-monospace,SFMono-Regular,Menlo,monospace; }
  header { padding:14px 20px; border-bottom:1px solid #30363d;
           display:flex; align-items:center; gap:16px; flex-wrap:wrap; }
  h1 { font-size:15px; margin:0; font-weight:600; letter-spacing:.02em; }
  .dot { width:8px; height:8px; border-radius:50%; background:#3fb950;
         animation:pulse 2s infinite; }
  @keyframes pulse { 50% { opacity:.35 } }
  main { display:grid; grid-template-columns:minmax(0,1fr) 320px; gap:18px; padding:18px; }
  @media (max-width:900px){ main { grid-template-columns:1fr } }
  canvas { width:100%; border-radius:8px; border:1px solid #30363d; display:block;
           background:#000; }
  .panel { background:#161b22; border:1px solid #30363d; border-radius:8px; padding:14px; }
  .panel h2 { font-size:11px; text-transform:uppercase; letter-spacing:.08em;
              color:#8b949e; margin:0 0 10px; font-weight:600; }
  .stat { display:flex; justify-content:space-between; padding:3px 0; }
  .stat b { font-weight:600; color:#58a6ff; }
  .face { display:flex; justify-content:space-between; align-items:center;
          padding:7px 9px; border-radius:6px; margin-bottom:6px; }
  .known { background:rgba(63,185,80,.14); border-left:3px solid #3fb950; }
  .unknown { background:rgba(210,153,34,.14); border-left:3px solid #d29922; }
  .score { font-variant-numeric:tabular-nums; opacity:.85; }
  input[type=range]{ width:100%; margin:8px 0 4px; accent-color:#58a6ff; }
  .thr { display:flex; justify-content:space-between; font-size:12px; color:#8b949e; }
  .bar { height:5px; background:#21262d; border-radius:3px; overflow:hidden; margin-top:3px; }
  .bar i { display:block; height:100%; background:#58a6ff; }
  .empty { color:#8b949e; font-style:italic; padding:6px 0; }
  .go-btns { display:flex; flex-wrap:wrap; gap:6px; margin:8px 0; }
  .go-btns button { flex:1 1 45%; padding:8px 6px; border-radius:6px;
    border:1px solid #30363d; background:#21262d; color:#e6edf3; cursor:pointer;
    font:inherit; font-size:12px; }
  .go-btns button:hover { border-color:#58a6ff; color:#58a6ff; }
  .go-btns button.active { border-color:#3fb950; color:#3fb950; }
  #stopMove { width:100%; margin-top:6px; padding:8px; border-radius:6px;
    border:1px solid #f85149; background:rgba(248,81,73,.12); color:#f85149;
    cursor:pointer; font:inherit; }
  #moveStatus { margin-top:8px; font-size:12px; color:#8b949e; min-height:2.5em; }
  .move-exec { display:flex; align-items:center; gap:8px; font-size:12px;
               color:#8b949e; margin-top:4px; }
</style>
<header>
  <span class="dot"></span>
  <h1>FACE RECOGNITION — LIVE</h1>
  <span id="src" style="color:#8b949e"></span>
  <a href="/logout" style="margin-left:auto;color:#8b949e;font-size:12px;
     text-decoration:none;border:1px solid #30363d;border-radius:6px;
     padding:4px 10px">Sign out</a>
</header>
<main>
  <div>
    <canvas id="cv" width="640" height="480"></canvas>
    <div id="linkstate" style="margin-top:6px;font-size:12px;color:#8b949e"></div>
  </div>
  <div>
    <div class="panel">
      <h2>Detected now</h2>
      <div id="faces"><div class="empty">no faces</div></div>
    </div>
    <div class="panel" style="margin-top:14px">
      <h2>Threshold <span id="tval" style="color:#58a6ff"></span></h2>
      <input type="range" id="thr" min="0.20" max="0.75" step="0.01">
      <div class="thr"><span>0.20 loose</span><span>0.75 strict</span></div>
      <div id="warn" style="display:none;margin-top:10px;padding:8px 10px;
           border-radius:6px;background:rgba(248,81,73,.15);
           border-left:3px solid #f85149;font-size:12px">
        Below the tuned value of <b id="tuned"></b> — strangers can be given a
        name here. Measured false-match rate rises sharply under it.
      </div>
      <p style="color:#8b949e;font-size:12px;margin:10px 0 0">
        Measured: impostor max <b style="color:#d29922" id="imax">—</b>,
        genuine min <b style="color:#3fb950" id="gmin">—</b>.
      </p>
    </div>
    <div class="panel" style="margin-top:14px">
      <h2>Go to person</h2>
      <label class="move-exec"><input type="checkbox" id="moveExec">
        Enable walk (robot turns and walks)</label>
      <div id="moveDryRun" style="display:none;margin:8px 0;padding:8px 10px;
           border-radius:6px;background:rgba(210,153,34,.15);
           border-left:3px solid #d29922;font-size:12px;color:#d29922">
        Dry run — the robot will not move. Tick Enable walk above, then click
        Go to again on the same person.
      </div>
      <div class="go-btns" id="goBtns"></div>
      <button type="button" id="stopMove">Stop movement</button>
      <div id="moveStatus">idle</div>
      <div class="stat"><span>distance</span><b id="mdist">—</b></div>
      <div class="stat"><span>aim offset</span><b id="moff">—</b></div>
      <div class="stat"><span>path ahead</span><b id="mobs">—</b></div>
      <div class="stat"><span>link</span><b id="mbeat">—</b></div>
    </div>
    <div class="panel" style="margin-top:14px">
      <h2>Performance</h2>
      <div class="stat"><span>video FPS</span><b id="fps">—</b></div>
      <div class="stat"><span>inference FPS</span><b id="ifps">—</b></div>
      <div class="stat"><span>detect</span><b id="dms">—</b></div>
      <div class="stat"><span>embed / face</span><b id="ems">—</b></div>
      <div class="stat"><span>enrolled</span><b id="enr">—</b></div>
    </div>
  </div>
</main>
<script>
// The shared token, substituted in when this page was served. Every request
// that can move the robot or change the threshold carries it, and /api/status
// carries it too -- that poll is what tells the robot the operator is still
// here, so losing this page stops an approach within MOVE_HEARTBEAT_TIMEOUT_S.
// Stop deliberately does not need it.
const TOKEN = "__FR_TOKEN__";
const AUTH = {'X-FR-Token': TOKEN};
const JSON_AUTH = {'Content-Type': 'application/json', 'X-FR-Token': TOKEN};

// ---------------------------------------------------------------- video
// An <img> pointed at an MJPEG endpoint dies silently when the connection
// drops -- the picture freezes and the only cure is reloading the page. This
// reads the stream explicitly, draws to a canvas, and reconnects on its own if
// frames stop arriving, so a wifi hiccup costs a second rather than a refresh.
const cv = document.getElementById('cv');
const ctx = cv.getContext('2d', {alpha:false});
const linkstate = document.getElementById('linkstate');
let lastFrame = 0, shown = 0, clientFps = 0, reconnects = 0;

function findMarker(buf, marker, from) {
  for (let i = from; i < buf.length - 1; i++)
    if (buf[i] === marker[0] && buf[i+1] === marker[1]) return i;
  return -1;
}

async function streamVideo() {
  let controller = new AbortController();
  // Watchdog: if no frame arrives for 3s the connection is dead in practice,
  // even if the socket is still nominally open. Abort so we reconnect.
  const watchdog = setInterval(() => {
    if (lastFrame && performance.now() - lastFrame > 3000) controller.abort();
  }, 1000);
  try {
    linkstate.textContent = 'connecting…';
    const res = await fetch('/video_feed', {signal: controller.signal,
      cache:'no-store', headers: AUTH, credentials: 'same-origin'});
    const reader = res.body.getReader();
    let buf = new Uint8Array(0);
    for (;;) {
      const {done, value} = await reader.read();
      if (done) break;
      const merged = new Uint8Array(buf.length + value.length);
      merged.set(buf); merged.set(value, buf.length);
      buf = merged;
      for (;;) {
        const s = findMarker(buf, [0xFF,0xD8], 0);
        if (s < 0) break;
        const e = findMarker(buf, [0xFF,0xD9], s + 2);
        if (e < 0) { if (s > 0) buf = buf.slice(s); break; }
        const jpeg = buf.slice(s, e + 2);
        buf = buf.slice(e + 2);
        try {
          const bmp = await createImageBitmap(new Blob([jpeg], {type:'image/jpeg'}));
          if (cv.width !== bmp.width) { cv.width = bmp.width; cv.height = bmp.height; }
          ctx.drawImage(bmp, 0, 0);
          bmp.close();
          shown++; lastFrame = performance.now();
        } catch (err) { /* a partial frame is not worth stopping for */ }
      }
      if (buf.length > 4_000_000) buf = new Uint8Array(0);   // desync guard
    }
  } catch (err) {
    // fall through to reconnect
  } finally {
    clearInterval(watchdog);
  }
  reconnects++;
  linkstate.textContent = 'reconnecting…';
  setTimeout(streamVideo, 800);
}

setInterval(() => {
  clientFps = shown; shown = 0;
  linkstate.textContent = `link ok — ${clientFps} fps rendered` +
    (reconnects ? `  ·  ${reconnects} reconnect${reconnects>1?'s':''}` : '');
}, 1000);
streamVideo();

// ---------------------------------------------------------------- controls
const thr = document.getElementById('thr');
let dragging = false;
// Last refusal from /api/go_to, held until the next request succeeds.
let goError = '';
thr.addEventListener('input', () => {
  dragging = true;
  document.getElementById('tval').textContent = (+thr.value).toFixed(2);
});
thr.addEventListener('change', async () => {
  await fetch('/api/threshold', {method:'POST', headers: JSON_AUTH,
    body: JSON.stringify({threshold:+thr.value})});
  dragging = false;
});
async function tick(){
  try {
    // credentials keeps the session cookie alive for login-based access; the
    // token header covers token-URL access. Both paths must refresh the deadman.
    const r = await fetch('/api/status', {headers: AUTH, credentials: 'same-origin'});
    if (!r.ok) {
      document.getElementById('mbeat').textContent = `poll ${r.status}`;
      document.getElementById('mbeat').style.color = '#f85149';
      return;
    }
    const s = await r.json();
    if (!dragging) { thr.value = s.threshold;
      document.getElementById('tval').textContent = s.threshold.toFixed(2); }
    document.getElementById('fps').textContent = s.fps.toFixed(1);
    document.getElementById('ifps').textContent = s.infer_fps.toFixed(1);
    document.getElementById('dms').textContent = s.detect_ms.toFixed(1)+' ms';
    document.getElementById('ems').textContent = s.embed_ms.toFixed(1)+' ms';
    document.getElementById('enr').textContent = s.enrolled.join(', ') || 'none';
    document.getElementById('src').textContent = s.source;
    document.getElementById('tuned').textContent = s.tuned_threshold.toFixed(2);
    document.getElementById('imax').textContent =
      s.impostor_max != null ? s.impostor_max.toFixed(3) : 'n/a';
    document.getElementById('gmin').textContent =
      s.genuine_min != null ? s.genuine_min.toFixed(3) : 'n/a';
    document.getElementById('warn').style.display =
      (s.threshold < s.tuned_threshold - 1e-9) ? 'block' : 'none';
    const box = document.getElementById('faces');
    if (!s.faces.length) { box.innerHTML = '<div class="empty">no faces</div>'; }
    else box.innerHTML = s.faces.map(f => {
      const pct = Math.max(0, Math.min(100, (f.score+0.2)/0.95*100));
      const tgt = s.move && s.move.active && f.label === s.move.target;
      return `<div class="face ${f.label!=='unknown'?'known':'unknown'}"` +
        (tgt ? ' style="outline:2px solid #58a6ff"' : '') + `>
        <span>${f.label}</span><span class="score">${f.score.toFixed(3)}</span></div>
        <div class="bar"><i style="width:${pct}%"></i></div>`;
    }).join('');
    const ms = document.getElementById('moveStatus');
    if (goError && !(s.move && s.move.active)) {
      ms.textContent = goError;
      ms.style.color = '#f85149';
    } else if (s.move) {
      const m = s.move;
      document.getElementById('moveDryRun').style.display =
        (m.active && !m.execute) ? 'block' : 'none';
      ms.textContent = m.message || m.status;
      // Searching shares the amber of blocked: the mission is still live, but
      // the robot is not making progress and the operator should look up.
      ms.style.color = m.status === 'done' ? '#3fb950'
        : (m.status === 'lost' || m.status === 'stopped' ? '#f85149'
           : (m.status === 'blocked' || m.status === 'searching' ? '#d29922'
              : (m.active ? '#58a6ff' : '#8b949e')));
      document.getElementById('mdist').textContent = m.target_dist == null
        ? '—' : `${m.target_dist.toFixed(2)} m (${m.dist_source})`;
      const off = document.getElementById('moff');
      off.textContent = m.active
        ? `${(m.offset*100).toFixed(0)}%  ω ${m.omega.toFixed(2)}` : '—';
      off.style.color = Math.abs(m.offset) > 0.30 ? '#d29922' : '#58a6ff';
      const obs = document.getElementById('mobs');
      if (!s.depth_ok) { obs.textContent = 'no depth'; obs.style.color = '#8b949e'; }
      // The corridor is only evaluated while a target is locked. Saying
      // "clear" the rest of the time reads as "nothing in the way" when it
      // actually means "not looking", which is the opposite of reassuring.
      else if (!m.active) { obs.textContent = 'not checked'; obs.style.color = '#8b949e'; }
      else if (m.obstacle_dist == null) { obs.textContent = 'no reading'; obs.style.color = '#8b949e'; }
      else {
        obs.textContent = `${m.obstacle_dist.toFixed(2)} m`;
        obs.style.color = m.blocked ? '#f85149' : '#3fb950';
      }
      const beat = document.getElementById('mbeat');
      if (m.active && m.heartbeat_age != null) {
        beat.textContent = `${m.heartbeat_age.toFixed(1)}s ago`;
        beat.style.color = m.heartbeat_age > 4 ? '#d29922' : '#3fb950';
      } else {
        beat.textContent = '—';
        beat.style.color = '#8b949e';
      }
    }
    const gb = document.getElementById('goBtns');
    const enrolled = s.enrolled || [];
    if (gb.dataset.n !== enrolled.join(',')) {
      gb.dataset.n = enrolled.join(',');
      gb.innerHTML = enrolled.map(n =>
        `<button type="button" data-name="${n}">Go to ${n}</button>`).join('');
      gb.querySelectorAll('button').forEach(btn => {
        btn.addEventListener('click', async () => {
          const exec = document.getElementById('moveExec').checked;
          const r = await fetch('/api/go_to', {method:'POST', headers: JSON_AUTH,
            body: JSON.stringify({target: btn.dataset.name, execute: exec})});
          const j = await r.json().catch(() => ({}));
          // A refusal has to survive the next status poll, which would
          // otherwise overwrite it 400 ms later with "idle".
          goError = (j && j.ok) ? '' : (j.error || 'request refused');
        });
      });
    }
    if (s.move && s.move.active && s.move.target) {
      gb.querySelectorAll('button').forEach(btn => {
        btn.classList.toggle('active', btn.dataset.name === s.move.target);
      });
    } else {
      gb.querySelectorAll('button').forEach(btn => btn.classList.remove('active'));
    }
  } catch(e) {
    document.getElementById('mbeat').textContent = 'poll failed';
    document.getElementById('mbeat').style.color = '#f85149';
  }
}
document.getElementById('stopMove').addEventListener('click', async () => {
  await fetch('/api/stop_move', {method:'POST'});
});
setInterval(tick, 400); tick();
</script>
"""


def open_source(source):
    """Return (read, close, info) for either the RealSense or a V4L2 camera.

    read(with_depth=False) -> (ok, colour, depth_uint16 or None)

    `info` carries what the approach controller needs to turn what it sees into
    metres: whether there is a depth stream, the depth unit scale, and the
    colour focal length.
    """
    if source == "realsense":
        import pyrealsense2 as rs

        pipe = rs.pipeline()
        # Ask for the best frame rate the resolution can offer and fall back.
        # On a USB 3 link the D435i does 1080p at 30 fps; on USB 2 the same
        # resolution caps at 8. Negotiating means one config works on both the
        # G1 (USB 3.2) and the Go2 (USB 2.1, port blocked by the lidar) without
        # anyone having to remember which robot they are on.
        #
        # Depth is tried first at every rate, and dropped rather than allowed
        # to cost frame rate: it is what makes the 30 cm standoff and the
        # obstacle corridor measurements instead of guesses, but recognition
        # has to keep working on a camera that will not give it.
        started, depth_on, profile = None, False, None
        for fps in (CAPTURE_FPS, 30, 15, 8, 6):
            for want_depth in (True, False):
                cfg = rs.config()
                cfg.enable_stream(rs.stream.color, CAPTURE_WIDTH, CAPTURE_HEIGHT,
                                  rs.format.bgr8, fps)
                if want_depth:
                    cfg.enable_stream(rs.stream.depth, DEPTH_WIDTH, DEPTH_HEIGHT,
                                      rs.format.z16, fps)
                try:
                    profile = pipe.start(cfg)
                    started, depth_on = fps, want_depth
                    break
                except Exception:
                    continue
            if started is not None:
                break
        if started is None:
            raise RuntimeError(
                f"RealSense refused {CAPTURE_WIDTH}x{CAPTURE_HEIGHT} at any frame "
                "rate. Try a smaller --range, or check the USB link speed."
            )
        if started != CAPTURE_FPS:
            print(f"  camera negotiated {started} fps "
                  f"(asked for {CAPTURE_FPS}) at {CAPTURE_WIDTH}x{CAPTURE_HEIGHT}",
                  flush=True)
        if not depth_on:
            print("  no depth stream -- approach falls back to face size, and "
                  "the obstacle corridor is unavailable", flush=True)

        align = rs.align(rs.stream.color) if depth_on else None
        depth_scale = 0.0
        if depth_on:
            depth_scale = float(
                profile.get_device().first_depth_sensor().get_depth_scale())
        intrinsics = (profile.get_stream(rs.stream.color)
                      .as_video_stream_profile().get_intrinsics())

        def read(with_depth: bool = False):
            try:
                frames = pipe.wait_for_frames(timeout_ms=FRAME_WAIT_MS)
            except RuntimeError:
                # A timeout is not fatal. Letting it propagate killed the
                # capture thread outright, which froze the picture for good
                # with nothing in the log to say why -- the dashboard just
                # stopped updating. Report a bad read and let grab_loop decide.
                return False, None, None
            # Aligning depth onto the colour frame costs real milliseconds at
            # 720p on the Jetson, so it only runs while an approach is actually
            # reading the depth.
            use_depth = with_depth and align is not None
            if use_depth:
                frames = align.process(frames)
            colour_frame = frames.get_color_frame()
            if not colour_frame:
                return False, None, None
            colour = np.asanyarray(colour_frame.get_data())
            depth = None
            if use_depth:
                depth_frame = frames.get_depth_frame()
                if depth_frame:
                    # Copied: the buffer belongs to the frame, and the frame is
                    # released the moment this function returns.
                    depth = np.asanyarray(depth_frame.get_data()).copy()
            return True, colour, depth

        return read, pipe.stop, {"depth": depth_on, "depth_scale": depth_scale,
                                 "fx": float(intrinsics.fx)}

    cap = open_camera(int(source), width=CAPTURE_WIDTH, height=CAPTURE_HEIGHT)

    def read(with_depth: bool = False):
        ok, frame = cap.read()
        return ok, frame, None

    # A V4L2 node publishes no intrinsics. A ~60 degree horizontal field of
    # view puts fx near the frame width, which is only ever good enough for the
    # pixel-size fallback -- never for the standoff on the robot.
    return read, cap.release, {"depth": False, "depth_scale": 0.0,
                               "fx": float(CAPTURE_WIDTH)}


def grab_loop(source) -> None:
    """Pull frames as fast as the camera delivers them. No inference here.

    Inference used to run inline with capture, which meant the video stream
    could never be smoother than the slowest recognition frame -- every embed
    stalled the picture. Splitting them lets the video run at sensor rate while
    boxes update at whatever rate inference manages. A box lagging the picture
    by 30 ms is invisible; a stuttering video is not.
    """
    read, close, info = open_source(source)
    with lock:
        state["depth_ok"] = bool(info["depth"])
        state["depth_scale"] = float(info["depth_scale"])
        state["fx"] = float(info["fx"])
    misses = 0
    try:
        while state["running"]:
            with lock:
                want_depth = state["depth_ok"] and move_state["active"]
            ok, frame, depth = read(want_depth)
            if not ok:
                # Say so once rather than silently showing a frozen picture.
                misses += 1
                if misses in (1, 10, 100):
                    print(f"  camera returned no frame ({misses}x) -- retrying",
                          flush=True)
                time.sleep(0.02)
                continue
            misses = 0
            with lock:
                state["raw"] = frame
                state["raw_version"] += 1
                state["depth"] = depth
    finally:
        # Runs on the way out of a normal shutdown because main() waits for
        # this thread. If the interpreter were torn down first instead,
        # librealsense would destruct under an in-flight call and abort the
        # process -- which also left the camera claimed for the next run.
        close()
        print("  camera released", flush=True)


def annotate_loop() -> None:
    """Draw the most recent detections onto the most recent frame and encode."""
    stream_max_w = app.config.get("STREAM_MAX_WIDTH", STREAM_MAX_WIDTH)
    last = -1
    frames, t0 = 0, time.perf_counter()
    idle = False              # True while no viewer is attached
    while state["running"]:
        with lock:
            frame, version = state["raw"], state["raw_version"]
            faces = list(state["faces"])
            move_target = move_state["target"] if move_state["active"] else ""
            viewers = state["viewers"]
        if frame is None or version == last:
            time.sleep(0.002)
            continue
        last = version

        # With nobody watching, the annotated JPEG is built and thrown away --
        # ~30 full-resolution encodes a second, most of them during missions when
        # move_loop wants the CPU. state["fps"] means "rate the browser is being
        # fed", so reporting 0 here is the truth, not a gap.
        if viewers <= 0:
            if not idle:
                idle = True
                with lock:
                    state["fps"] = 0.0
            frames, t0 = 0, time.perf_counter()
            continue
        idle = False

        canvas = frame.copy()

        if move_target:
            # The line the robot is yawing to put the target's face on. Without
            # it a "walking toward" message and a face off to one side look the
            # same from the browser.
            mid = canvas.shape[1] // 2
            cv2.line(canvas, (mid, 0), (mid, canvas.shape[0]), (120, 120, 120), 1)

        for f in faces:
            x, y, w, h = f["box"]
            known = f["label"] != "unknown"
            is_target = move_target and f["label"] == move_target
            if is_target:
                colour = (255, 200, 0)
                thickness = 3
            else:
                colour = (0, 200, 0) if known else (0, 140, 255)
                thickness = 2
            cv2.rectangle(canvas, (x, y), (x + w, y + h), colour, thickness)
            text = f"{f['label']} {f['score']:.2f}"
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(canvas, (x, y - th - 9), (x + tw + 8, y), colour, -1)
            cv2.putText(canvas, text, (x + 4, y - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        frames += 1
        elapsed = time.perf_counter() - t0
        if elapsed >= 0.5:
            with lock:
                state["fps"] = frames / elapsed
            frames, t0 = 0, time.perf_counter()

        # Shrink only the outgoing picture. Boxes and labels are drawn at full
        # resolution first and scale down with it, which keeps this a two-line
        # change instead of rescaling every coordinate.
        if stream_max_w and canvas.shape[1] > stream_max_w:
            ratio = stream_max_w / canvas.shape[1]
            canvas = cv2.resize(canvas, (stream_max_w,
                                         max(1, int(round(canvas.shape[0] * ratio)))),
                                interpolation=cv2.INTER_AREA)

        ok, buf = cv2.imencode(".jpg", canvas, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        if ok:
            with lock:
                state["frame"] = buf.tobytes()
                state["frame_version"] += 1


def infer_loop(pipeline: FacePipeline, db: IdentityDB) -> None:
    """Detect and recognise on whatever the newest frame happens to be.

    Runs as fast as it can and simply skips frames it could not keep up with,
    so a slow inference pass delays the boxes rather than the video.
    """
    tracker = FaceTracker(history=SMOOTH_FRAMES)
    last = -1
    frame_no = 0
    while state["running"]:
        with lock:
            frame, version = state["raw"], state["raw_version"]
            threshold = state["threshold"]
        if frame is None or version == last:
            time.sleep(0.002)
            continue
        last = version
        frame_no += 1

        t = time.perf_counter()
        faces = pipeline.detect(
            frame,
            min_width=MIN_FACE_WIDTH_PX,
            scale=DETECT_SCALE,
            require_in_frame=True,
            min_confidence=MIN_DETECT_CONFIDENCE,
        )
        detect_ms = (time.perf_counter() - t) * 1000

        results, embed_ms = [], 0.0
        seen = set()
        for face in faces:
            track_id = tracker.assign(face.box)
            seen.add(track_id)
            if tracker.needs_recognition(track_id, RECOGNISE_EVERY, frame_no):
                t = time.perf_counter()
                emb = pipeline.embed(frame, face)
                embed_ms += (time.perf_counter() - t) * 1000
                all_scores = db.scores(emb)
                label = max(all_scores, key=all_scores.get)
                tracker.observe(track_id, label, all_scores[label], all_scores)
            label, score, _ = tracker.decide(track_id, threshold, HOLD_RATIO)
            results.append({"label": label, "score": score,
                            "px": face.width, "box": face.box})
        tracker.end_frame(seen)

        with lock:
            state["faces"] = results
            state["detect_ms"] = detect_ms
            state["embed_ms"] = embed_ms / max(len(faces), 1)
            state["infer_fps"] = 1000.0 / max(detect_ms + embed_ms, 1e-3)


def _public_move_state() -> dict:
    age = None
    if move_state["active"] and move_state["last_beat"] > 0:
        age = time.perf_counter() - move_state["last_beat"]
    return {
        "active": move_state["active"],
        "target": move_state["target"],
        "execute": move_state["execute"],
        "status": move_state["status"],
        "message": move_state["message"],
        "target_px": move_state["target_px"],
        "target_dist": move_state["target_dist"],
        "dist_source": move_state["dist_source"],
        "offset": move_state["offset"],
        "omega": move_state["omega"],
        "obstacle_dist": move_state["obstacle_dist"],
        "blocked": move_state["blocked"],
        "heartbeat_age": age,
    }


class MoveConfig(NamedTuple):
    stop_distance_m: float
    stop_px: float          # 0 means derive it from the camera's focal length
    vx: float
    max_omega: float
    obstacle_m: float
    # Seconds spent actually commanding forward motion, not wall-clock since
    # the click. Blocked, halted, and search time do not count.
    timeout: float
    # Give up only after this long with no sight of the target at all.
    search_timeout: float
    # How often a velocity command goes out, and how long each one is good for.
    step_interval: float
    chunk_s: float


def _size_distance(fx: float, px: int) -> float | None:
    """Metres from apparent face width, or None."""
    if fx > 0 and px > 0:
        return fx * FACE_WIDTH_M / px
    return None


def _approach_distance(depth_dist: float | None,
                       size_dist: float | None) -> tuple[float | None, str]:
    """Metres for approach/stop. Prefer face-size when depth is in the D435 blind
    zone or reads farther than size (depth often lags or sticks high up close)."""
    if depth_dist is None:
        return size_dist, "size" if size_dist is not None else ""
    if size_dist is None:
        return depth_dist, "depth"
    if depth_dist < DEPTH_MIN_RELIABLE_M or size_dist < depth_dist:
        return size_dist, "size"
    return depth_dist, "depth"


def _depth_at_box(depth, scale: float, box) -> float | None:
    """Median depth in metres over the middle of a face box, or None.

    The middle only: the edges of a detection box straddle the background, and
    a single stray reading off the wall behind someone's ear would otherwise
    put them a metre further away than they are. The median then throws out the
    speckle that survives.
    """
    x, y, w, h = box
    half_w = max(1, int(w * 0.2))
    half_h = max(1, int(h * 0.2))
    cx, cy = int(x + w / 2), int(y + h / 2)
    y0, y1 = max(0, cy - half_h), min(depth.shape[0], cy + half_h + 1)
    x0, x1 = max(0, cx - half_w), min(depth.shape[1], cx + half_w + 1)
    if y1 <= y0 or x1 <= x0:
        return None
    patch = depth[y0:y1, x0:x1].astype(np.float32) * scale
    valid = patch[(patch > 0.1) & (patch < 10.0)]
    if valid.size < 10:
        return None
    return float(np.median(valid))


def _corridor_distance(depth, scale: float) -> float | None:
    """Nearest thing in the forward corridor, in metres, or None if clear.

    A band across the middle of the frame rather than the whole of it: the top
    is ceiling and the bottom is floor, and at the camera's height on the G1
    the floor comes into view a couple of metres out and would read as a wall.
    A low percentile rather than the minimum, because a handful of noisy pixels
    is not an obstacle.
    """
    h, w = depth.shape[:2]
    x0 = int(w * (0.5 - CORRIDOR_WIDTH_FRAC / 2))
    x1 = int(w * (0.5 + CORRIDOR_WIDTH_FRAC / 2))
    y0, y1 = int(h * CORRIDOR_TOP_FRAC), int(h * CORRIDOR_BOTTOM_FRAC)
    if y1 <= y0 or x1 <= x0:
        return None
    patch = depth[y0:y1, x0:x1].astype(np.float32) * scale
    valid = patch[(patch > 0.15) & (patch < 6.0)]
    if valid.size < 50:
        return None
    return float(np.percentile(valid, CORRIDOR_PERCENTILE))


def _halt(loco: G1Locomotion | None, execute: bool) -> None:
    if execute and loco is not None:
        try:
            loco.stop()
        except Exception:
            pass


def _finish(loco, execute: bool, status: str, message: str) -> None:
    """Stop the robot and end the approach. Safe to call from any branch."""
    print(f"  approach {status}: {message}", flush=True)
    _halt(loco, execute)
    with lock:
        move_state["active"] = False
        move_state["status"] = status
        move_state["message"] = message


def _match_target(faces: list, target: str, threshold: float):
    """Largest face box for *target*, using the same hold floor as live labels."""
    hold = threshold * HOLD_RATIO
    candidates = [f for f in faces
                  if f["label"] == target and f["score"] >= hold]
    return max(candidates, key=lambda f: f["px"]) if candidates else None


def move_loop(loco: G1Locomotion | None, cfg: MoveConfig) -> None:
    """Run one go-to-person mission per click: confirm, walk, search if lost.

    The phases are described where MOVE_PHASE_SEARCH is defined. The property
    that matters is that losing the face changes the phase and never ends the
    mission: a person turning their head must not cost the operator a click.
    Only three things end it -- the standoff is reached, the operator stops it,
    or a budget expires -- and each of them stops the robot first.

    Only the target steers. Everyone else in frame is ignored entirely, which
    is the whole point of doing this off recognition rather than off detection.
    """
    heartbeat_timeout = app.config.get("HEARTBEAT_TIMEOUT", MOVE_HEARTBEAT_TIMEOUT_S)

    mission = 0.0             # started_at of the mission the FSM is running
    phase = MOVE_PHASE_SEARCH
    confirm_seen = 0.0        # seconds the face has held its name, vs MOVE_CONFIRM_S
    confirm_tick = 0.0
    lost_since = 0.0          # first frame of the current run of misses
    last_seen_at = 0.0
    search_since = 0.0
    sweep_since = 0.0
    sweep_sign = 1.0
    walked = 0.0              # seconds of forward motion only, vs cfg.timeout
    ever_approached = False   # resume confirm is shorter once a walk has started
    last_forward_at = 0.0
    last_logged_phase = ""
    reached = 0
    last_distance = None
    last_px = 0
    last_offset = 0.0
    last_cmd = None           # last approach command, re-sent across a dropout
    last_step = 0.0
    last_halt = 0.0
    stall_base_dist = None
    stall_base_px = 0
    stall_since_walked = 0.0

    def _log_phase(note: str) -> None:
        nonlocal last_logged_phase
        if note == last_logged_phase:
            return
        last_logged_phase = note
        print(f"  approach {note}", flush=True)

    def _send_step(step_vx: float, omega: float) -> tuple[bool, bool]:
        """Issue one locomotion chunk. (mission_ended, sent_to_robot)."""
        nonlocal last_step
        now = time.perf_counter()
        if now - last_step < cfg.step_interval:
            return False, False
        last_step = now
        if not execute or loco is None:
            return False, False
        try:
            loco.connect()
            chunk = cfg.chunk_s
            if step_vx > MIN_MOVE_VX * 0.5:
                chunk = max(cfg.chunk_s, FORWARD_CHUNK_MIN_S)
            code = loco.step(step_vx, omega, chunk)
            if code:
                _finish(loco, execute, "stopped",
                        f"SetVelocity returned {code} — is the robot "
                        "standing and in main operation mode?")
                return True, False
        except Exception as exc:
            _finish(loco, execute, "stopped", f"locomotion error: {exc}")
            return True, False
        return False, True

    def _note_forward(step_vx: float, sent: bool) -> None:
        """Accumulate approach budget only after a real forward command."""
        nonlocal walked, last_forward_at
        if not sent or not execute or step_vx <= MIN_MOVE_VX * 0.5:
            return
        now = time.perf_counter()
        if last_forward_at:
            walked += min(0.5, now - last_forward_at)
        last_forward_at = now

    def _reset_stall() -> None:
        nonlocal stall_base_dist, stall_base_px, stall_since_walked
        stall_base_dist = None
        stall_base_px = 0
        stall_since_walked = 0.0

    def _check_stall(cur_dist: float | None, cur_px: int) -> bool:
        """True if the mission should end: commands sent, robot did not move."""
        nonlocal stall_base_dist, stall_base_px, stall_since_walked
        if not execute or walked < STALL_CHECK_WALKED_S:
            return False
        if stall_base_dist is None:
            if cur_dist is not None:
                stall_base_dist = cur_dist
                stall_base_px = cur_px
                stall_since_walked = walked
            return False
        if walked - stall_since_walked < STALL_CHECK_WALKED_S:
            return False
        progressed = False
        if cur_dist is not None:
            progressed = cur_dist <= stall_base_dist - STALL_MIN_PROGRESS_M
        elif stall_base_px > 0 and cur_px > 0:
            progressed = cur_px >= stall_base_px * (1.0 + STALL_MIN_PX_GAIN)
        if progressed:
            _reset_stall()
            if cur_dist is not None:
                stall_base_dist = cur_dist
                stall_base_px = cur_px
                stall_since_walked = walked
            return False
        fsm = loco.fsm_id() if loco is not None else None
        base = (f"{stall_base_dist:.2f} m"
                if stall_base_dist is not None else f"{stall_base_px}px")
        hint = "hold R1+Y on the remote (~2 s) for walking mode"
        if fsm is not None and fsm != FSM_WALKING:
            hint = f"FSM is {fsm}, need {FSM_WALKING} — {hint}"
        _finish(loco, execute, "stopped",
                f"commands sent but robot not moving ({base} unchanged) "
                f"— {hint}")
        return True

    def _approach_vx(distance: float | None, offset: float) -> float:
        """Forward speed for closed-loop approach, never hard zero for yaw."""
        if distance is not None and distance <= cfg.stop_distance_m + STOP_CREEP_MARGIN_M:
            return 0.0
        step_vx = cfg.vx
        if distance is not None and distance < SLOW_RADIUS_M:
            span = max(SLOW_RADIUS_M - cfg.stop_distance_m, 1e-3)
            ratio = (distance - cfg.stop_distance_m) / span
            step_vx = max(MIN_MOVE_VX, cfg.vx * max(0.0, min(1.0, ratio)))
        off = min(1.0, abs(offset) / max(TURN_ONLY_ERR, 1e-3))
        floor = TURN_MIN_VX_RATIO
        step_vx *= floor + (1.0 - floor) * (1.0 - 0.5 * off)
        return max(MIN_MOVE_VX, step_vx)

    def _send_halt() -> None:
        """Hold still. Rate-limited: the waiting phases run at loop rate."""
        nonlocal last_halt
        now = time.perf_counter()
        if now - last_halt < HALT_INTERVAL_S:
            return
        last_halt = now
        _halt(loco, execute)

    while state["running"]:
        with lock:
            active = move_state["active"]
            target = move_state["target"]
            execute = move_state["execute"]
            started_at = move_state["started_at"]
            last_beat = move_state["last_beat"]
            threshold = state["threshold"]
            faces = list(state["faces"])
            depth = state["depth"]
            depth_scale = state["depth_scale"]
            fx = state["fx"]
            frame = state["raw"]

        if not active or not target:
            mission = 0.0
            time.sleep(0.05)
            continue

        now = time.perf_counter()

        # A new click is a new mission: nothing from the last one carries over.
        if started_at != mission:
            mission = started_at
            phase = MOVE_PHASE_SEARCH
            confirm_seen = confirm_tick = lost_since = 0.0
            last_seen_at = search_since = sweep_since = now
            sweep_sign = 1.0
            walked = last_forward_at = 0.0
            ever_approached = False
            reached = 0
            last_distance = None
            last_px = 0
            last_offset = 0.0
            last_cmd = None
            last_step = last_halt = 0.0
            last_logged_phase = ""
            _reset_stall()
            _log_phase(f"mission start -> {target}"
                       + ("" if execute else " (dry-run, no locomotion)"))

        # Deadman: the operator must still be watching. Grace at the start so
        # confirming + the first walk commands are not cut off by a late poll.
        silent = now - last_beat
        if now - started_at > MOVE_HEARTBEAT_GRACE_S and silent > heartbeat_timeout:
            _finish(loco, execute, "stopped",
                    f"lost contact with the dashboard ({silent:.1f}s) — stopped")
            continue

        frame_w = frame.shape[1] if frame is not None else CAPTURE_WIDTH

        # The largest match, not the first: two tracks can briefly carry the
        # same name, and the bigger box is the nearer person.
        match = _match_target(faces, target, threshold)

        px, distance, offset = 0, None, last_offset
        obstacle_dist, blocked = None, False

        if match is None:
            if lost_since == 0.0:
                lost_since = now
            with lock:
                move_state["target_px"] = 0
                move_state["target_dist"] = None
                move_state["dist_source"] = ""
                move_state["obstacle_dist"] = None
                move_state["blocked"] = False
        else:
            lost_since = 0.0
            last_seen_at = now
            px = int(match["px"])
            box_x, _, box_w, _ = match["box"]
            offset = ((box_x + box_w / 2.0) - frame_w / 2.0) / (frame_w / 2.0)
            last_offset = offset

            size_dist = _size_distance(fx, px)
            depth_dist = None
            if depth is not None and depth_scale > 0:
                depth_dist = _depth_at_box(depth, depth_scale, match["box"])
            distance, dist_source = _approach_distance(depth_dist, size_dist)
            if distance is not None:
                last_distance = distance
            last_px = px

            if depth is not None and depth_scale > 0:
                obstacle_dist = _corridor_distance(depth, depth_scale)
                if obstacle_dist is not None:
                    # The target's own body fills the corridor as we close on
                    # them, so only something meaningfully nearer counts.
                    limit = cfg.obstacle_m
                    if distance is not None:
                        limit = min(limit, distance - OBSTACLE_MARGIN_M)
                        limit = max(OBSTACLE_MIN_LIMIT_M, limit)
                    blocked = obstacle_dist < limit

            with lock:
                move_state["target_px"] = px
                move_state["target_dist"] = distance
                move_state["dist_source"] = dist_source
                move_state["offset"] = offset
                move_state["obstacle_dist"] = obstacle_dist
                move_state["blocked"] = blocked

        # ------------------------------------------------ phase transitions
        if match is not None:
            if phase == MOVE_PHASE_SEARCH:
                # Back in view. Every reacquisition earns its own confirm
                # window, so a one-frame flicker cannot restart a walk.
                phase = MOVE_PHASE_CONFIRM
                confirm_seen = confirm_tick = 0.0
                reached = 0
                kind = "resume" if ever_approached else "initial"
                need = MOVE_RESUME_CONFIRM_S if ever_approached else MOVE_CONFIRM_S
                _log_phase(f"found {target} -> confirming ({kind}, {need:.0f}s)")
        elif phase == MOVE_PHASE_CONFIRM:
            if now - lost_since >= MOVE_CONFIRM_LOST_S:
                phase = MOVE_PHASE_SEARCH
                search_since = sweep_since = now
        elif phase == MOVE_PHASE_APPROACH:
            if now - lost_since >= MOVE_LOST_GRACE_S:
                # Close in, the face overflows the frame and the detector drops
                # it. That is arrival, not a target walking away, and stopping
                # is the right reading either way.
                if last_distance is not None and last_distance <= NEAR_ARRIVAL_M:
                    _finish(loco, execute, "done",
                            f"reached {target} (~{last_distance:.2f} m, face "
                            "too close to keep tracking)")
                    continue
                phase = MOVE_PHASE_SEARCH
                search_since = sweep_since = now
                last_cmd = None
                _reset_stall()
                _log_phase(f"lost {target} -> searching")

        # ------------------------------------------------------- searching
        if phase == MOVE_PHASE_SEARCH:
            last_forward_at = 0.0
            _reset_stall()
            gone = now - last_seen_at
            if cfg.search_timeout > 0 and gone >= cfg.search_timeout:
                _finish(loco, execute, "lost",
                        f"{target} not seen for {gone:.0f}s — stopped")
                continue
            if now - search_since < SEARCH_HOLD_S:
                # Stand still first: most losses come back on their own, and
                # turning would carry the camera away from a face that was
                # about to reappear.
                omega, doing = 0.0, "waiting"
                _send_halt()
            else:
                if now - sweep_since >= SEARCH_SWEEP_S:
                    sweep_sign = -sweep_sign
                    sweep_since = now
                # Lead toward the side the face was last on: a positive offset
                # is right of centre, and a negative omega turns right.
                lead = -1.0 if last_offset > 0 else 1.0
                omega = min(cfg.max_omega, SEARCH_OMEGA) * lead * sweep_sign
                doing = "scanning"
                ended, sent = _send_step(0.0, omega)
                if ended:
                    continue
            with lock:
                move_state["omega"] = omega
                move_state["status"] = MOVE_PHASE_SEARCH
                move_state["message"] = (
                    f"lost {target} — {doing} ({gone:.0f}s), still on mission")
            time.sleep(0.05)
            continue

        # ------------------------------------------------------ confirming
        if phase == MOVE_PHASE_CONFIRM:
            last_forward_at = 0.0
            _send_halt()
            confirm_need = (MOVE_RESUME_CONFIRM_S if ever_approached
                            else MOVE_CONFIRM_S)
            # Only time with the face actually in view counts, so a blink
            # pauses the countdown instead of running it down unwatched.
            if match is not None and confirm_tick:
                confirm_seen += min(0.5, now - confirm_tick)
            confirm_tick = now
            left = confirm_need - confirm_seen
            if match is None:
                note = f"{target} slipped out of view — countdown paused"
            else:
                seen = f"{distance:.2f} m" if distance is not None else f"{px}px"
                note = (f"{target} in view ({seen}) — holding still, "
                        f"{max(0.0, left):.1f}s")
            if match is None or left > 0:
                with lock:
                    move_state["omega"] = 0.0
                    move_state["status"] = MOVE_PHASE_CONFIRM
                    move_state["message"] = note
                time.sleep(0.05)
                continue
            phase = MOVE_PHASE_APPROACH
            ever_approached = True
            last_forward_at = 0.0
            _reset_stall()
            if execute and loco is not None:
                try:
                    loco.connect()
                    fsm = loco.fsm_id()
                except Exception:
                    fsm = None
                note = f"confirmed {target} -> walking"
                if fsm is not None:
                    note += f" (fsm={fsm})"
                    if fsm != FSM_WALKING:
                        print(f"  approach WARNING: fsm {fsm}, expected "
                              f"{FSM_WALKING} for sustained walk — hold R1+Y",
                              flush=True)
                _log_phase(note)
            else:
                _log_phase(f"confirmed {target} -> walking")

        # ----------------------------------------------------- approaching
        # Budget counts forward commands only -- blocked, turning-in-place,
        # confirming arrival, and searching must not eat the walk allowance.
        if cfg.timeout > 0 and walked >= cfg.timeout:
            _finish(loco, execute, "stopped",
                    f"{walked:.0f}s of forward motion without reaching {target} "
                    "— stopped")
            continue

        if match is None:
            # Inside MOVE_LOST_GRACE_S: coast on the last heading rather than
            # stutter to a halt over one dropped frame.
            aim = 0.0 if abs(last_offset) <= TURN_DEADBAND else last_offset
            omega = max(-cfg.max_omega, min(cfg.max_omega, -TURN_GAIN * aim))
            if last_cmd is not None:
                step_vx, _ = last_cmd
            else:
                step_vx = max(MIN_MOVE_VX, cfg.vx * SEEK_COAST_VX_RATIO)
                if last_distance is not None and last_distance < SLOW_RADIUS_M:
                    span = max(SLOW_RADIUS_M - cfg.stop_distance_m, 1e-3)
                    ratio = (last_distance - cfg.stop_distance_m) / span
                    step_vx = max(MIN_MOVE_VX,
                                  step_vx * max(0.0, min(1.0, ratio)))
            ended, sent = _send_step(step_vx, omega)
            if ended:
                continue
            if sent:
                _note_forward(step_vx, sent)
                if _check_stall(last_distance, last_px):
                    continue
            with lock:
                move_state["omega"] = omega
                move_state["status"] = MOVE_PHASE_APPROACH
                move_state["message"] = (
                    f"{target} out of view — coasting on last heading")
            time.sleep(0.05)
            continue

        stop_px = cfg.stop_px
        if stop_px <= 0 and fx > 0:
            stop_px = fx * FACE_WIDTH_M / cfg.stop_distance_m

        # Depth, blended distance, or face width — whichever says we are at the
        # standoff first. Several consecutive readings so one speckle does not
        # end the walk after a single step.
        close_enough = (
            (distance is not None and distance <= cfg.stop_distance_m)
            or (stop_px > 0 and px >= stop_px)
        )
        if not close_enough and fx > 0 and px > 0:
            size_only = _size_distance(fx, px)
            if size_only is not None and size_only <= cfg.stop_distance_m:
                close_enough = True
                distance = size_only
        if close_enough:
            reached_stable += 1
            shown = f"{distance:.2f} m" if distance is not None else f"{px}px"
            if reached_stable >= MOVE_REACHED_STABLE_FRAMES:
                _finish(loco, execute, "done", f"reached {target} ({shown})")
                continue
            with lock:
                move_state["status"] = "approaching"
                move_state["message"] = (
                    f"closing on {target} ({shown}) — confirming...")
            time.sleep(0.05)
            continue
        reached_stable = 0

        # NOTE: a wall-clock check lived here -- `elapsed >= cfg.timeout` since
        # the click. It was removed deliberately. It contradicted MoveConfig's
        # own contract for this field ("not wall-clock since the click; blocked,
        # halted and search time do not count"), it sat ABOVE the blocked branch
        # so waiting for an obstacle to clear consumed the allowance, it had no
        # `> 0` guard so a 0 meant "stop instantly" rather than "no limit", and
        # being wall-clock it always fired before the forward-motion budget
        # below -- making that one unreachable. The forward-motion budget is the
        # documented behaviour and is kept, opt-in via --move-timeout.

        if blocked:
            # Stop and wait rather than steer around it. Nothing here plans a
            # path, and a robot picking its own way past an obstacle it can
            # only half see is worse than one standing still.
            _halt(loco, execute)
            with lock:
                move_state["status"] = "blocked"
                move_state["message"] = (
                    f"something {obstacle_dist:.2f} m ahead — waiting")
            time.sleep(0.1)
            continue

        # Yaw toward the face, and walk only once roughly squared up. Turning
        # while badly off-axis just carries the robot past the person.
        aim = 0.0 if abs(offset) <= TURN_DEADBAND else offset
        omega = max(-cfg.max_omega, min(cfg.max_omega, -TURN_GAIN * aim))
        if abs(offset) >= TURN_ONLY_ERR:
            step_vx = 0.0
            phase = "turning to face"
        else:
            step_vx = cfg.vx
            if distance is not None and distance < SLOW_RADIUS_M:
                span = max(SLOW_RADIUS_M - cfg.stop_distance_m, 1e-3)
                ratio = (distance - cfg.stop_distance_m) / span
                step_vx = max(MIN_MOVE_VX, cfg.vx * max(0.0, min(1.0, ratio)))
            step_vx *= 1.0 - 0.5 * min(1.0, abs(offset) / TURN_ONLY_ERR)
            phase = "walking toward" if execute else "would walk toward"

        shown = f"{distance:.2f} m" if distance is not None else f"{px}px"
        with lock:
            move_state["omega"] = omega
            move_state["status"] = "approaching"
            move_state["message"] = (
                f"{phase} {target} — {shown}, stop at "
                f"{cfg.stop_distance_m:.2f} m")

        now = time.perf_counter()
        if now - last_step >= MOVE_STEP_INTERVAL:
            last_step = now
            if execute and loco is not None:
                try:
                    loco.connect()
                    code = loco.step(step_vx, omega)
                    if code:
                        _finish(loco, execute, "stopped",
                                f"SetVelocity returned {code} — is the robot "
                                "standing and in main operation mode?")
                        continue
                except Exception as exc:
                    _finish(loco, execute, "stopped", f"locomotion error: {exc}")
                    continue

        time.sleep(0.05)


@app.route("/login", methods=["GET", "POST"])
def login():
    auth = app.config.get("AUTH")
    if auth is None:
        return render_template_string(NO_ACCOUNT_PAGE), 503
    error = ""
    if request.method == "POST":
        from werkzeug.security import check_password_hash

        user = request.form.get("username", "")
        password = request.form.get("password", "")
        if (secrets.compare_digest(user, auth["username"])
                and check_password_hash(auth["hash"], password)):
            session["user"] = user
            session.permanent = True
            return redirect("/")
        # A deliberate pause. There is no lockout here, and without it the
        # whole password space is guessable at request rate over the LAN.
        time.sleep(1.0)
        error = "Wrong username or password."
    return render_template_string(LOGIN_PAGE, error=error)


@app.route("/logout", methods=["GET", "POST"])
def logout():
    session.clear()
    return redirect("/login")


@app.route("/")
@require_auth
def index():
    # Straight substitution rather than a Jinja variable: PAGE is full of JS
    # template literals and CSS braces, and this keeps the token out of any
    # escaping question.
    return render_template_string(
        PAGE.replace("__FR_TOKEN__", app.config.get("TOKEN", "")))


@app.route("/video_feed")
@require_auth
def video_feed():
    def generate():
        # Register as a viewer BEFORE the first wait. annotate_loop only encodes
        # while this count is above zero, so incrementing late would mean waiting
        # for a frame nothing is producing. The finally runs on client
        # disconnect too, which reaches a generator as GeneratorExit.
        with lock:
            state["viewers"] += 1
        try:
            # Send each frame once, as soon as it exists. The previous version
            # slept a fixed 30 ms per iteration and re-sent whatever was in the
            # buffer, which both capped the stream and added latency on top of
            # the pipeline.
            last_seen = -1
            sent = 0
            while state["running"]:
                with lock:
                    frame, version = state["frame"], state["frame_version"]
                    move_active = move_state["active"]
                if frame is None or version == last_seen:
                    time.sleep(0.005)
                    continue
                last_seen = version
                sent += 1
                # A second deadman source: the MJPEG reader keeps running even
                # when the status poll is throttled in a background tab.
                if move_active and sent % 12 == 0:
                    _note_operator_alive()
                yield (b"--f\r\nContent-Type: image/jpeg\r\n"
                       b"Content-Length: " + str(len(frame)).encode() + b"\r\n\r\n"
                       + frame + b"\r\n")
        finally:
            with lock:
                state["viewers"] = max(0, state["viewers"] - 1)

    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=f")


@app.route("/api/status")
def status():
    # Readable unauthenticated so a spare tab can watch, but only an
    # authenticated request counts as the operator being present. That is what
    # makes the browser's existing 400 ms poll the deadman, with no second
    # request and nothing extra to keep running.
    if _authed():
        _note_operator_alive()
    with lock:
        return jsonify(
            threshold=state["threshold"], fps=state["fps"], faces=state["faces"],
            infer_fps=state["infer_fps"],
            detect_ms=state["detect_ms"], embed_ms=state["embed_ms"],
            enrolled=app.config["ENROLLED"], source=app.config["SOURCE"],
            tuned_threshold=app.config["TUNED_THRESHOLD"],
            impostor_max=app.config["IMPOSTOR_MAX"],
            genuine_min=app.config["GENUINE_MIN"],
            depth_ok=state["depth_ok"],
            move=_public_move_state(),
        )


@app.route("/api/go_to", methods=["POST"])
@require_auth
def go_to():
    body = request.get_json(force=True)
    target = str(body.get("target", "")).strip().lower()
    if target not in app.config["ENROLLED"]:
        return jsonify(ok=False, error=f"unknown person: {target}"), 400
    execute = bool(body.get("execute", False))

    # NOTE: there was an FSM pre-flight here that refused execute=true unless
    # the robot reported FSM 200. It was removed 2026-08-29: the id came from
    # LocoClient.Start() in the SDK source, but this robot walks in FSM 501, so
    # the gate refused a perfectly capable robot and explained it confidently
    # and wrongly. G1Locomotion.fsm_id() is kept for diagnostics; do not gate
    # on a specific id without a list of walking states from the vendor.
    #
    # The concern behind it stands and needs a better answer: SetVelocity acks
    # every command, so a zero return proves nothing about motion. The honest
    # check is the effect -- the target distance should shrink while we are
    # commanding forward travel -- not a mode id.
    now = time.perf_counter()
    loco = app.config.get("LOCO")
    halt_now = False
    with lock:
        if move_state["active"] and move_state["target"] == target:
            # Same person, approach already running -- do not reset timers.
            move_state["last_beat"] = now
            prev = move_state["execute"]
            if prev != execute:
                move_state["execute"] = execute
                if execute:
                    move_state["message"] = (
                        f"walk enabled — continuing toward {target}")
                    print(f"  approach walk enabled mid-mission -> {target}",
                          flush=True)
                else:
                    move_state["message"] = (
                        f"dry-run — robot halted, mission still active "
                        f"({target})")
                    print(f"  approach walk disabled mid-mission -> {target}",
                          flush=True)
                    halt_now = prev
            return jsonify(ok=True, move=_public_move_state())
        move_state["active"] = True
        move_state["target"] = target
        move_state["execute"] = execute
        move_state["status"] = MOVE_PHASE_SEARCH
        move_state["message"] = (
            f"{'moving toward' if execute else 'dry-run toward'} {target}"
        )
        move_state["started_at"] = now
        # Seed the deadman from this request, so the approach starts with a
        # full interval rather than inheriting a stale timestamp.
        move_state["last_beat"] = now
        _reset_move_readings()
    if halt_now:
        _halt(loco, True)
    return jsonify(ok=True, move=_public_move_state())


@app.route("/api/stop_move", methods=["POST"])
def stop_move():
    # No token, on purpose. Anyone who can reach this robot should be able to
    # stop it, including from a phone that never loaded the dashboard. The
    # failure mode of an unauthenticated stop is a halted robot.
    loco = app.config.get("LOCO")
    with lock:
        execute = move_state["execute"]
        move_state["active"] = False
        move_state["status"] = "stopped"
        move_state["message"] = "stopped by user"
        _reset_move_readings()
    # Unconditionally, not only when the run was in execute mode: if a walk was
    # somehow already underway this button has to be the thing that ends it.
    if loco is not None:
        try:
            loco.stop()
        except Exception:
            pass
    return jsonify(ok=True, execute=execute, move=_public_move_state())


@app.route("/api/threshold", methods=["POST"])
@require_auth
def set_threshold():
    value = float(request.get_json(force=True)["threshold"])
    with lock:
        state["threshold"] = max(0.0, min(1.0, value))
    return jsonify(ok=True, threshold=state["threshold"])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--realsense", action="store_true")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=5050)
    ap.add_argument("--range", choices=list(RANGE_MODES), default="mid",
                    help="near=640x480/~4.7m, mid=1280x720/~7.0m, "
                         "far=1920x1080/~10.5m; 30fps each on USB 3 (G1), "
                         "less on USB 2 (Go2)")
    ap.add_argument("--token", default="",
                    help="shared token for the control endpoints; generated "
                         "and printed at startup if not given")
    ap.add_argument("--set-login", action="store_true",
                    help="create or change the dashboard login account, then "
                         f"exit. Stored in {AUTH_PATH}, mode 600.")
    ap.add_argument("--move-iface", default="eth0",
                    help="network interface for G1 LocoClient (eth0 on G1)")
    ap.add_argument("--stop-distance", type=float, default=DEFAULT_STOP_DISTANCE_M,
                    help="standoff in metres, camera to face "
                         f"(minimum {STOP_DISTANCE_FLOOR_M})")
    ap.add_argument("--stop-px", type=float, default=0.0,
                    help="override the face-width stop gate; 0 derives it from "
                         "--stop-distance and the camera's focal length")
    ap.add_argument("--move-vx", type=float, default=DEFAULT_MOVE_VX,
                    help="forward speed (m/s) during go-to-person")
    ap.add_argument("--move-omega", type=float, default=DEFAULT_MAX_OMEGA,
                    help="maximum yaw rate (rad/s) while aiming at the target")
    ap.add_argument("--obstacle-distance", type=float, default=DEFAULT_OBSTACLE_M,
                    help="stop and wait if the forward corridor is closer than "
                         "this (metres); needs depth")
    ap.add_argument("--move-timeout", type=float, default=DEFAULT_MOVE_TIMEOUT,
                    help="optional cap on seconds spent commanding forward "
                         "motion toward the target; blocked, halted and search "
                         "time do not count. Default %(default)s = no cap: "
                         "follow until reached, stopped, stalled or lost")
    ap.add_argument("--search-timeout", type=float,
                    default=DEFAULT_SEARCH_TIMEOUT,
                    help="give up a mission after this many seconds with no "
                         "sight of the target at all (default %(default)s)")
    ap.add_argument("--heartbeat-timeout", type=float,
                    default=MOVE_HEARTBEAT_TIMEOUT_S,
                    help="stop approach if the dashboard goes quiet for this "
                         "many seconds (after an initial grace period)")
    ap.add_argument("--step-interval", type=float, default=MOVE_STEP_INTERVAL,
                    help="seconds between velocity commands (default "
                         "%(default)s)")
    ap.add_argument("--stream-width", type=int, default=STREAM_MAX_WIDTH,
                    help="downscale the browser stream to this width before "
                         "encoding; processing stays at full capture "
                         "resolution. 0 sends the full frame (default "
                         "%(default)s)")
    ap.add_argument("--chunk-duration", type=float, default=None,
                    help="how long each velocity command stays valid; must "
                         "exceed --step-interval (default: step-interval + "
                         "0.15)")
    args = ap.parse_args()

    if args.stop_distance < STOP_DISTANCE_FLOOR_M:
        ap.error(f"--stop-distance below the {STOP_DISTANCE_FLOOR_M} m floor. "
                 "That is inside arm's reach with no margin for depth noise.")

    if args.chunk_duration is None:
        args.chunk_duration = args.step_interval + 0.15
    if args.chunk_duration <= args.step_interval:
        ap.error(f"--chunk-duration ({args.chunk_duration}) must exceed "
                 f"--step-interval ({args.step_interval}), or each command "
                 "expires before its replacement arrives.")

    global CAPTURE_WIDTH, CAPTURE_HEIGHT, CAPTURE_FPS
    CAPTURE_WIDTH, CAPTURE_HEIGHT, CAPTURE_FPS, reach = RANGE_MODES[args.range]

    if args.set_login:
        # Before any camera or model work: this path only touches a file.
        username = input("username: ").strip()
        if not username:
            print("username cannot be empty")
            return 1
        password = getpass.getpass("password: ")
        if len(password) < 6:
            print("password must be at least 6 characters")
            return 1
        if password != getpass.getpass("repeat password: "):
            print("passwords did not match")
            return 1
        save_auth(username, password)
        print(f"saved to {AUTH_PATH} (mode 600)")
        print("sign in at http://<robot-ip>:5050/login")
        return 0

    token = args.token or secrets.token_urlsafe(9)
    app.config["TOKEN"] = token
    app.config["HEARTBEAT_TIMEOUT"] = args.heartbeat_timeout
    app.config["STREAM_MAX_WIDTH"] = max(0, args.stream_width)

    auth = load_auth()
    app.config["AUTH"] = auth
    # A signing key must exist either way, because _authed() reads the session
    # on every request. When there is no account the key is throwaway -- nothing
    # issues a cookie, and the token is the only way in.
    app.secret_key = auth["secret"] if auth else secrets.token_hex(32)
    app.permanent_session_lifetime = timedelta(hours=12)

    configure_runtime_threads()
    db = IdentityDB.load()
    pipeline = FacePipeline()
    # Name the provider that actually loaded. onnxruntime falls back to CPU
    # without raising, so "I installed the GPU build" is not evidence it is in use.
    print(f"Inference: {pipeline.providers[0]} "
          f"(det threads {threads_label(DET_THREADS)}, "
          f"embed {threads_label(EMB_THREADS)})", flush=True)
    state["threshold"] = load_threshold()
    source = "realsense" if args.realsense else args.camera
    app.config["ENROLLED"] = db.names
    app.config["SOURCE"] = "RealSense D435i" if args.realsense else f"camera {args.camera}"

    # Surface the tuned value and the measurements behind it, so the slider
    # cannot silently be left somewhere unsafe.
    app.config["TUNED_THRESHOLD"] = state["threshold"]
    metrics = {}
    if CONFIG_PATH.exists():
        metrics = json.loads(CONFIG_PATH.read_text()).get("metrics", {})
    app.config["IMPOSTOR_MAX"] = metrics.get("impostor_max")
    app.config["GENUINE_MIN"] = metrics.get("genuine_min")

    loco = G1Locomotion(iface=args.move_iface)
    app.config["LOCO"] = loco
    move_cfg = MoveConfig(
        stop_distance_m=args.stop_distance,
        stop_px=args.stop_px,
        vx=args.move_vx,
        max_omega=args.move_omega,
        obstacle_m=args.obstacle_distance,
        timeout=args.move_timeout,
        search_timeout=args.search_timeout,
        step_interval=args.step_interval,
        chunk_s=args.chunk_duration,
    )

    # Four independent loops: camera, inference, annotation, and go-to-person.
    # The capture thread is kept hold of: it owns the camera, and shutdown has
    # to wait for it rather than let the interpreter pull the rug out.
    grab_thread = threading.Thread(target=grab_loop, args=(source,), daemon=True)
    grab_thread.start()
    for target, targs in (
        (infer_loop, (pipeline, db)),
        (annotate_loop, ()),
        (move_loop, (loco, move_cfg)),
    ):
        threading.Thread(target=target, args=targs, daemon=True).start()

    print(f"Enrolled: {', '.join(db.names)}   threshold {state['threshold']:.2f}")
    print(f"Range '{args.range}': {CAPTURE_WIDTH}x{CAPTURE_HEIGHT}@{CAPTURE_FPS}fps, "
          f"recognises to ~{reach:.1f} m")
    print(f"min face {MIN_FACE_WIDTH_PX}px  hold ratio {HOLD_RATIO}  "
          f"smoothing {SMOOTH_FRAMES} frames")
    print(f"Go-to: stop {args.stop_distance:.2f} m from the face, "
          f"vx<={args.move_vx} m/s, omega<={args.move_omega} rad/s, "
          f"obstacle stop {args.obstacle_distance:.2f} m, iface={args.move_iface}")
    print(f"       one click per mission: {MOVE_CONFIRM_S:.0f}s first confirm, "
          f"{MOVE_RESUME_CONFIRM_S:.0f}s resume; losing the face searches")
    print("       budgets: %s, %.0fs without sight of the target"
          % ("no forward-motion cap (follows until reached)"
             if args.move_timeout <= 0 else
             "%.0fs of forward motion" % args.move_timeout,
             args.search_timeout))
    print("       dry-run until 'Enable walk' is ticked on the dashboard")
    print(f"       deadman: {MOVE_HEARTBEAT_GRACE_S:.0f}s grace, then "
          f"{args.heartbeat_timeout:.0f}s without dashboard contact")
    shown_host = "127.0.0.1" if args.host == "0.0.0.0" else args.host
    if auth:
        print(f"Dashboard: http://{shown_host}:{args.port}/  "
              f"-- sign in as '{auth['username']}'")
        print(f"           token URL still works: /?token={token}")
    else:
        print(f"Dashboard: http://{shown_host}:{args.port}/?token={token}")
        print("           no login account set -- run ./start.sh --set-login "
              "once to stop needing the token in the URL")
    if args.host == "0.0.0.0":
        print("Bound to all interfaces -- reachable from other devices on this "
              "network.")
    try:
        app.run(host=args.host, port=args.port, threaded=True, debug=False)
    except KeyboardInterrupt:
        print("\nstopping...", flush=True)
    finally:
        with lock:
            state["running"] = False
        # Wait for the capture thread to close the camera itself. Without this
        # the interpreter exits while wait_for_frames is still in flight, and
        # librealsense's destructors abort the process:
        #   terminate called without an active exception / Aborted (core dumped)
        # The camera could then still be claimed when the next run starts.
        grab_thread.join(timeout=FRAME_WAIT_MS / 1000.0 + 3.0)
        if grab_thread.is_alive():
            print("  camera thread did not exit -- the device may need a "
                  "replug before the next run", flush=True)
        loco.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
