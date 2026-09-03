# Face Recognition — Unitree G1

Live face recognition and **go-to-person** walking for the Unitree G1 humanoid,
using an Intel RealSense D435i and CPU-only ONNX models (no training required).

Recognises enrolled colleagues by name, labels everyone else as `unknown`, and
can walk toward a named person from a browser dashboard.

---

## Features

- **Real-time recognition** — SCRFD-500m detector + ArcFace MobileFaceNet
  (InsightFace buffalo_s), 512-d embeddings, pose sub-centroids (k=10)
- **Live dashboard** — browser video feed, threshold slider, performance stats
- **Go-to-person (G1)** — one click to confirm, walk, search-if-lost, stop at
  standoff (default **55 cm**); dry-run until "Enable walk" is checked
- **Measured security** — threshold tuned against 1,360 demographically matched
  FairFace impostors (not guessed from docs)
- **Temporal tracking** — identity held on tracks, not per-frame flicker
- **Remote deploy** — sync gallery from laptop to robot over SSH

---

## Hardware

| Component | Requirement |
|---|---|
| Robot | Unitree **G1** (primary); Go2 supported as legacy |
| Camera | Intel **RealSense D435/D435i** (auto-probed; use `--realsense`) |
| Network | dashboard over the robot's wifi; locomotion via **`eth0`** (Unitree SDK) |
| Compute | CPU only — ONNX Runtime, no GPU required |

---

## Quick start (G1)

```bash
cd ~/face_recognition_ws
./install.sh          # once: venv, Python packages, model files
./verify.sh --local   # optional: pre-flight without starting
./start.sh --local --realsense --range mid
```

`start.sh` prints URLs like:

```text
http://<robot-ip>:5050/?token=<token>
```

Open that URL in a browser. The token is required for video, recognition, and
go-to. **Stop movement** works without the token.

Optional: create a login account (survives restarts, stored outside the repo):

```bash
./start.sh --set-login
# then sign in at http://<robot-ip>:5050/login
```

Stop:

```bash
./stop.sh --local
```

---

## Dashboard — go-to-person

1. Stand in view until your name appears under **Detected now**
2. Click **Go to &lt;name&gt;** (dry-run first — robot shows status but does not move)
3. Check **Enable walk**, click the same person again
4. Robot confirms identity (~5 s), yaws, walks forward, stops at standoff
5. **Stop movement** anytime

Defaults (in code, not CLI):

- Stop distance: **0.55 m** from face
- Forward speed: 0.12 m/s
- Robot must be standing in walking mode (remote **R1+Y** → FSM 501)

---

## Deploy from a laptop

Sync the gallery and start on the G1 over wifi:

```bash
./start.sh --g1
```

Push code as well as the gallery:

```bash
./start.sh --g1 --sync
```

Legacy Go2 modes: `./start.sh --wifi` or `./start.sh --ethernet`.

Remote modes need the robot's address and SSH password, which are kept out of
this repository. Copy `.fr_robots.env.example` to `~/.fr_robots.env` and fill
it in, or export `FR_HOST_G1` / `FR_USER` / `FR_PASSWORD`. Running on the robot
itself with `--local` needs none of it.

---

## Enrolment (add or update people)

Run on the machine where enrolment photos live (usually the dev laptop):

```bash
# Put photos in dataset/enroll/<name>/ then:
./venv/bin/python scripts/enroll.py --report
./venv/bin/python scripts/tune_threshold.py --save
./venv/bin/python scripts/evaluate.py
./venv/bin/python scripts/verify_deploy.py --baseline
```

Push the updated gallery to the G1:

```bash
./start.sh --g1
```

**Photo guidelines (match the G1 camera):** best is 15–20 frames captured
while the person stands **1.5–4 m** in front of the running dashboard
(straight-on plus slight left/right turns, include 2–4 m if you use go-to).
Phone photos are OK if eye-level, single face, same range — not chin-up
selfies or group shots. Run `enroll.py --report` and fix any `!!` warnings.

---

## Project layout

```text
face_recognition_ws/
├── start.sh stop.sh verify.sh install.sh   entry points
├── backup.sh recover.sh                    disaster recovery
├── lib.sh                                  shared shell helpers
├── db/embeddings.npz                       face gallery (committed)
├── db/config.json                          tuned threshold + metrics
├── models/candidates/buffalo_s/            SCRFD + ArcFace ONNX weights
└── scripts/
    ├── dashboard.py                        web UI + go-to FSM
    ├── face_pipeline.py                    detect, embed, match, track
    ├── g1_locomotion.py                    G1 LocoClient wrapper
    ├── enroll.py recognize.py              gallery build + CLI demo
    └── verify_deploy.py                    confirm robot matches dev machine
```

Deep design notes, measured numbers, and traps: see **`context.md`**.

---

## Configuration

| Setting | Default (G1) | Flag |
|---|---|---|
| Resolution | 1280×720 (`mid`) | `--range near\|mid\|far` |
| Port | 5050 | `--port N` |
| Camera | RealSense auto-probe | `--realsense` or `--webcam N` |
| Threshold | 0.45 (from `db/config.json`) | slider in dashboard |
| Inference threads | 4 | `FR_THREADS` env var |

Go-to tuning flags exist (`--stop-distance`, `--move-vx`, etc.) but normal
use needs none of them — defaults are baked into `scripts/dashboard.py`.

---

## Recovery

If source files are accidentally deleted:

```bash
./recover.sh              # from latest .fr_backup_latest
git checkout -- .         # if git history exists
```

Snapshot after changes:

```bash
./backup.sh
```

---

## Safety

- **Experimental locomotion** — test in open space, keep **Stop movement** ready
- **Dry-run first** — confirm the correct person is targeted before enabling walk
- **No anti-spoofing** — a printed photo can fool the recogniser today
- **Do not lower the threshold** below the tuned 0.45 without remeasuring FMR
- **Deadman** — if the browser tab closes, the robot stops the approach within ~8 s

---

## Development

```bash
./verify.sh --local              # fast checks
./verify.sh --local --full       # includes accuracy evaluation (~2 min)
./venv/bin/python scripts/verify_deploy.py
./venv/bin/python scripts/selftest.py
python3 -m py_compile scripts/*.py
```

Compare pipelines:

```bash
./venv/bin/python scripts/compare_models.py
```

---

## Models & data

| Item | Source | Licence |
|---|---|---|
| SCRFD-500m + ArcFace-mbf | InsightFace buffalo_s | check InsightFace model zoo terms |
| FairFace impostors | `scripts/fetch_impostors.py` | CC BY 4.0 (attribution required) |
| Enrolled embeddings | computed locally | not derived from public weights |

Large binary assets (`models/*.onnx`, `db/embeddings.npz`) may be in git;
raw enrolment photos under `dataset/` are typically local only.

---

## Licence

Application code: proprietary / internal use.

Third-party models and FairFace impostor data carry their own licences — see
`dataset/impostors/ATTRIBUTION.txt` when present and InsightFace model docs.
