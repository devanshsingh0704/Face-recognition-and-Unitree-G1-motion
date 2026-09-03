"""Confirm the deployed pipeline behaves identically to the development machine.

Run this on the robot after copying the workspace. It checks that the models
load, that recognition produces the same scores the laptop produced, and that
throughput is good enough to be useful.

    ./venv/bin/python scripts/verify_deploy.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from face_pipeline import FacePipeline, IdentityDB, load_threshold

# Expected scores are generated on the development machine and shipped with the
# gallery, rather than hardcoded here -- hardcoded numbers go stale the moment
# anyone is enrolled or re-enrolled, and then report a deployment failure that
# is really just an out-of-date constant.
#
#   laptop:  python scripts/verify_deploy.py --baseline    (writes db/baseline.json)
#   robot :  python scripts/verify_deploy.py               (compares against it)
BASELINE = Path(__file__).resolve().parent.parent / "db" / "baseline.json"
TOLERANCE = 0.02


def sample_photos(ws: Path) -> list[Path]:
    """One test photo per enrolled person, for a quick cross-machine check."""
    out = []
    for person in sorted((ws / "dataset" / "test").glob("*")):
        if not person.is_dir():
            continue
        photos = sorted(person.glob("*.jpg"))
        if photos:
            out.append(photos[0])
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--baseline", action="store_true",
                    help="write db/baseline.json from this machine (run on the laptop)")
    args = ap.parse_args()

    ws = Path(__file__).resolve().parent.parent
    pipeline = FacePipeline()
    db = IdentityDB.load()
    threshold = load_threshold()

    if args.baseline:
        out = {}
        for path in sample_photos(ws):
            img = cv2.imread(str(path))
            face = pipeline.largest_face(img)
            if face is None:
                continue
            out[path.name] = db.scores(pipeline.embed(img, face))
        BASELINE.write_text(json.dumps(out, indent=2))
        print(f"wrote {BASELINE} with {len(out)} reference photos:")
        for k, v in out.items():
            print(f"  {k}: " + "  ".join(f"{n} {s:.4f}" for n, s in v.items()))
        return 0

    if not BASELINE.exists():
        print(f"No baseline at {BASELINE}\n"
              "Generate it on the development machine with:\n"
              "  ./venv/bin/python scripts/verify_deploy.py --baseline\n"
              "then copy db/baseline.json across.", file=sys.stderr)
        return 1
    expected_all = json.loads(BASELINE.read_text())

    print(f"backend      : {pipeline.backend} ({pipeline.embedding_dim}-d)")
    print(f"enrolled     : {', '.join(db.names)}")
    print(f"threshold    : {threshold}")
    print(f"sub-centroids: " + ", ".join(f"{n}:{len(db.subcentroids[n])}" for n in db.names))
    print()

    ok = True
    for name, expect in expected_all.items():
        matches = list(ws.glob(f"dataset/test/*/{name}")) + [ws / name]
        path = next((m for m in matches if m.exists()), None)
        if path is None:
            print(f"  skip {name} (not present on this machine)")
            continue
        img = cv2.imread(str(path))
        face = pipeline.largest_face(img)
        if face is None:
            print(f"  FAIL {name}: no face detected")
            ok = False
            continue
        scores = db.scores(pipeline.embed(img, face))
        deltas = {k: scores[k] - v for k, v in expect.items()}
        worst = max(abs(d) for d in deltas.values())
        status = "ok  " if worst <= TOLERANCE else "FAIL"
        ok &= worst <= TOLERANCE
        detail = "  ".join(f"{k} {scores[k]:.4f} (exp {expect[k]:.4f})" for k in expect)
        print(f"  {status} {name:<22} {face.width}px  {detail}   max delta {worst:.4f}")

    print()
    frame = np.zeros((480, 640, 3), np.uint8)
    samples = sample_photos(ws)
    img = cv2.imread(str(samples[0])) if samples else None
    if img is not None:
        frame = cv2.resize(img, (640, 480))
    det, emb = [], []
    face = pipeline.largest_face(frame)
    for _ in range(15):
        t = time.perf_counter()
        pipeline.detect(frame, min_width=28)
        det.append((time.perf_counter() - t) * 1000)
        if face is not None:
            t = time.perf_counter()
            pipeline.embed(frame, face)
            emb.append((time.perf_counter() - t) * 1000)
    d, e = float(np.mean(det)), float(np.mean(emb or [0]))
    print(f"speed @640x480: detect {d:.1f} ms   embed {e:.1f} ms/face   "
          f"=> {1000 / max(d + e, 1e-3):.1f} fps single face")

    print("\n" + ("DEPLOYMENT VERIFIED — matches the development machine"
                  if ok else "MISMATCH — scores differ from the laptop, investigate"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
