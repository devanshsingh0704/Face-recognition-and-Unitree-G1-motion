"""Run recognition on a still image, a folder, or a live camera.

    python scripts/recognize.py --image photo.jpg
    python scripts/recognize.py --folder dataset/test/ramesh
    python scripts/recognize.py --camera 0          # laptop webcam
    python scripts/recognize.py --realsense         # D435i colour stream

Press q to quit a live window. --headless writes annotated frames to output/
instead of opening a window, which is how this runs on the Go2.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from face_pipeline import WS, FacePipeline, IdentityDB, load_threshold, open_camera

OUTPUT_DIR = WS / "output"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

KNOWN_COLOUR = (0, 200, 0)
UNKNOWN_COLOUR = (0, 140, 255)


def annotate(frame, face, label: str, score: float, threshold: float) -> None:
    x, y, w, h = face.box
    known = label != "unknown"
    colour = KNOWN_COLOUR if known else UNKNOWN_COLOUR
    cv2.rectangle(frame, (x, y), (x + w, y + h), colour, 2)

    text = f"{label} {score:.2f}" if known else f"unknown ({score:.2f})"
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(frame, (x, y - th - 8), (x + tw + 6, y), colour, -1)
    cv2.putText(frame, text, (x + 3, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)


def process(frame, pipeline: FacePipeline, db: IdentityDB, threshold: float):
    results = []
    for face in pipeline.detect(frame):
        emb = pipeline.embed(frame, face)
        label, score = db.identify(emb, threshold)
        annotate(frame, face, label, score, threshold)
        results.append((label, score, face.width))
    return results


def run_still(paths: list[Path], pipeline, db, threshold: float) -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in paths:
        frame = cv2.imread(str(path))
        if frame is None:
            print(f"{path.name}: unreadable")
            continue
        results = process(frame, pipeline, db, threshold)
        out = OUTPUT_DIR / f"annotated_{path.name}"
        cv2.imwrite(str(out), frame)
        if not results:
            print(f"{path.name}: no face detected")
        for label, score, width in results:
            print(f"{path.name}: {label:<12} score {score:.3f}  face {width}px")
    print(f"\nAnnotated images written to {OUTPUT_DIR}")
    return 0


def open_realsense():
    """Colour stream from a D435i. Returns (read_fn, close_fn)."""
    try:
        import pyrealsense2 as rs
    except ImportError:
        print("pyrealsense2 not installed:\n"
              "  venv/bin/pip install pyrealsense2", file=sys.stderr)
        return None, None

    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pipe.start(cfg)

    def read():
        frames = pipe.wait_for_frames(timeout_ms=5000)
        colour = frames.get_color_frame()
        if not colour:
            return False, None
        return True, np.asanyarray(colour.get_data())

    return read, pipe.stop


def run_live(source, pipeline, db, threshold: float, headless: bool) -> int:
    if source == "realsense":
        read, close = open_realsense()
        if read is None:
            return 1
    else:
        try:
            cap = open_camera(int(source))
        except RuntimeError as exc:
            print(exc, file=sys.stderr)
            return 1
        read, close = (lambda: cap.read()), cap.release

    if headless:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Running. Press q to quit." if not headless else "Running headless.")

    frames = 0
    fps = 0.0
    t0 = time.perf_counter()
    try:
        while True:
            ok, frame = read()
            if not ok:
                print("Frame grab failed -- camera disconnected?", file=sys.stderr)
                break

            process(frame, pipeline, db, threshold)
            frames += 1
            elapsed = time.perf_counter() - t0
            if elapsed >= 1.0:
                fps, frames, t0 = frames / elapsed, 0, time.perf_counter()

            cv2.putText(frame, f"{fps:.1f} FPS  thr {threshold:.2f}", (8, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            if headless:
                cv2.imwrite(str(OUTPUT_DIR / "live.jpg"), frame)
            else:
                cv2.imshow("face recognition", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    except KeyboardInterrupt:
        pass
    finally:
        close()
        if not headless:
            cv2.destroyAllWindows()
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--image", type=Path)
    src.add_argument("--folder", type=Path)
    src.add_argument("--camera", type=int)
    src.add_argument("--realsense", action="store_true")
    ap.add_argument("--threshold", type=float, default=None)
    ap.add_argument("--headless", action="store_true")
    args = ap.parse_args()

    db = IdentityDB.load()
    pipeline = FacePipeline()
    threshold = args.threshold if args.threshold is not None else load_threshold()
    print(f"Enrolled: {', '.join(db.names)}   threshold {threshold:.3f}")

    if args.image:
        return run_still([args.image], pipeline, db, threshold)
    if args.folder:
        paths = sorted(p for p in args.folder.rglob("*")
                       if p.suffix.lower() in IMAGE_SUFFIXES)
        if not paths:
            print(f"No images under {args.folder}", file=sys.stderr)
            return 1
        return run_still(paths, pipeline, db, threshold)
    return run_live("realsense" if args.realsense else args.camera,
                    pipeline, db, threshold, args.headless)


if __name__ == "__main__":
    raise SystemExit(main())
