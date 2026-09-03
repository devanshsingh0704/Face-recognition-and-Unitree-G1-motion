"""Measure real throughput, including a constrained run that approximates the Go2.

This laptop has 12 threads. The Go2's onboard compute has far less, so an
unconstrained benchmark here tells you nothing about whether the pipeline will
hold real time on the robot. Restricting OpenCV to a small thread count gives a
pessimistic-but-honest lower bound before you deploy.

    python scripts/benchmark.py
    python scripts/benchmark.py --threads 4 --frames 100
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from face_pipeline import WS, FacePipeline, IdentityDB, load_threshold


def find_sample_image() -> Path | None:
    for folder in ("dataset/test", "dataset/enroll", "dataset/impostors"):
        for suffix in ("*.jpg", "*.jpeg", "*.png"):
            hits = sorted((WS / folder).rglob(suffix))
            if hits:
                return hits[0]
    return None


def time_stage(fn, frames: int) -> tuple[float, float]:
    """Return (mean_ms, p95_ms) over `frames` calls."""
    times = []
    for _ in range(frames):
        t = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t) * 1000)
    arr = np.array(times)
    return float(arr.mean()), float(np.percentile(arr, 95))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--threads", type=int, nargs="*", default=[0, 4, 2, 1],
                    help="thread counts to test; 0 means unrestricted")
    ap.add_argument("--frames", type=int, default=50)
    ap.add_argument("--image", type=Path, default=None)
    ap.add_argument("--resolution", type=int, nargs=2, default=[640, 480],
                    metavar=("W", "H"))
    args = ap.parse_args()

    sample_path = args.image or find_sample_image()
    if sample_path is None:
        print("No sample image found. Add photos under dataset/ first.", file=sys.stderr)
        return 1

    image = cv2.imread(str(sample_path))
    if image is None:
        print(f"Cannot read {sample_path}", file=sys.stderr)
        return 1
    w, h = args.resolution
    frame = cv2.resize(image, (w, h))
    print(f"Sample: {sample_path.name} -> {w}x{h}, {args.frames} iterations\n")

    pipeline = FacePipeline()
    faces = pipeline.detect(frame)
    if not faces:
        print(f"No face detected in {sample_path.name}; timing detection only.\n")
    face = faces[0] if faces else None

    try:
        db = IdentityDB.load()
        threshold = load_threshold()
    except FileNotFoundError:
        db, threshold = None, 0.0

    print(f"{'threads':>8} {'detect':>16} {'embed':>16} {'total':>16} {'FPS':>7}")
    print(f"{'':>8} {'mean/p95 ms':>16} {'mean/p95 ms':>16} {'mean ms':>16}")
    print("-" * 68)

    for n in args.threads:
        cv2.setNumThreads(n if n > 0 else -1)
        label = "all" if n == 0 else str(n)

        det_mean, det_p95 = time_stage(lambda: pipeline.detect(frame), args.frames)
        if face is not None:
            emb_mean, emb_p95 = time_stage(
                lambda: pipeline.embed(frame, face), args.frames
            )
        else:
            emb_mean = emb_p95 = 0.0

        total = det_mean + emb_mean
        fps = 1000.0 / total if total > 0 else 0.0
        print(f"{label:>8} {det_mean:7.1f}/{det_p95:<8.1f} "
              f"{emb_mean:7.1f}/{emb_p95:<8.1f} {total:15.1f} {fps:7.1f}")

    cv2.setNumThreads(-1)

    if db is not None:
        emb = pipeline.embed(frame, face) if face is not None else None
        if emb is not None:
            m_mean, _ = time_stage(lambda: db.identify(emb, threshold), 1000)
            print(f"\nmatching against {len(db.names)} enrolled: {m_mean:.4f} ms "
                  "(negligible -- it is one matrix multiply)")

    print("\nNote: 'embed' is per face. A frame with 3 people costs 3x that stage.")
    print("The low thread counts approximate the Go2; treat them as the real budget.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
