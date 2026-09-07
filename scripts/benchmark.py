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
from face_pipeline import (
    DEFAULT_INFERENCE_THREADS,
    DET_THREADS,
    EMB_THREADS,
    WS,
    FacePipeline,
    IdentityDB,
    load_threshold,
    threads_label,
)


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


def draw_overlay(frame: np.ndarray, faces: list) -> np.ndarray:
    """Mirror what dashboard.annotate_loop draws, so the cost is comparable.

    Kept as a copy rather than an import because importing dashboard pulls in
    Flask and builds its module-level app and state.
    """
    canvas = frame.copy()
    mid = canvas.shape[1] // 2
    cv2.line(canvas, (mid, 0), (mid, canvas.shape[0]), (120, 120, 120), 1)
    for f in faces:
        x, y, w, h = f.box
        cv2.rectangle(canvas, (x, y), (x + w, y + h), (0, 200, 0), 2)
        text = f"name {f.score:.2f}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(canvas, (x, y - th - 9), (x + tw + 8, y), (0, 200, 0), -1)
        cv2.putText(canvas, text, (x + 4, y - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return canvas


def downscale(canvas: np.ndarray, max_w: int) -> np.ndarray:
    if not max_w or canvas.shape[1] <= max_w:
        return canvas
    ratio = max_w / canvas.shape[1]
    return cv2.resize(canvas, (max_w, max(1, int(round(canvas.shape[0] * ratio)))),
                      interpolation=cv2.INTER_AREA)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--threads", type=int, nargs="*", default=[0, 4, 2, 1],
                    help="thread counts to test; 0 means unrestricted")
    ap.add_argument("--frames", type=int, default=50)
    ap.add_argument("--image", type=Path, default=None)
    ap.add_argument("--resolution", type=int, nargs=2, default=[640, 480],
                    metavar=("W", "H"))
    ap.add_argument("--jpeg-quality", type=int, default=80,
                    help="mirrors dashboard.JPEG_QUALITY (default %(default)s)")
    ap.add_argument("--stream-width", type=int, default=960,
                    help="mirrors dashboard.STREAM_MAX_WIDTH; 0 disables "
                         "downscaling (default %(default)s)")
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
    print(f"Inference provider: {pipeline.providers[0]}  "
          f"(det threads {threads_label(DET_THREADS)}, "
          f"embed {threads_label(EMB_THREADS)})\n")
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

    # --- frame delivery -----------------------------------------------------
    # The cost of getting a picture to the browser, which the table above never
    # measured. This is the work the dashboard now skips when no viewer is
    # attached, so these numbers are what that saves per frame.
    cv2.setNumThreads(DEFAULT_INFERENCE_THREADS)
    canvas = draw_overlay(frame, faces)
    small = downscale(canvas, args.stream_width)
    jpeg_args = [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality]

    ann_mean, ann_p95 = time_stage(lambda: draw_overlay(frame, faces), args.frames)
    enc_mean, enc_p95 = time_stage(
        lambda: cv2.imencode(".jpg", canvas, jpeg_args), args.frames)
    scaled_mean, scaled_p95 = time_stage(
        lambda: cv2.imencode(".jpg", downscale(canvas, args.stream_width), jpeg_args),
        args.frames)

    full_kb = len(cv2.imencode(".jpg", canvas, jpeg_args)[1]) / 1024
    small_kb = len(cv2.imencode(".jpg", small, jpeg_args)[1]) / 1024

    print(f"\nframe delivery at {w}x{h}, {len(faces)} face(s), "
          f"quality {args.jpeg_quality}, {DEFAULT_INFERENCE_THREADS} threads")
    print(f"{'stage':>26} {'mean/p95 ms':>16}")
    print("-" * 44)
    print(f"{'annotate (draw boxes)':>26} {ann_mean:7.2f}/{ann_p95:<8.2f}")
    print(f"{'encode full frame':>26} {enc_mean:7.2f}/{enc_p95:<8.2f}")
    if args.stream_width and canvas.shape[1] > args.stream_width:
        print(f"{f'encode at {args.stream_width}px wide':>26} "
              f"{scaled_mean:7.2f}/{scaled_p95:<8.2f}")
        print(f"\n  per delivered frame: {ann_mean + enc_mean:.2f} ms full, "
              f"{ann_mean + scaled_mean:.2f} ms downscaled")
        print(f"  bandwidth per frame: {full_kb:.1f} KB full, {small_kb:.1f} KB "
              f"downscaled ({100 * small_kb / full_kb:.0f}%)")
    else:
        print(f"\n  per delivered frame: {ann_mean + enc_mean:.2f} ms "
              f"({full_kb:.1f} KB)")
        print(f"  no downscale applied: frame is {canvas.shape[1]}px wide, "
              f"--stream-width is {args.stream_width}")
    print("  all of it skipped while no browser has /video_feed open.")
    cv2.setNumThreads(-1)

    print("\nNote: 'embed' is per face. A frame with 3 people costs 3x that stage.")
    print("The low thread counts approximate the Go2; treat them as the real budget.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
