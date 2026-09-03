"""Check which camera nodes actually deliver an image.

A camera that opens successfully can still hand back pure black -- a privacy
shutter or a firmware mute looks exactly like a working device until you
inspect the pixels. This grabs real frames and reports what is in them.

    python scripts/check_camera.py
"""

from __future__ import annotations

import sys
import time

import cv2

# The HP HD Camera in this laptop emits pure black for ~8 seconds after open.
# A short warm-up reads that as a dead device, so wait properly before judging.
WARMUP_TIMEOUT = 25.0


def probe(index: int) -> tuple[str, bool]:
    cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
    if not cap.isOpened():
        return "cannot open (no such node, or in use)", False

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    start = time.time()
    frame = None
    try:
        while time.time() - start < WARMUP_TIMEOUT:
            ok, f = cap.read()
            if ok and f is not None:
                frame = f
                if f.std() > 3.0:
                    return (f"{w}x{h} OK (ready in {time.time() - start:.1f}s, "
                            f"mean {f.mean():.1f}, std {f.std():.1f})"), True
            time.sleep(0.05)
    finally:
        cap.release()

    if frame is None:
        return f"{w}x{h} opens but delivers no frames", False
    return (f"{w}x{h} still blank after {WARMUP_TIMEOUT:.0f}s "
            f"(mean {frame.mean():.2f}, std {frame.std():.2f})"), False


def main() -> int:
    print("Probing video nodes -- takes a moment per node.\n")
    usable = []
    for index in range(4):
        status, ok = probe(index)
        print(f"  /dev/video{index}: {status}")
        if ok:
            usable.append(index)

    if usable:
        print(f"\nUsable: {', '.join(f'/dev/video{i}' for i in usable)}")
        print(f"Start the dashboard with:  "
              f"./venv/bin/python scripts/dashboard.py --camera {usable[0]}")
        return 0

    print("\nNo usable camera.")
    print("  - check the physical privacy shutter beside the webcam lens")
    print("  - check the camera-mute key on the keyboard (F-key with a camera icon)")
    print("  - close any other app holding the camera")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
