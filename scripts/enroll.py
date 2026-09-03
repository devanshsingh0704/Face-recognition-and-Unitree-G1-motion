"""Build the identity database from dataset/enroll/<person>/*.jpg

Each subfolder name becomes the label the robot reports. Photos that fail
quality checks are skipped and reported rather than silently included -- a
blurry or tiny face drags the person's centroid toward nothing in particular
and quietly degrades every future match.

    python scripts/enroll.py
    python scripts/enroll.py --min-face 60 --report
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from face_pipeline import WS, FacePipeline, IdentityDB, MIN_FACE_WIDTH_PX

ENROLL_DIR = WS / "dataset" / "enroll"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Below this variance-of-Laplacian a crop is too soft to trust.
BLUR_THRESHOLD = 45.0


def blur_score(face_crop: np.ndarray) -> float:
    gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def enroll_person(
    pipeline: FacePipeline, folder: Path, min_face: int, verbose: bool
) -> tuple[np.ndarray, list[str]]:
    """Return (embeddings, warnings) for one person's photo folder."""
    embeddings: list[np.ndarray] = []
    warnings: list[str] = []

    photos = sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    for photo in photos:
        image = cv2.imread(str(photo))
        if image is None:
            warnings.append(f"{photo.name}: unreadable")
            continue

        faces = pipeline.detect(image)
        if not faces:
            warnings.append(f"{photo.name}: no face found")
            continue
        if len(faces) > 1:
            warnings.append(
                f"{photo.name}: {len(faces)} faces, using the largest "
                "(crop this photo if that is the wrong person)"
            )

        face = max(faces, key=lambda f: f.box[2] * f.box[3])
        if face.width < min_face:
            warnings.append(
                f"{photo.name}: face only {face.width}px wide, need {min_face}px"
            )
            continue

        x, y, w, h = face.box
        crop = image[max(y, 0) : y + h, max(x, 0) : x + w]
        if crop.size and blur_score(crop) < BLUR_THRESHOLD:
            warnings.append(f"{photo.name}: too blurry, skipped")
            continue

        embeddings.append(pipeline.embed(image, face))
        if verbose:
            print(f"    ok  {photo.name}  ({face.width}px, score {face.score:.2f})")

    return (np.stack(embeddings) if embeddings else np.empty((0, 128), np.float32)), warnings


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--min-face", type=int, default=MIN_FACE_WIDTH_PX)
    ap.add_argument("--report", action="store_true", help="list every accepted photo")
    args = ap.parse_args()

    if not ENROLL_DIR.exists():
        print(f"No enrolment folder at {ENROLL_DIR}", file=sys.stderr)
        return 1

    people = sorted(d for d in ENROLL_DIR.iterdir() if d.is_dir())
    if not people:
        print(
            f"No people found. Create one folder per person under {ENROLL_DIR}\n"
            "  e.g. dataset/enroll/ramesh/  with 15-20 photos inside",
            file=sys.stderr,
        )
        return 1

    pipeline = FacePipeline()
    db = IdentityDB()
    total_warnings = 0

    for folder in people:
        print(f"\n{folder.name}")
        embeddings, warnings = enroll_person(
            pipeline, folder, args.min_face, args.report
        )
        for w in warnings:
            print(f"    !!  {w}")
        total_warnings += len(warnings)

        if len(embeddings) == 0:
            print("    FAILED -- no usable photos, this person is not enrolled")
            continue

        db.add(folder.name, embeddings)

        # Spread of a person's own photos. Tight clusters (>0.75) mean the
        # photos are too similar -- more angle and lighting variety needed.
        if len(embeddings) > 1:
            sims = embeddings @ db.centroids[folder.name]
            print(
                f"    enrolled {len(embeddings)} photos  "
                f"self-similarity min {sims.min():.3f} mean {sims.mean():.3f}"
            )
            if sims.mean() > 0.85:
                print(
                    "    note: photos are very alike -- add more angles, "
                    "lighting and low-angle shots"
                )
        else:
            print("    enrolled 1 photo -- too few, aim for 15-20")

    if not db.centroids:
        print("\nNothing enrolled.", file=sys.stderr)
        return 1

    db.save()
    print(f"\nEnrolled {len(db.centroids)} people: {', '.join(db.names)}")
    print(f"Database written to {WS / 'db' / 'embeddings.npz'}")
    if total_warnings:
        print(f"{total_warnings} photo(s) skipped or flagged -- see !! lines above")

    # Cross-person similarity: how confusable the enrolled people are with each
    # other. Anything above ~0.5 here is a pair worth extra test photos.
    if len(db.names) > 1:
        print("\nCross-person similarity (lower is better):")
        names = db.names
        matrix = np.stack([db.centroids[n] for n in names])
        sims = matrix @ matrix.T
        for i, a in enumerate(names):
            for j, b in enumerate(names):
                if j > i and sims[i, j] > 0.4:
                    print(f"    {a} vs {b}: {sims[i, j]:.3f}  <- similar, watch this pair")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
