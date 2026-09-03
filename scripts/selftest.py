"""Prove the pipeline works before any colleague photos exist.

FairFace has no identity labels -- every image is a different person -- so real
recognition cannot be tested with it directly. Instead this builds pseudo-people
by augmenting single faces (flip, brightness, rotation, scale), enrols some
variants and scores the held-out ones. That exercises every stage for real:
detection, landmark alignment, embedding, centroid building and matching.

It is a plumbing and sanity check, not an accuracy claim. Real numbers come
from tune_threshold.py once actual photos are enrolled.

    python scripts/selftest.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from face_pipeline import WS, FacePipeline, IdentityDB, cosine_similarity

IMPOSTOR_DIR = WS / "dataset" / "impostors"
N_PSEUDO_PEOPLE = 6
N_ENROL_VARIANTS = 5


def augment(image: np.ndarray) -> list[np.ndarray]:
    """Plausible appearance variation for one face."""
    h, w = image.shape[:2]
    out = [image, cv2.flip(image, 1)]

    for beta in (-35, 35):  # darker / brighter
        out.append(cv2.convertScaleAbs(image, alpha=1.0, beta=beta))

    for angle in (-12, 12):  # head tilt
        m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        out.append(cv2.warpAffine(image, m, (w, h), borderMode=cv2.BORDER_REPLICATE))

    # Distance: shrink then restore, losing real detail the way a far face does.
    small = cv2.resize(image, (w // 3, h // 3), interpolation=cv2.INTER_AREA)
    out.append(cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR))
    return out


def main() -> int:
    faces_dir = IMPOSTOR_DIR / "indian"
    photos = sorted(faces_dir.glob("*.jpg"))
    if len(photos) < N_PSEUDO_PEOPLE + 100:
        print(f"Need impostors first: python scripts/fetch_impostors.py", file=sys.stderr)
        return 1

    pipeline = FacePipeline()
    print("1. models load                          OK")

    # --- detection rate on real photos ---------------------------------
    sample = photos[:200]
    detected = 0
    widths = []
    for p in sample:
        img = cv2.imread(str(p))
        face = pipeline.largest_face(img)
        if face is not None:
            detected += 1
            widths.append(face.width)
    rate = detected / len(sample)
    print(f"2. detection on {len(sample)} real faces      "
          f"{detected}/{len(sample)} = {rate:.1%}  "
          f"(median face {int(np.median(widths))}px)")
    if rate < 0.9:
        print("   !! low detection rate -- investigate before trusting anything")

    # --- pseudo-identity enrolment -------------------------------------
    db = IdentityDB()
    holdout: dict[str, list[np.ndarray]] = {}
    people_used = 0
    idx = 0
    while people_used < N_PSEUDO_PEOPLE and idx < len(photos):
        img = cv2.imread(str(photos[idx]))
        idx += 1
        if img is None:
            continue
        variants = augment(img)
        embs = []
        for v in variants:
            face = pipeline.largest_face(v)
            if face is not None:
                embs.append(pipeline.embed(v, face))
        if len(embs) < N_ENROL_VARIANTS + 1:
            continue
        name = f"person_{people_used}"
        db.add(name, np.stack(embs[:N_ENROL_VARIANTS]))
        holdout[name] = embs[N_ENROL_VARIANTS:]
        people_used += 1

    print(f"3. enrolled {people_used} pseudo-people          "
          f"{N_ENROL_VARIANTS} variants each, rest held out")

    # --- genuine vs impostor -------------------------------------------
    genuine = [
        cosine_similarity(db.centroids[name], emb)
        for name, embs in holdout.items()
        for emb in embs
    ]

    centroids = np.stack([db.centroids[n] for n in db.names])
    impostor_scores = []
    for p in photos[idx : idx + 400]:
        img = cv2.imread(str(p))
        if img is None:
            continue
        face = pipeline.largest_face(img)
        if face is None:
            continue
        impostor_scores.append(float((centroids @ pipeline.embed(img, face)).max()))

    g, im = np.array(genuine), np.array(impostor_scores)
    print(f"4. genuine  n={len(g):<4} min {g.min():.3f} mean {g.mean():.3f}")
    print(f"   impostor n={len(im):<4} max {im.max():.3f} mean {im.mean():.3f}")

    gap = g.min() - im.max()
    print(f"5. separation gap {gap:+.3f}", end="   ")
    if gap > 0:
        print("PASS -- distributions do not overlap")
    else:
        print("overlap present (expected: augmentation is harsher than real photos)")

    # Threshold that would perfectly separate, for orientation only.
    best_t, best_err = 0.0, 1e9
    for t in np.arange(0.2, 0.8, 0.01):
        err = float((im >= t).mean()) + float((g < t).mean())
        if err < best_err:
            best_t, best_err = float(t), err
    fmr = float((im >= best_t).mean())
    fnmr = float((g < best_t).mean())
    print(f"6. best separating threshold {best_t:.2f}  FMR {fmr:.3f}  FNMR {fnmr:.3f}")

    print("\nPipeline is working end to end. These numbers are from augmented")
    print("copies of single photos, not real people -- they validate the plumbing,")
    print("not the accuracy. Enrol real photos, then run tune_threshold.py.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
