"""Cross-person evaluation: can the system tell enrolled people apart?

Rejecting strangers is the easy half. The hard half is separating the people
you enrolled from each other, and it cannot be measured at all until at least
two are enrolled.

Reports, for every photo:
  own    -- similarity to that person's own centroid
  best   -- the identity actually chosen (nearest centroid)
  margin -- own minus the best competing identity; NEGATIVE means a mix-up

A confusion matrix follows, then the identity-level summary. Enrolment photos
are marked because they built the centroid they are scored against, so their
"own" figures are optimistic; test photos are the honest ones.

    python scripts/evaluate.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from face_pipeline import WS, FacePipeline, IdentityDB, load_threshold

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def score_folder(pipeline, db, folder: Path, names, centroids):
    """Return per-photo (filename, own, best_name, best_score, margin)."""
    rows = []
    person = folder.name
    own_idx = names.index(person)
    for photo in sorted(folder.iterdir()):
        if photo.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        image = cv2.imread(str(photo))
        if image is None:
            continue
        face = pipeline.largest_face(image)
        if face is None:
            continue
        scored = db.scores(pipeline.embed(image, face))
        scores = np.array([scored[n] for n in names])
        own = float(scores[own_idx])
        order = np.argsort(scores)[::-1]
        best = int(order[0])
        rival = int(order[1]) if len(order) > 1 else best
        margin = own - float(scores[rival]) if best == own_idx else own - float(scores[best])
        rows.append((photo.name, own, names[best], float(scores[best]), margin))
    return rows


def main() -> int:
    db = IdentityDB.load()
    pipeline = FacePipeline()
    threshold = load_threshold()
    names = db.names
    centroids = np.stack([db.centroids[n] for n in names])

    if len(names) < 2:
        print("Need at least 2 enrolled people to measure confusion.", file=sys.stderr)
        return 1

    print(f"Enrolled: {', '.join(names)}    threshold {threshold:.2f}\n")

    print("Centroid-to-centroid similarity")
    sims = centroids @ centroids.T
    header = "".join(f"{n:>12}" for n in names)
    print(f"{'':>12}{header}")
    for i, a in enumerate(names):
        row = "".join(f"{sims[i, j]:>12.3f}" for j in range(len(names)))
        print(f"{a:>12}{row}")
    worst = max(
        ((sims[i, j], names[i], names[j]) for i in range(len(names))
         for j in range(len(names)) if i < j), default=(0, "", "")
    )
    if worst[0] >= threshold:
        print(f"\n  !! {worst[1]} and {worst[2]} are {worst[0]:.3f} alike -- above the "
              f"{threshold:.2f} threshold.")
        print("  !! Centroid similarity is not the same as a mix-up, but it is the "
              "warning sign.\n")

    all_rows = []
    for split in ("enroll", "test"):
        base = WS / "dataset" / split
        if not base.exists():
            continue
        for folder in sorted(d for d in base.iterdir() if d.is_dir()):
            if folder.name not in names:
                continue
            rows = score_folder(pipeline, db, folder, names, centroids)
            if not rows:
                continue
            tag = "ENROLMENT (optimistic)" if split == "enroll" else "TEST (honest)"
            print(f"\n{folder.name} -- {tag}")
            print(f"  {'photo':<20}{'own':>8}{'chosen':>12}{'score':>8}{'margin':>9}")
            for fn, own, best, bscore, margin in rows:
                flag = "  <-- WRONG PERSON" if best != folder.name else (
                    "  <-- below threshold" if own < threshold else "")
                print(f"  {fn:<20}{own:>8.3f}{best:>12}{bscore:>8.3f}{margin:>9.3f}{flag}")
            all_rows.append((split, folder.name, rows))

    print("\n" + "=" * 62)
    print("SUMMARY")
    print("=" * 62)
    for split, person, rows in all_rows:
        wrong = [r for r in rows if r[2] != person]
        below = [r for r in rows if r[1] < threshold]
        owns = np.array([r[1] for r in rows])
        margins = np.array([r[4] for r in rows])
        label = "enrol" if split == "enroll" else "test "
        print(f"  {person:<10} {label}  n={len(rows):<3} "
              f"own min {owns.min():.3f}  margin min {margins.min():+.3f}  "
              f"misidentified {len(wrong)}  below-threshold {len(below)}")
        for fn, own, best, bscore, _ in wrong:
            print(f"      {fn} -> called {best} ({bscore:.3f}) instead of {person}")

    total = sum(len(r) for _, _, r in all_rows)
    total_wrong = sum(len([x for x in r if x[2] != p]) for _, p, r in all_rows)
    print(f"\n  identification accuracy: {total - total_wrong}/{total} "
          f"= {(total - total_wrong) / total:.1%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
