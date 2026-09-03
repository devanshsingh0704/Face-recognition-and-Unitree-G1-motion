"""Measure the right similarity threshold instead of guessing it.

Two distributions matter:

  genuine  -- test photos of enrolled people vs their own centroid
  impostor -- impostor faces vs every enrolled centroid

A good threshold sits in the gap between them. This script sweeps candidate
thresholds and reports, at each one:

  FMR  false match rate      -- impostors wrongly given a name  (security risk)
  FNMR false non-match rate  -- colleagues wrongly called unknown (annoyance)

For a robot that greets people, FMR is the one that hurts: calling a stranger
"Ramesh" is worse than failing to recognise Ramesh once. The default policy
therefore picks the loosest threshold that still holds FMR at or below target.

    python scripts/tune_threshold.py
    python scripts/tune_threshold.py --target-fmr 0.001
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from face_pipeline import WS, FacePipeline, IdentityDB, save_threshold

TEST_DIR = WS / "dataset" / "test"
IMPOSTOR_DIR = WS / "dataset" / "impostors"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def embed_folder(pipeline: FacePipeline, folder: Path) -> list[np.ndarray]:
    """Embed the largest face in every image under folder (recursively)."""
    out = []
    for photo in sorted(folder.rglob("*")):
        if photo.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        image = cv2.imread(str(photo))
        if image is None:
            continue
        face = pipeline.largest_face(image)
        if face is None:
            continue
        out.append(pipeline.embed(image, face))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--target-fmr", type=float, default=0.001,
                    help="max acceptable false match rate (default 0.1%%)")
    ap.add_argument("--save", action="store_true", help="write the chosen threshold")
    args = ap.parse_args()

    db = IdentityDB.load()
    pipeline = FacePipeline()
    names = db.names
    centroids = np.stack([db.centroids[n] for n in names])
    print(f"Enrolled: {', '.join(names)}")

    # --- genuine scores -------------------------------------------------
    genuine: list[float] = []
    genuine_missing: list[str] = []
    for name in names:
        folder = TEST_DIR / name
        if not folder.exists():
            genuine_missing.append(name)
            continue
        for emb in embed_folder(pipeline, folder):
            genuine.append(db.scores(emb)[name])

    if genuine_missing:
        print(f"No test photos for: {', '.join(genuine_missing)}  (dataset/test/<name>/)")

    # Fall back to enrolment photos so the script is still usable before test
    # photos exist -- but say so loudly, because scores measured this way are
    # optimistic: those photos built the centroid they are being scored against.
    used_enrol_fallback = False
    if not genuine:
        used_enrol_fallback = True
        for name in names:
            for emb in db.samples.get(name, []):
                genuine.append(float(db.centroids[name] @ emb))
        print("\n!! No test photos found -- falling back to enrolment photos.")
        print("!! These numbers are OPTIMISTIC. Add dataset/test/<name>/ photos")
        print("!! shot on a different day before trusting any threshold.\n")

    # --- impostor scores ------------------------------------------------
    impostor_embs = embed_folder(pipeline, IMPOSTOR_DIR) if IMPOSTOR_DIR.exists() else []
    if not impostor_embs:
        print(f"No impostors under {IMPOSTOR_DIR} -- run scripts/fetch_impostors.py",
              file=sys.stderr)
        return 1

    # Every impostor is compared against every enrolled person -- and against
    # every pose sub-centroid within each person. Any of those scoring above
    # threshold is a false match, so this is the number that must stay low.
    impostor_best = np.array([max(db.scores(e).values()) for e in impostor_embs])

    g = np.array(genuine)
    print(f"genuine  n={len(g):<6} min {g.min():.3f}  mean {g.mean():.3f}  max {g.max():.3f}")
    print(f"impostor n={len(impostor_best):<6} min {impostor_best.min():.3f}  "
          f"mean {impostor_best.mean():.3f}  max {impostor_best.max():.3f}")

    gap = g.min() - impostor_best.max()
    print(f"\nseparation gap: {gap:+.3f}", end="  ")
    print("(clean separation)" if gap > 0 else "(distributions OVERLAP -- see below)")

    # --- sweep ----------------------------------------------------------
    print(f"\n{'thresh':>7} {'FMR':>9} {'FNMR':>9}   {'accept':>7}")
    print("-" * 38)
    rows = []
    for t in np.arange(0.20, 0.75, 0.01):
        fmr = float((impostor_best >= t).mean())
        fnmr = float((g < t).mean())
        rows.append((float(t), fmr, fnmr))
        if abs(t * 100 % 5) < 1e-6:
            print(f"{t:7.2f} {fmr:9.4f} {fnmr:9.4f}   {1 - fnmr:7.1%}")

    # When the two distributions are cleanly separated, the best threshold is
    # the middle of the gap, not the loosest value that scrapes past the FMR
    # target. Hugging the impostor ceiling leaves no headroom for a stranger who
    # happens to score higher than anyone in this particular impostor set --
    # which is exactly how a live face gets mistaken for a colleague.
    ok = [r for r in rows if r[1] <= args.target_fmr]
    if gap > 0.02:
        chosen = round((float(g.min()) + float(impostor_best.max())) / 2, 2)
        fmr = float((impostor_best >= chosen).mean())
        fnmr = float((g < chosen).mean())
        print(f"\nChosen threshold {chosen:.2f} -- midpoint of a clean gap")
        print(f"  margin above impostor max ({impostor_best.max():.3f}): "
              f"{chosen - impostor_best.max():+.3f}")
        print(f"  margin below genuine min  ({g.min():.3f}): {g.min() - chosen:+.3f}")
        print(f"  false match rate     {fmr:.4f}")
        print(f"  false non-match rate {fnmr:.4f}")
    elif ok:
        chosen, fmr, fnmr = min(ok, key=lambda r: r[0])
        print(f"\nChosen threshold {chosen:.2f} for FMR <= {args.target_fmr:.2%}")
        print(f"  false match rate     {fmr:.4f}  ({int(fmr * len(impostor_best))} of {len(impostor_best)} impostors)")
        print(f"  false non-match rate {fnmr:.4f}  ({int(fnmr * len(g))} of {len(g)} genuine)")
        print(f"  colleagues recognised {1 - fnmr:.1%} of the time")
    else:
        chosen, fmr, fnmr = max(rows, key=lambda r: r[0])
        print(f"\n!! Cannot reach FMR <= {args.target_fmr:.2%} at any threshold.")
        print("!! Best available is the strictest setting. Usually this means")
        print("!! the enrolment photos are too few or too uniform.")

    resolution = 1.0 / len(impostor_best)
    print(f"\nMeasurement floor: {resolution:.4f} -- with {len(impostor_best)} impostors "
          f"you cannot resolve an FMR below {resolution:.2%}.")
    if used_enrol_fallback:
        print("Reminder: genuine scores came from enrolment photos, so FNMR is optimistic.")

    if args.save:
        save_threshold(chosen, {
            "fmr": fmr, "fnmr": fnmr,
            "n_genuine": len(g), "n_impostor": len(impostor_best),
            "target_fmr": args.target_fmr,
            "genuine_from_enrolment": used_enrol_fallback,
            "impostor_max": float(impostor_best.max()),
            "genuine_min": float(g.min()),
        })
        print(f"\nSaved to {WS / 'db' / 'config.json'}")
    else:
        print("\n(run with --save to write this threshold)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
