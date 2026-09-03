"""Build the impostor set from FairFace (CC BY 4.0).

Impostors are faces that must always come back as "unknown". They are what
scripts/tune_threshold.py uses to measure the false-match rate, so the set has
to look like the people who will actually walk past the robot. A threshold
tuned against Western celebrity photos will happily misidentify an Indian
visitor as a colleague while reporting a perfect score.

Default pull is Indian faces aged 20-59, plus a smaller mixed-origin set so a
foreign visitor is not out-of-distribution.

    python scripts/fetch_impostors.py
    python scripts/fetch_impostors.py --split train --count 5000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from face_pipeline import WS

IMPOSTOR_DIR = WS / "dataset" / "impostors"

# FairFace ClassLabel orderings, confirmed against the dataset schema.
RACES = [
    "East Asian",
    "Indian",
    "Black",
    "White",
    "Middle Eastern",
    "Latino_Hispanic",
    "Southeast Asian",
]
AGES = ["0-2", "3-9", "10-19", "20-29", "30-39", "40-49", "50-59", "60-69", "more than 70"]

ATTRIBUTION = """FairFace: Face Attribute Dataset for Balanced Race, Gender, and Age
Karkkainen, K. and Joo, J., WACV 2021
https://github.com/joojs/fairface

Licensed CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/).
Images in this folder are a filtered subset, used here as an impostor /
negative set for recognition threshold calibration.
"""


def save_subset(rows, out_dir: Path, limit: int, prefix: str) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    saved = 0
    for row in rows:
        if saved >= limit:
            break
        image = row["image"]
        if image.mode != "RGB":
            image = image.convert("RGB")
        image.save(out_dir / f"{prefix}_{saved:05d}.jpg", quality=95)
        saved += 1
        if saved % 250 == 0:
            print(f"    {saved}/{limit}", flush=True)
    return saved


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", default="validation", choices=["validation", "train"])
    ap.add_argument("--config", default="1.25", choices=["0.25", "1.25"],
                    help="FairFace crop padding; 1.25 leaves margin for our detector")
    ap.add_argument("--race", default="Indian", choices=RACES + ["all"])
    ap.add_argument("--min-age", type=int, default=20)
    ap.add_argument("--max-age", type=int, default=59)
    ap.add_argument("--count", type=int, default=2000, help="target primary impostors")
    ap.add_argument("--world-count", type=int, default=300,
                    help="extra mixed-origin faces so visitors are in-distribution")
    args = ap.parse_args()

    try:
        from datasets import load_dataset
    except ImportError:
        print("datasets not installed: venv/bin/pip install datasets", file=sys.stderr)
        return 1

    keep_ages = {
        i for i, band in enumerate(AGES)
        if band != "more than 70"
        and int(band.split("-")[0]) >= args.min_age
        and int(band.split("-")[1]) <= args.max_age
    }
    if not keep_ages:
        print(f"No age band inside {args.min_age}-{args.max_age}", file=sys.stderr)
        return 1

    print(f"Loading FairFace {args.config} / {args.split} (first run downloads ~1-3 GB)")
    ds = load_dataset("HuggingFaceM4/FairFace", args.config, split=args.split)
    print(f"  {len(ds)} rows total")
    print(f"  ages kept: {sorted(AGES[i] for i in keep_ages)}")

    if args.race != "all":
        race_idx = RACES.index(args.race)
        primary = ds.filter(
            lambda r: r["race"] == race_idx and r["age"] in keep_ages,
            desc="filtering primary",
        )
        label = args.race.lower().replace(" ", "_")
    else:
        primary = ds.filter(lambda r: r["age"] in keep_ages, desc="filtering primary")
        label = "all"

    print(f"\nPrimary pool ({args.race}, {args.min_age}-{args.max_age}): {len(primary)}")
    if len(primary) < args.count:
        print(
            f"  only {len(primary)} available (asked for {args.count}) -- "
            f"rerun with --split train for a much larger pool"
        )

    n = save_subset(primary, IMPOSTOR_DIR / label, args.count, label)
    print(f"  saved {n} -> {IMPOSTOR_DIR / label}")

    world_n = 0
    if args.world_count > 0 and args.race != "all":
        race_idx = RACES.index(args.race)
        world = ds.filter(
            lambda r: r["race"] != race_idx and r["age"] in keep_ages,
            desc="filtering world",
        ).shuffle(seed=0)
        world_n = save_subset(world, IMPOSTOR_DIR / "world", args.world_count, "world")
        print(f"  saved {world_n} -> {IMPOSTOR_DIR / 'world'}")

    (IMPOSTOR_DIR / "ATTRIBUTION.txt").write_text(ATTRIBUTION)
    print(f"\nTotal impostors: {n + world_n}")
    print(f"Attribution written to {IMPOSTOR_DIR / 'ATTRIBUTION.txt'} (CC BY 4.0 requires it)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
