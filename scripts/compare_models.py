"""Head-to-head comparison of detector + recogniser pairs on our own data.

Swapping models on vibes is how projects end up slower and less accurate at the
same time. We have a labelled benchmark -- 69 enrolment photos, 20 held-out test
photos, 1,360 impostors -- so a candidate can be judged instead of guessed at.

Every pipeline is measured identically: same photos, same 8 pose sub-centroids,
same cosine matching, same gap definition. The only variable is the models.

    python scripts/compare_models.py
    python scripts/compare_models.py --impostors 400     # quicker pass

Reported per pipeline:
  identification accuracy   did each test photo pick the right person
  genuine min               worst score a real person got
  impostor max              best score any of 1,360 strangers got
  gap                       genuine min - impostor max; bigger is better
  detect / embed ms         cost per frame and per face
"""

from __future__ import annotations

import argparse
import glob
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from face_pipeline import MIN_FACE_WIDTH_PX, WS, FacePipeline, spherical_kmeans

CAND = WS / "models" / "candidates"
K_SUB = 8


def l2(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


class CurrentPipeline:
    """YuNet + SFace, exactly as the workspace runs today."""

    name = "YuNet + SFace (current)"

    def __init__(self) -> None:
        self.p = FacePipeline()

    def detect(self, img):
        f = self.p.largest_face(img)
        return f

    def embed(self, img, f):
        return self.p.embed(img, f)


class InsightPipeline:
    """SCRFD detector + ArcFace recogniser from an InsightFace model pack."""

    def __init__(self, name: str, det: Path, rec: Path, det_size: int = 640) -> None:
        from insightface.model_zoo import get_model

        self.name = name
        self.det = get_model(str(det))
        self.det.prepare(ctx_id=-1, input_size=(det_size, det_size))
        self.rec = get_model(str(rec))
        self.rec.prepare(ctx_id=-1)

    def detect(self, img):
        bboxes, kpss = self.det.detect(img, max_num=0, metric="default")
        if bboxes is None or len(bboxes) == 0:
            return None
        # largest box, to match the current pipeline's behaviour
        areas = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])
        i = int(np.argmax(areas))
        return (bboxes[i], None if kpss is None else kpss[i])

    def embed(self, img, f):
        from insightface.app.common import Face

        bbox, kps = f
        face = Face(bbox=bbox[:4], kps=kps, det_score=float(bbox[4]))
        return l2(np.asarray(self.rec.get(img, face), dtype=np.float32).flatten())


def face_width(pipeline, f) -> int:
    if isinstance(pipeline, CurrentPipeline):
        return f.width
    bbox = f[0]
    return int(bbox[2] - bbox[0])


BLUR_THRESHOLD = 45.0  # same variance-of-Laplacian gate enroll.py uses


def passes_quality(img, pipeline, f) -> bool:
    """Blur and size gates, matching scripts/enroll.py.

    Leaving these out cost +0.070 of impostor ceiling when measured: rejected
    blurry photos cluster into their own sub-centroid, and that cluster matches
    strangers well. The gates must be applied to every candidate pipeline or
    the comparison is measuring data hygiene rather than models.
    """
    if face_width(pipeline, f) < MIN_FACE_WIDTH_PX:
        return False
    if isinstance(pipeline, CurrentPipeline):
        x, y, w, h = f.box
    else:
        b = f[0]
        x, y, w, h = int(b[0]), int(b[1]), int(b[2] - b[0]), int(b[3] - b[1])
    crop = img[max(y, 0): y + h, max(x, 0): x + w]
    if crop.size == 0:
        return False
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var()) >= BLUR_THRESHOLD


def embed_folder(pipeline, pattern: str, limit: int | None = None,
                 gated: bool = False):
    """Return (embeddings, detect_ms, embed_ms, n_files, n_detected)."""
    embs, dts, ets = [], [], []
    files = sorted(glob.glob(pattern))
    if limit:
        files = files[:limit]
    for fn in files:
        img = cv2.imread(fn)
        if img is None:
            continue
        t = time.perf_counter()
        f = pipeline.detect(img)
        dts.append((time.perf_counter() - t) * 1000)
        if f is None:
            continue
        if gated and not passes_quality(img, pipeline, f):
            continue
        t = time.perf_counter()
        embs.append(pipeline.embed(img, f))
        ets.append((time.perf_counter() - t) * 1000)
    return (np.array(embs), float(np.mean(dts or [0])), float(np.mean(ets or [0])),
            len(files), len(embs))


def evaluate(pipeline, people, n_impostors) -> dict:
    subs, det_ms, emb_ms, used = {}, [], [], {}
    for who in people:
        e, d, m, nf, nd = embed_folder(
            pipeline, f"{WS}/dataset/enroll/{who}/*.jpg", gated=True)
        used[who] = (nd, nf)
        if len(e) == 0:
            raise RuntimeError(f"{pipeline.name}: no enrolment embeddings for {who}")
        subs[who] = spherical_kmeans(e.astype(np.float32), K_SUB)
        det_ms.append(d)
        emb_ms.append(m)

    def score_all(emb):
        return {w: float((subs[w] @ emb).max()) for w in people}

    genuine, correct, total, missed = [], 0, 0, 0
    for who in people:
        e, d, m, nf, nd = embed_folder(pipeline, f"{WS}/dataset/test/{who}/*.jpg")
        missed += nf - nd
        for emb in e:
            s = score_all(emb)
            genuine.append(s[who])
            correct += max(s, key=s.get) == who
            total += 1

    imp, d, m, nf, nd = embed_folder(
        pipeline, f"{WS}/dataset/impostors/*/*.jpg", limit=n_impostors
    )
    impostor = np.array([max(score_all(e).values()) for e in imp])

    g = np.array(genuine)
    return {
        "name": pipeline.name,
        "acc": f"{correct}/{total}",
        "missed": missed,
        "gmin": g.min(), "gmean": g.mean(),
        "imax": impostor.max(), "ip99": float(np.percentile(impostor, 99)),
        "gap": g.min() - impostor.max(),
        "det_ms": float(np.mean(det_ms)), "emb_ms": float(np.mean(emb_ms)),
        "n_imp": len(impostor), "dim": len(imp[0]) if len(imp) else 0,
        "used": used,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--impostors", type=int, default=1360)
    ap.add_argument("--only", nargs="*", default=None, metavar="PACK",
                    help="restrict to these model packs, e.g. --only buffalo_s")
    args = ap.parse_args()

    people = sorted(d.name for d in (WS / "dataset" / "enroll").iterdir() if d.is_dir())
    print(f"people: {', '.join(people)}   impostors: {args.impostors}\n")

    candidates = [CurrentPipeline()]
    for label, pack, det, rec in [
        ("SCRFD-500m + ArcFace-mbf (buffalo_s)", "buffalo_s", "det_500m.onnx", "w600k_mbf.onnx"),
        ("SCRFD-10g + ArcFace-r50 (buffalo_l)", "buffalo_l", "det_10g.onnx", "w600k_r50.onnx"),
    ]:
        if args.only is not None and pack not in args.only:
            continue
        d, r = CAND / pack / det, CAND / pack / rec
        if d.exists() and r.exists():
            candidates.append(InsightPipeline(label, d, r))
        else:
            print(f"skipping {label}: models not found under {CAND / pack}")

    rows = []
    for c in candidates:
        print(f"running {c.name} ...", flush=True)
        try:
            rows.append(evaluate(c, people, args.impostors))
        except Exception as exc:
            print(f"  FAILED: {type(exc).__name__}: {exc}")

    print(f"\n{'pipeline':<38}{'dim':>5}{'acc':>8}{'miss':>6}"
          f"{'gen min':>9}{'imp max':>9}{'gap':>8}{'det ms':>8}{'emb ms':>8}")
    print("-" * 99)
    for r in rows:
        u = " ".join(f"{k}:{v[0]}/{v[1]}" for k, v in r["used"].items())
        print(f"{r['name']:<38}{r['dim']:>5}{r['acc']:>8}{r['missed']:>6}"
              f"{r['gmin']:>9.3f}{r['imax']:>9.3f}{r['gap']:>+8.3f}"
              f"{r['det_ms']:>8.1f}{r['emb_ms']:>8.1f}   {u}")

    if len(rows) > 1:
        base = rows[0]
        print(f"\nversus current (gap {base['gap']:+.3f}):")
        for r in rows[1:]:
            delta = r["gap"] - base["gap"]
            verdict = "BETTER" if delta > 0.01 else ("worse" if delta < -0.01 else "no real change")
            print(f"  {r['name']:<38} gap {delta:+.3f}  {verdict}"
                  f"   speed {(r['det_ms']+r['emb_ms'])/(base['det_ms']+base['emb_ms']):.1f}x cost")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
