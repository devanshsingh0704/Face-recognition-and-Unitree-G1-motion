"""Core face detection + recognition pipeline.

Two stages, both CPU-only ONNX so the same code runs on the laptop and on
Unitree robots (G1, Go2) without modification:

  1. SCRFD-500m (default) or YuNet -- finds faces, returns box + landmarks
  2. ArcFace-mbf (default) or SFace -- aligned face crop into an embedding

Identity is decided by cosine similarity against enrolled embeddings. There is
no training step and no classifier to retrain when a person is added.
"""

from __future__ import annotations

import json
import os
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

WS = Path(__file__).resolve().parent.parent
MODELS = WS / "models"
DB_PATH = WS / "db" / "embeddings.npz"
CONFIG_PATH = WS / "db" / "config.json"

DETECTOR_MODEL = MODELS / "face_detection_yunet_2023mar.onnx"
RECOGNIZER_MODEL = MODELS / "face_recognition_sface_2021dec.onnx"

# InsightFace buffalo_s: SCRFD-500m detector + MobileFaceNet ArcFace recogniser.
# Measured against YuNet+SFace on this workspace's own benchmark:
#   separation gap   +0.071 -> +0.187   (2.6x wider)
#   cost per face    131.7ms -> 98.5ms  (25% cheaper)
#   model size       37MB -> 15.5MB
#   usable face size 32px -> 20px       (range 2.9m -> 4.7m at 640x480)
# Embeddings are 512-d and are NOT interchangeable with SFace's 128-d ones.
ARCFACE_DIR = MODELS / "candidates" / "buffalo_s"
ARCFACE_DETECTOR = ARCFACE_DIR / "det_500m.onnx"
ARCFACE_RECOGNIZER = ARCFACE_DIR / "w600k_mbf.onnx"

DEFAULT_BACKEND = "arcface"

# Provisional only -- scripts/tune_threshold.py replaces this with a value
# measured against your own enrolled people.
#
# NOT the 0.363 that OpenCV documents for SFace. Measured against 700 real
# Indian faces from FairFace, 0.363 produced a 8.4% false match rate: roughly
# one stranger in twelve would be handed a colleague's name. Observed impostor
# maximum was 0.459. 0.45 gave 0.14% and 0.50 gave zero, so this starts at 0.45.
DEFAULT_THRESHOLD = 0.45

# Cap BLAS / OpenCV threads so inference does not starve capture on the Orin.
DEFAULT_INFERENCE_THREADS = int(os.environ.get("FR_THREADS", "4"))


def configure_runtime_threads(n: int = DEFAULT_INFERENCE_THREADS) -> None:
    """Limit parallel threads before constructing models."""
    n = max(1, n)
    cv2.setNumThreads(n)
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(var, str(n))


# Faces smaller than this are too low-resolution for a reliable embedding.
#
# Measured by rescaling the held-out photos to an exact face width:
#   32px -> 20/20 identified correctly, 19/20 above threshold
#   25px -> 19/20 correct, 17/20 above threshold
#   20px -> 16/20 correct, 11/18 above threshold  (detection also starts failing)
#
# 28 sits just under the last fully-reliable size. The old value of 40 was
# discarding faces that recognise perfectly well, which cost real range: at
# 640x480 a 40px face is 2.3m away, a 28px face is 3.3m.
MIN_FACE_WIDTH_PX = 28


@dataclass
class Face:
    """One detected face in one frame."""

    box: tuple[int, int, int, int]  # x, y, w, h
    landmarks: np.ndarray  # (5, 2) -- eyes, nose, mouth corners
    score: float  # detector confidence
    raw: np.ndarray  # the 15-value detector row, needed by alignCrop

    @property
    def width(self) -> int:
        return self.box[2]


class FacePipeline:
    """Detector + recognizer. Construct once, reuse across frames."""

    def __init__(
        self,
        det_score_threshold: float = 0.5,
        nms_threshold: float = 0.3,
        top_k: int = 50,
        input_size: tuple[int, int] = (320, 320),
        backend: str = DEFAULT_BACKEND,
    ) -> None:
        """backend 'arcface' = SCRFD + ArcFace (default), 'sface' = YuNet + SFace."""
        self.backend = backend

        if backend == "arcface":
            for path in (ARCFACE_DETECTOR, ARCFACE_RECOGNIZER):
                if not path.exists():
                    raise FileNotFoundError(
                        f"Missing model: {path}\n"
                        "Download buffalo_s from the InsightFace releases page and "
                        f"unzip it into {ARCFACE_DIR}"
                    )
            from insightface.model_zoo import get_model

            self.detector = get_model(str(ARCFACE_DETECTOR))
            self.detector.prepare(ctx_id=-1, input_size=(640, 640),
                                  det_thresh=det_score_threshold)
            self.recognizer = get_model(str(ARCFACE_RECOGNIZER))
            self.recognizer.prepare(ctx_id=-1)
            self.embedding_dim = 512
            return

        for path in (DETECTOR_MODEL, RECOGNIZER_MODEL):
            if not path.exists():
                raise FileNotFoundError(f"Missing model: {path}")

        self.detector = cv2.FaceDetectorYN.create(
            str(DETECTOR_MODEL), "", input_size,
            det_score_threshold, nms_threshold, top_k,
        )
        self.recognizer = cv2.FaceRecognizerSF.create(str(RECOGNIZER_MODEL), "")
        self._input_size = input_size
        self.embedding_dim = 128

    def detect(
        self,
        image: np.ndarray,
        min_width: int = 0,
        scale: float = 1.0,
        require_in_frame: bool = False,
        min_confidence: float = 0.0,
    ) -> list[Face]:
        """Find every face in a BGR image.

        min_width drops detections too small to embed reliably. A 24px face
        carries almost no identity information -- passing it to the recogniser
        wastes time and produces meaningless scores.

        scale < 1.0 runs the detector on a downscaled copy and maps the results
        back to full-resolution coordinates. Detection cost falls roughly with
        the pixel count, and faces large enough to recognise survive the
        downscale easily. Boxes and landmarks are both linear in the scale
        factor, so the mapping back is exact.
        """
        if self.backend == "arcface":
            # SCRFD resizes to its own input size internally, so `scale` does not
            # apply -- it already runs at a fixed cost regardless of frame size.
            return self._detect_scrfd(image, min_width, require_in_frame,
                                      min_confidence)

        if scale != 1.0:
            small = cv2.resize(image, None, fx=scale, fy=scale,
                               interpolation=cv2.INTER_LINEAR)
        else:
            small = image

        h, w = small.shape[:2]
        # YuNet needs to be told the frame size before every detect on a new size.
        self.detector.setInputSize((w, h))
        _, raw = self.detector.detect(small)
        if raw is None:
            return []

        inv = 1.0 / scale
        fh, fw = image.shape[:2]
        faces = []
        for row in raw:
            row = row.copy()
            if scale != 1.0:
                row[:14] *= inv  # box corners and the 5 landmarks alike
            x, y, bw, bh = (int(v) for v in row[:4])
            if bw < min_width:
                continue
            if float(row[14]) < min_confidence:
                continue

            # A face running off the edge of the frame gives a partial crop, and
            # a partial crop does not just score badly -- it scores unreliably.
            # Measured: cropping a stranger to a forehead-only view moved their
            # similarity to an enrolled person from 0.115 to 0.281. Half a face
            # can drift toward any identity, so refuse to judge it at all.
            if require_in_frame:
                landmarks = row[4:14].reshape(5, 2)
                if (
                    x < 0 or y < 0 or x + bw > fw or y + bh > fh
                    or landmarks[:, 0].min() < 0 or landmarks[:, 0].max() > fw
                    or landmarks[:, 1].min() < 0 or landmarks[:, 1].max() > fh
                ):
                    continue

            faces.append(
                Face(
                    box=(x, y, bw, bh),
                    landmarks=row[4:14].reshape(5, 2),
                    score=float(row[14]),
                    raw=row,
                )
            )
        return faces

    def _detect_scrfd(
        self, image: np.ndarray, min_width: int, require_in_frame: bool,
        min_confidence: float,
    ) -> list[Face]:
        boxes, kpss = self.detector.detect(image, max_num=0, metric="default")
        if boxes is None or len(boxes) == 0:
            return []

        fh, fw = image.shape[:2]
        faces = []
        for i, b in enumerate(boxes):
            x, y = int(b[0]), int(b[1])
            bw, bh = int(b[2] - b[0]), int(b[3] - b[1])
            score = float(b[4])
            if bw < min_width or score < min_confidence:
                continue
            kps = None if kpss is None else np.asarray(kpss[i], dtype=np.float32)
            if require_in_frame:
                if x < 0 or y < 0 or x + bw > fw or y + bh > fh:
                    continue
                if kps is not None and (
                    kps[:, 0].min() < 0 or kps[:, 0].max() > fw
                    or kps[:, 1].min() < 0 or kps[:, 1].max() > fh
                ):
                    continue
            faces.append(Face(box=(x, y, bw, bh),
                              landmarks=kps if kps is not None else np.zeros((5, 2), np.float32),
                              score=score, raw=b))
        return faces

    def embed(self, image: np.ndarray, face: Face) -> np.ndarray:
        """Align a detected face and return its L2-normalised 128-d embedding.

        Alignment matters: SFace is trained on faces warped to a canonical
        position using the 5 landmarks. Feeding it a raw crop measurably
        degrades accuracy.
        """
        if self.backend == "arcface":
            from insightface.app.common import Face as IFace

            x, y, w, h = face.box
            iface = IFace(bbox=np.array([x, y, x + w, y + h], dtype=np.float32),
                          kps=face.landmarks, det_score=face.score)
            feat = self.recognizer.get(image, iface)
        else:
            aligned = self.recognizer.alignCrop(image, face.raw)
            feat = self.recognizer.feature(aligned)

        vec = np.asarray(feat, dtype=np.float32).flatten()
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec

    def largest_face(self, image: np.ndarray) -> Face | None:
        """Best single face in the image -- used for enrolment photos."""
        faces = self.detect(image)
        if not faces:
            return None
        return max(faces, key=lambda f: f.box[2] * f.box[3])


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Similarity of two normalised embeddings, in [-1, 1]. Higher is closer."""
    return float(np.dot(a, b))


def spherical_kmeans(X: np.ndarray, k: int, iters: int = 50) -> np.ndarray:
    """Cluster L2-normalised embeddings into k unit-norm centroids."""
    k = max(1, min(k, len(X)))
    # Deterministic seeding: evenly spaced samples. Avoids the run-to-run
    # variation of random init, which matters because enrolment must be
    # reproducible.
    centres = X[np.linspace(0, len(X) - 1, k).astype(int)].copy()
    for _ in range(iters):
        assign = np.argmax(X @ centres.T, axis=1)
        for j in range(k):
            members = X[assign == j]
            if len(members):
                v = members.mean(axis=0)
                centres[j] = v / np.linalg.norm(v)
    return centres.astype(np.float32)


# One averaged face per person cannot represent both an eye-level portrait and
# the Go2's upward view -- averaging them produces a centroid that matches
# neither well, and matches strangers better. Splitting each person into a few
# pose clusters keeps each one tight.
#
# Measured on the held-out sets (gap = genuine min minus impostor max):
#   k=1  +0.021     k=4  +0.046     k=8  +0.068     k=21 (every photo)  +0.111
#
# k=21 measures best but is not chosen: it means a single enrolment photo can
# admit a match on its own, so one bad photo compromises the identity, and the
# impostor ceiling rises as the roster grows (more photos, more chances for a
# stranger to hit one). k=8 keeps 2-3 photos behind each sub-centroid.
#
# Re-measured at 3 people (43/25/13 photos), where 8 clusters had become too
# few for the largest gallery: k=8 gave +0.123, k=10 gave +0.135. Scaling k to
# gallery size was tried and is worse -- giving a small gallery fewer clusters
# broadens them and lets impostors in (ceiling 0.372 -> 0.400). Fixed k wins.
# spherical_kmeans clamps k to the photo count, so small galleries degrade
# gracefully.
DEFAULT_SUBCENTROIDS = 10


class IdentityDB:
    """Enrolled people as a few pose sub-centroids plus every per-photo embedding.

    Matching uses the best sub-centroid, so a person seen from below is compared
    against their upward-view cluster rather than against an average of every
    pose. The per-photo embeddings are kept so threshold tuning can measure
    genuine spread and so sub-centroids can be rebuilt with a different k.
    """

    def __init__(self, n_subcentroids: int = DEFAULT_SUBCENTROIDS) -> None:
        self.centroids: dict[str, np.ndarray] = {}
        self.samples: dict[str, np.ndarray] = {}
        self.subcentroids: dict[str, np.ndarray] = {}
        self.n_subcentroids = n_subcentroids

    def add(self, name: str, embeddings: np.ndarray) -> None:
        if len(embeddings) == 0:
            raise ValueError(f"No embeddings supplied for {name!r}")
        centroid = embeddings.mean(axis=0)
        centroid /= np.linalg.norm(centroid)
        self.centroids[name] = centroid.astype(np.float32)
        self.samples[name] = embeddings.astype(np.float32)
        self.subcentroids[name] = spherical_kmeans(
            embeddings.astype(np.float32), self.n_subcentroids
        )

    @property
    def names(self) -> list[str]:
        return sorted(self.centroids)

    def scores(self, embedding: np.ndarray) -> dict[str, float]:
        """Similarity to each enrolled person, via their best-matching pose."""
        out = {}
        for name in self.names:
            subs = self.subcentroids.get(name)
            if subs is None or len(subs) == 0:
                out[name] = float(self.centroids[name] @ embedding)
            else:
                out[name] = float((subs @ embedding).max())
        return out

    def identify(self, embedding: np.ndarray, threshold: float) -> tuple[str, float]:
        """Best match for an embedding, or ("unknown", score) below threshold."""
        if not self.centroids:
            return "unknown", 0.0
        scores = self.scores(embedding)
        best = max(scores, key=scores.get)
        score = scores[best]
        return (best if score >= threshold else "unknown"), score

    def save(self, path: Path = DB_PATH) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, np.ndarray] = {}
        for name, vec in self.centroids.items():
            payload[f"centroid::{name}"] = vec
        for name, mat in self.samples.items():
            payload[f"samples::{name}"] = mat
        for name, mat in self.subcentroids.items():
            payload[f"subcentroids::{name}"] = mat
        np.savez_compressed(path, **payload)

    @classmethod
    def load(cls, path: Path = DB_PATH) -> "IdentityDB":
        if not path.exists():
            raise FileNotFoundError(
                f"No identity database at {path}\nRun scripts/enroll.py first."
            )
        db = cls()
        data = np.load(path)
        for key in data.files:
            kind, _, name = key.partition("::")
            if kind == "centroid":
                db.centroids[name] = data[key]
            elif kind == "samples":
                db.samples[name] = data[key]
            elif kind == "subcentroids":
                db.subcentroids[name] = data[key]
        return db


def _iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x1, y1 = max(ax, bx), max(ay, by)
    x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


class FaceTracker:
    """Follows faces across frames so identity can be smoothed over time.

    A single frame is a noisy vote. Motion blur, a blink or an unlucky angle can
    push one frame's score across the threshold in either direction, which shows
    up as a label flickering between a name and "unknown" -- and, worse, lets a
    single bad frame put a colleague's name on a stranger.

    Tracks are matched by box overlap and each keeps a short score history. The
    reported identity uses the median of that history, so a lone outlier frame
    cannot decide anything. It also means recognition can run every Nth frame
    instead of every frame, since a track keeps its identity in between.
    """

    def __init__(self, history: int = 9, iou_threshold: float = 0.25,
                 max_missed: int = 15) -> None:
        self.history = history
        self.iou_threshold = iou_threshold
        self.max_missed = max_missed
        self._tracks: dict[int, dict] = {}
        self._next_id = 0

    def assign(self, box: tuple[int, int, int, int]) -> int:
        """Match a detection to an existing track, or start a new one.

        IoU alone is too brittle for a walking person. At ~11 fps inference the
        gap between frames is ~90 ms, in which someone walking at 1 m/s moves
        roughly half a face width -- enough to drop IoU under any sane threshold
        and split one person into two tracks. A split track loses its confirmed
        identity and has to earn it again, which is exactly what the flicker
        looked like.

        So: predict where each track should be from its recent velocity, then
        match on centre distance measured in face widths. IoU is kept as a
        fallback for the stationary case.
        """
        cx, cy = box[0] + box[2] / 2, box[1] + box[3] / 2
        best_id, best_cost = None, None

        for tid, track in self._tracks.items():
            px, py, pw, ph = track["box"]
            vx, vy = track.get("vel", (0.0, 0.0))
            # where this track is expected to be now
            ex = px + pw / 2 + vx * (track["missed"] + 1)
            ey = py + ph / 2 + vy * (track["missed"] + 1)
            scale = max(pw, box[2], 1)
            dist = ((cx - ex) ** 2 + (cy - ey) ** 2) ** 0.5 / scale
            size_ratio = max(box[2] / max(pw, 1), pw / max(box[2], 1))

            # Same face if it is within ~1.2 face widths of the prediction and
            # has not changed size implausibly. Falls back to plain IoU.
            if dist <= 1.2 and size_ratio <= 2.0:
                cost = dist
            elif _iou(box, track["box"]) >= self.iou_threshold:
                cost = 1.5
            else:
                continue
            if best_cost is None or cost < best_cost:
                best_id, best_cost = tid, cost

        if best_id is not None:
            prev = self._tracks[best_id]["box"]
            ox, oy = prev[0] + prev[2] / 2, prev[1] + prev[3] / 2
            steps = self._tracks[best_id]["missed"] + 1
            self._tracks[best_id]["vel"] = ((cx - ox) / steps, (cy - oy) / steps)

        if best_id is None:
            best_id = self._next_id
            self._next_id += 1
            self._tracks[best_id] = {"box": box, "scores": deque(maxlen=self.history),
                                     "label": "unknown", "missed": 0,
                                     "confirmed": False, "confirmed_as": None,
                                     "steal": 0, "own_scores": {}, "vel": (0.0, 0.0)}
        self._tracks[best_id]["box"] = box
        self._tracks[best_id]["missed"] = 0
        return best_id

    def observe(self, track_id: int, label: str, score: float,
                all_scores: dict[str, float] | None = None) -> None:
        track = self._tracks[track_id]
        track["scores"].append(score)
        track["label"] = label
        if all_scores:
            # Keep each identity's own similarity so a held name can display its
            # real score rather than the winning rival's.
            track["own_scores"] = all_scores

    def smoothed(self, track_id: int) -> tuple[str, float]:
        """Identity for a track, using the median of its recent scores."""
        track = self._tracks[track_id]
        if not track["scores"]:
            return "unknown", 0.0
        return track["label"], float(np.median(track["scores"]))

    # A confirmed identity is only surrendered if a *different* person wins
    # convincingly this many times in a row. One odd frame cannot steal a name.
    STEAL_STREAK = 4

    def decide(
        self, track_id: int, acquire: float, hold_ratio: float = 0.85
    ) -> tuple[str, float, bool]:
        """Identity belongs to the track, not to the frame.

        Turning sideways, blinking, or walking through a shadow drops the score
        far below threshold even though the person has not changed. Deciding
        again every frame makes the label strobe between a name and "unknown" --
        which is what a profile view looked like before this.

        So: earning a name needs the full tuned threshold, and it must be earned
        against strangers, which is where the security property lives. But once
        earned, the name stays for as long as the track is continuously
        followed. The track ends when the person leaves frame (`max_missed`),
        and the identity ends with it.

        The one way to lose a name early is for a *different* enrolled person to
        clear the acquire threshold on STEAL_STREAK consecutive recognitions --
        which covers the tracker mistakenly hopping between two nearby people.

        Returns (label, score, confirmed).
        """
        track = self._tracks[track_id]
        if not track["scores"]:
            return "unknown", 0.0, False

        score = float(np.median(track["scores"]))
        best = track["label"]

        if track["confirmed"]:
            held = track["confirmed_as"]
            own = track["own_scores"].get(held)
            own_score = own if own is not None else score
            hold_floor = acquire * hold_ratio
            if best != held and score >= acquire:
                track["steal"] = track.get("steal", 0) + 1
                if track["steal"] >= self.STEAL_STREAK:
                    track["confirmed_as"] = best
                    track["steal"] = 0
                    return best, score, True
            else:
                track["steal"] = 0
            if own_score < hold_floor:
                track["confirmed"] = False
                track["confirmed_as"] = None
                track["steal"] = 0
                return "unknown", own_score, False
            return held, own_score, True

        if score >= acquire and best != "unknown":
            track["confirmed"] = True
            track["confirmed_as"] = best
            track["steal"] = 0
            return best, score, True

        return "unknown", score, False

    def needs_recognition(self, track_id: int, every: int, frame_no: int) -> bool:
        """True when this track should be re-embedded on this frame."""
        track = self._tracks[track_id]
        return not track["scores"] or (frame_no + track_id) % every == 0

    def end_frame(self, seen: set[int]) -> None:
        """Age out tracks that were not matched this frame."""
        for tid in list(self._tracks):
            if tid in seen:
                continue
            self._tracks[tid]["missed"] += 1
            if self._tracks[tid]["missed"] > self.max_missed:
                del self._tracks[tid]


def open_camera(
    index: int,
    width: int = 640,
    height: int = 480,
    warmup_timeout: float = 25.0,
    verbose: bool = True,
):
    """Open a V4L2 camera and block until it actually delivers image data.

    Some UVC modules -- the HP HD Camera in this laptop among them -- accept
    the open, report success, and then emit pure black for the first several
    seconds. Measured here: ~8 s before the first real frame. Code that starts
    reading immediately sees black and looks broken, so wait for real pixels
    rather than assuming the device is ready when open() returns.

    The V4L2 backend is requested explicitly; OpenCV's automatic choice can
    land on a backend that never recovers from the blank period.
    """
    import time

    cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open /dev/video{index}")

    # MJPG must be requested before the resolution. UVC cameras commonly expose
    # high resolutions only in MJPG; in raw YUYV this laptop's camera silently
    # caps at 640x480 no matter what is asked for. Resolution is not cosmetic
    # here -- a larger face yields a better embedding, and at 640x480 measured
    # similarity drops by roughly 0.1 versus the enrolment photos.
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

    # Resolution must be set before the warm-up: on some drivers changing it
    # mid-stream restarts capture and re-triggers the blank period.
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    if verbose:
        print(f"Opening /dev/video{index} -- this camera needs a few seconds "
              "to wake up...", flush=True)

    start = time.time()
    frame = None
    while time.time() - start < warmup_timeout:
        ok, f = cap.read()
        if ok and f is not None:
            frame = f
            # Real imagery has spatial variation; a blanked sensor does not.
            if f.std() > 3.0:
                if verbose:
                    print(f"  camera ready after {time.time() - start:.1f}s "
                          f"({f.shape[1]}x{f.shape[0]}, mean {f.mean():.0f})", flush=True)
                return cap
        time.sleep(0.05)

    cap.release()
    detail = (f"last frame mean {frame.mean():.2f}, std {frame.std():.2f}"
              if frame is not None else "no frames at all")
    raise RuntimeError(
        f"/dev/video{index} delivered no real image within {warmup_timeout:.0f}s "
        f"({detail}).\nCheck the privacy shutter, the camera-mute key, and that "
        "no other app holds the camera."
    )


def load_threshold() -> float:
    """Tuned threshold if tune_threshold.py has run, else the default."""
    if CONFIG_PATH.exists():
        return float(json.loads(CONFIG_PATH.read_text())["threshold"])
    return DEFAULT_THRESHOLD


def save_threshold(threshold: float, metrics: dict) -> None:
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(
        json.dumps({"threshold": threshold, "metrics": metrics}, indent=2)
    )
