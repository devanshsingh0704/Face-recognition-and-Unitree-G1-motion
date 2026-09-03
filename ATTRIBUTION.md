# Attribution

## FairFace — impostor set for threshold tuning

Threshold tuning measures the false-match rate against a large set of faces
belonging to nobody enrolled. That set is **FairFace**:

> Karkkainen, K. and Joo, J. *FairFace: Face Attribute Dataset for Balanced
> Race, Gender, and Age for Bias Measurement and Mitigation.*
> IEEE/CVF Winter Conference on Applications of Computer Vision (WACV), 2021.
> https://github.com/joojs/fairface

Licensed **CC BY 4.0** — https://creativecommons.org/licenses/by/4.0/

`scripts/fetch_impostors.py` downloads it and filters to the deployment
population. **The images are not redistributed here**; `dataset/` is excluded
from this repository.

Why this dataset and not another: the threshold is only meaningful against
faces resembling the people the system will actually meet. Tuning against a set
that does not match the deployment population produces a reassuring number and
a system that fails in the field. **DigiFace-1M was considered and rejected** —
its licence permits non-commercial research use only.

## Face detection and recognition models

**buffalo_s** from InsightFace — SCRFD-500m detector (`det_500m.onnx`) plus
ArcFace MobileFaceNet recogniser (`w600k_mbf.onnx`).

> https://github.com/deepinsight/insightface

Fetched by `install.sh` from the InsightFace v0.7 release. Not redistributed
here; `models/` is excluded from this repository.

**YuNet** (`face_detection_yunet_2023mar.onnx`) from OpenCV Zoo, kept as a
fallback detector.

> https://github.com/opencv/opencv_zoo

## Robot SDK

Locomotion uses `unitree_sdk2_python`.

> https://github.com/unitreerobotics/unitree_sdk2_python

## Not included in this repository

No enrolled person's data is published here. Face embeddings
(`db/embeddings.npz`), reference scores (`db/baseline.json`) and all photographs
(`dataset/`) are excluded — they are biometric identifiers of identifiable
people. Build your own gallery with `scripts/enroll.py`.
