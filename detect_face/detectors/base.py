from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

import numpy as np


@dataclass(frozen=True)
class FaceDetection:
    """Canonical face detection used internally across backends."""
    bbox: List[float]                 # [x1,y1,x2,y2] float
    score: float                      # confidence
    landmarks: List[List[float]]      # [[x,y] x5] (may be empty if backend lacks)


@runtime_checkable
class FaceDetector(Protocol):
    """Backend contract.

    Implementations must accept BGR images (H,W,3) and return canonical detections.
    """
    name: str

    def predict(
        self,
        img_bgr: np.ndarray,
        *,
        conf_thresh: float = 0.5,
        nms_thresh: float = 0.3,
    ) -> List[FaceDetection]:
        ...


def as_minimal_face_dict(d: FaceDetection) -> Dict[str, Any]:
    """Minimal JSON-friendly representation attached to person detections."""
    lm = d.landmarks[:5] if d.landmarks else []
    lm_out: List[List[float]] = []
    for item in lm:
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            lm_out.append([float(item[0]), float(item[1])])
    return {
        "bbox": [float(v) for v in d.bbox[:4]],
        "score": float(d.score),
        "landmarks": lm_out,
        # face_ind is added later per-person (during attachment)
    }