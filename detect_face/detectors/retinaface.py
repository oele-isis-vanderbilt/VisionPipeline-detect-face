from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np

from .base import FaceDetection, FaceDetector

try:  # soft dependency
    from retinaface.pre_trained_models import get_model  # type: ignore
except Exception as e:  # pragma: no cover
    raise ImportError(
        "retinaface-pytorch is required. Install with: pip install -U retinaface-pytorch"
    ) from e

__all__ = ["RetinaFaceDetector", "get_retinaface_detector", "available_retinaface_variants"]


class RetinaFaceDetector(FaceDetector):
    """RetinaFace backend wrapper (retinaface-pytorch).

    Produces canonical FaceDetection objects with bbox/score/landmarks.
    """

    name = "retinaface"

    def __init__(
        self,
        *,
        variant: str = "resnet50_2020-07-20",
        max_size: int = 1024,
        device: Optional[str] = None,
    ) -> None:
        self.variant = str(variant)
        self.max_size = int(max_size)
        self.device = device
        self._model = self._init_model()

    def _init_model(self):
        # Some versions accept device=..., some do not.
        try:
            if self.device is None:
                model = get_model(self.variant, max_size=self.max_size)
            else:
                try:
                    model = get_model(self.variant, max_size=self.max_size, device=self.device)  # type: ignore[call-arg]
                except TypeError:
                    model = get_model(self.variant, max_size=self.max_size)
        except Exception as e:  # pragma: no cover
            raise RuntimeError(f"Failed to load RetinaFace model '{self.variant}': {e}") from e

        try:
            model.eval()
        except Exception:
            pass
        return model

    def predict(
        self,
        img_bgr: np.ndarray,
        *,
        conf_thresh: float = 0.5,
        nms_thresh: float = 0.3,
    ) -> List[FaceDetection]:
        if img_bgr is None or not isinstance(img_bgr, np.ndarray):
            return []
        if img_bgr.ndim != 3 or img_bgr.shape[2] != 3:
            return []

        # Forward kwargs when supported
        try:
            raw = self._model.predict_jsons(
                img_bgr,
                confidence_threshold=float(conf_thresh),
                nms_threshold=float(nms_thresh),
            )
        except TypeError:
            raw = self._model.predict_jsons(img_bgr)
        except Exception:
            return []

        if not raw:
            return []

        out: List[FaceDetection] = []
        for det in raw:
            try:
                bbox = det.get("bbox")
                if bbox is None or len(bbox) < 4:
                    continue
                x1, y1, x2, y2 = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
                score = float(det.get("score", 0.0))

                lms = det.get("landmarks") or []
                lm_out: List[List[float]] = []
                if isinstance(lms, Sequence):
                    for item in lms:
                        if isinstance(item, Sequence) and len(item) >= 2:
                            lm_out.append([float(item[0]), float(item[1])])
                if len(lm_out) >= 5:
                    lm_out = lm_out[:5]

                out.append(FaceDetection(bbox=[x1, y1, x2, y2], score=score, landmarks=lm_out))
            except Exception:
                continue
        return out


def get_retinaface_detector(
    *,
    variant: str = "resnet50_2020-07-20",
    max_size: int = 1024,
    device: Optional[str] = None,
) -> RetinaFaceDetector:
    return RetinaFaceDetector(variant=variant, max_size=max_size, device=device)


def available_retinaface_variants() -> List[str]:
    """Return supported *named* RetinaFace variants for this backend.

    Notes
    -----
    - The `retinaface-pytorch` package's `get_model()` registry currently exposes a
      small set of pre-trained model keys (often just one).
    - This function is intentionally conservative and lists only the known
      stable variant(s) for auto-download by name.
    """
    return ["resnet50_2020-07-20"]