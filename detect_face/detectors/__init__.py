from __future__ import annotations

from typing import List, Optional

from .base import FaceDetector
from .retinaface import get_retinaface_detector, available_retinaface_variants

__all__ = [
    "get_face_detector",
    "available_detectors",
    "available_variants",
]


def available_detectors() -> List[str]:
    """List available detector backend names."""
    return ["retinaface"]


def available_variants(detector_name: str) -> List[str]:
    """List supported *named* variants for a detector backend.

    These are the variants that can be selected by name (often auto-downloaded)
    without providing an explicit weights path.
    """
    name = str(detector_name).lower().strip()
    if name == "retinaface":
        return available_retinaface_variants()
    raise ValueError(f"Unknown face detector backend: {detector_name!r}. Available: {available_detectors()}")


def get_face_detector(
    *,
    name: str = "retinaface",
    variant: str = "resnet50_2020-07-20",
    max_size: int = 1024,
    device: Optional[str] = None,
) -> FaceDetector:
    """Factory for face detector backends.

    Parameters
    ----------
    name:
        Backend name (e.g., "retinaface").
    variant:
        Backend-specific model variant string.
    max_size:
        Cap longer side for detectors that support it.
    device:
        Backend device string (best-effort; may fall back to CPU).
    """
    name = str(name).lower().strip()
    if name == "retinaface":
        return get_retinaface_detector(variant=variant, max_size=max_size, device=device)

    raise ValueError(f"Unknown face detector backend: {name!r}. Available: {available_detectors()}")