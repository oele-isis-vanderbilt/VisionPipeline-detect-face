from __future__ import annotations

"""detect_face: augment det-v1 / track-v1 JSON with face detections.

PyPI package: detect-face-lib
Module name: detect_face
detect_face/
├── __init__.py
├── cli/
│   ├── __init__.py
│   ├── __main__.py
│   └── detect_faces.py
├── core/
│   ├── __init__.py
│   ├── io.py
│   ├── normalize.py
│   ├── pipeline.py
│   ├── result.py
│   └── schema.py
├── detectors/
│   ├── __init__.py
│   ├── base.py
│   └── retinaface.py
└── viz/
    ├── __init__.py
    └── draw.py
"""

from .core.pipeline import detect_faces_video
from .detectors import get_face_detector, available_detectors

__all__ = [
    "detect_faces_video",
    "get_face_detector",
    "available_detectors",
]