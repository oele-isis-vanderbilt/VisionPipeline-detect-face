from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class FaceResult:
    """Result object for face augmentation runs."""
    payload: Dict[str, Any]
    stats: Dict[str, Any]
    paths: Optional[Dict[str, str]] = None