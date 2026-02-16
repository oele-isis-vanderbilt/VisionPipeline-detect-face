from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np


# -------------------------
# JSON I/O
# -------------------------

def load_json(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: Dict[str, Any], out_path: str | Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, separators=(",", ":"))


# -------------------------
# Run folder logic (tracking-style)
# -------------------------

@dataclass(frozen=True)
class ArtifactPlan:
    enabled: bool
    run_dir: Optional[Path]
    json_path: Optional[Path]
    frames_dir: Optional[Path]
    video_path: Optional[Path]


def plan_artifacts(
    *,
    out_dir: str | Path = "out",
    run_name: Optional[str] = None,
    save_json_flag: bool = False,
    save_frames: bool = False,
    save_video: Optional[str | Path] = None,
) -> ArtifactPlan:
    """Create artifact plan without side effects unless enabled.

    Mirrors track-lib behavior: no directories created unless something is enabled.
    """
    enabled = bool(save_json_flag or save_frames or save_video)
    if not enabled:
        return ArtifactPlan(False, None, None, None, None)

    out_dir_p = Path(out_dir)
    run_dir = out_dir_p / run_name if run_name else out_dir_p
    # Create run_dir now since artifacts are enabled
    run_dir.mkdir(parents=True, exist_ok=True)

    json_path = run_dir / "faces.json" if save_json_flag else None
    frames_dir = run_dir / "frames" if save_frames else None
    if frames_dir is not None:
        frames_dir.mkdir(parents=True, exist_ok=True)

    video_path: Optional[Path] = None
    if save_video:
        video_path = run_dir / Path(save_video).name

    return ArtifactPlan(True, run_dir, json_path, frames_dir, video_path)


# -------------------------
# Video I/O
# -------------------------

class VideoReader:
    def __init__(self, path: str | Path) -> None:
        self.path = str(path)
        self.cap = cv2.VideoCapture(self.path)
        if not self.cap.isOpened():
            raise FileNotFoundError(f"Failed to open video: {self.path}")
        self._fps = float(self.cap.get(cv2.CAP_PROP_FPS) or 0.0) or 30.0
        self._w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        self._h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        self._n = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    @property
    def fps(self) -> float:
        return self._fps

    @property
    def size(self) -> Tuple[int, int]:
        return (self._w, self._h)

    @property
    def frame_count(self) -> int:
        return self._n

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        ok, frame = self.cap.read()
        if not ok:
            return False, None
        return True, frame

    def release(self) -> None:
        try:
            self.cap.release()
        except Exception:
            pass


class VideoWriter:
    def __init__(
        self,
        path: str | Path,
        *,
        fps: float,
        size: Tuple[int, int],
        fourcc: str = "mp4v",
    ) -> None:
        self.path = str(path)
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._fps = float(fps)
        self._size = (int(size[0]), int(size[1]))
        fourcc_code = cv2.VideoWriter_fourcc(*fourcc)
        self.writer = cv2.VideoWriter(self.path, fourcc_code, self._fps, self._size)
        if not self.writer.isOpened():
            raise RuntimeError(f"Failed to open VideoWriter: {self.path}")

    def write(self, frame: np.ndarray) -> None:
        if frame.shape[1] != self._size[0] or frame.shape[0] != self._size[1]:
            frame = cv2.resize(frame, self._size)
        self.writer.write(frame)

    def release(self) -> None:
        try:
            self.writer.release()
        except Exception:
            pass