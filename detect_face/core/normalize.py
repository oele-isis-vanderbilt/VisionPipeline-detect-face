from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from .schema import (
    K_BBOX,
    K_CLASS_ID,
    K_CLASS_NAME,
    K_DET_IND,
    K_FRAME_INDEX,
    K_KEYPOINTS,
)


@dataclass(frozen=True)
class NormalizedInput:
    payload: Dict[str, Any]
    schema_version: str                  # "det-v1" | "track-v1" | "unknown"
    frames: List[Dict[str, Any]]         # alias to payload["frames"]
    has_any_keypoints: bool              # any det has keypoints list


def detect_schema_version(payload: Dict[str, Any]) -> str:
    v = payload.get("schema_version")
    if isinstance(v, str):
        v = v.strip()
        if v in {"det-v1", "track-v1"}:
            return v
    return "unknown"


def _ensure_frame_index(frame: Dict[str, Any], fallback_index: int) -> int:
    if K_FRAME_INDEX in frame:
        try:
            return int(frame[K_FRAME_INDEX])
        except Exception:
            pass
    # common alternative keys
    if "frame" in frame:
        try:
            return int(frame["frame"])
        except Exception:
            pass
    if "frame_index" in frame:
        try:
            return int(frame["frame_index"])
        except Exception:
            pass
    frame[K_FRAME_INDEX] = int(fallback_index)
    return int(fallback_index)


def normalize_input_payload(payload: Dict[str, Any]) -> NormalizedInput:
    """Normalize a det-v1 or track-v1-like payload in-place.

    Requirements after normalization:
    - payload["frames"] is a list
    - each frame has integer frame_index (K_FRAME_INDEX)
    - each frame has a "detections" list (possibly empty)
    - each detection has bbox float[4] or we raise
    """
    if "frames" not in payload or not isinstance(payload["frames"], list):
        raise ValueError("Invalid JSON: missing root['frames'] list")

    frames: List[Dict[str, Any]] = payload["frames"]
    has_kpts = False

    for fi, fr in enumerate(frames):
        if not isinstance(fr, dict):
            raise ValueError(f"Invalid JSON: frames[{fi}] is not an object")

        _ensure_frame_index(fr, fi)

        dets = fr.get("detections")
        if not isinstance(dets, list):
            fr["detections"] = []
            dets = fr["detections"]

        for di, det in enumerate(dets):
            if not isinstance(det, dict):
                raise ValueError(f"Invalid JSON: frame {fi} det {di} is not an object")

            bbox = det.get(K_BBOX)
            if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
                raise ValueError(f"Frame {fi} det {di} missing/invalid bbox")
            det[K_BBOX] = [float(v) for v in bbox[:4]]

            # ensure det_ind exists if present upstream or derivable
            if K_DET_IND not in det:
                # if upstream had det_ind under another key, you can extend here later
                det[K_DET_IND] = int(di)

            # keypoints presence
            kpts = det.get(K_KEYPOINTS)
            if isinstance(kpts, list) and len(kpts) > 0:
                has_kpts = True

            # normalize optional class_id / class_name if present
            if K_CLASS_ID in det:
                try:
                    det[K_CLASS_ID] = int(det[K_CLASS_ID])
                except Exception:
                    pass
            if K_CLASS_NAME in det and det[K_CLASS_NAME] is not None:
                det[K_CLASS_NAME] = str(det[K_CLASS_NAME])

    return NormalizedInput(
        payload=payload,
        schema_version=detect_schema_version(payload),
        frames=frames,
        has_any_keypoints=has_kpts,
    )


def default_associate_class_ids(payload: Dict[str, Any]) -> Optional[List[int]]:
    """Heuristic default: if class names exist and include 'person', return its id(s).

    Returns:
      - list[int] if found
      - None if no safe default (caller can decide to associate all or warn)
    """
    frames = payload.get("frames")
    if not isinstance(frames, list):
        return None

    found: List[int] = []
    for fr in frames:
        dets = fr.get("detections") if isinstance(fr, dict) else None
        if not isinstance(dets, list):
            continue
        for det in dets:
            if not isinstance(det, dict):
                continue
            cname = det.get(K_CLASS_NAME)
            cid = det.get(K_CLASS_ID)
            if isinstance(cname, str) and cname.lower() == "person" and isinstance(cid, (int, float, str)):
                try:
                    found.append(int(cid))
                except Exception:
                    pass

    if not found:
        return None
    # unique preserve order
    out: List[int] = []
    seen = set()
    for x in found:
        if x not in seen:
            out.append(x)
            seen.add(x)
    return out


DeviceSpec = Union[str, int, None]


def resolve_auto_device() -> str:
    """Resolve the best available compute device.

    Priority:
      1) CUDA (cuda:0)
      2) MPS (mps)
      3) CPU (cpu)

    Notes
    -----
    - Uses a soft import of torch. If torch is unavailable, returns "cpu".
    """
    try:
        import torch  # type: ignore

        if torch.cuda.is_available():
            return "cuda:0"
        # MPS is available on macOS builds that include it
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():  # type: ignore[attr-defined]
            return "mps"
        return "cpu"
    except Exception:
        return "cpu"


def normalize_device(device: DeviceSpec, *, default: str = "auto") -> str:
    """Normalize device spec into a canonical string.

    Accepted inputs
    ---------------
    - None -> `default` ("auto" by default)
    - "auto" -> resolved best device (cuda:0 / mps / cpu)
    - "cpu" / "mps" / "cuda" / "cuda:0" / "cuda:1" ...
    - integer or numeric string ("0", "1", ...) -> "cuda:<idx>"

    Returns
    -------
    str
        Canonical device string.
    """
    if device is None:
        device = default

    # ints map to cuda:<idx>
    if isinstance(device, int):
        return f"cuda:{device}"

    s = str(device).strip().lower()
    if s == "":
        s = str(default).strip().lower()

    if s == "auto":
        return resolve_auto_device()

    # numeric string -> cuda:<idx>
    if s.isdigit():
        return f"cuda:{int(s)}"

    # common aliases
    if s == "gpu":
        s = "cuda"

    if s == "cuda":
        return "cuda:0"

    # pass through cuda:<idx>, cpu, mps, etc.
    return s