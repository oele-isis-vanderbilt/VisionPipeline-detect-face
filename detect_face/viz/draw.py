from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from ..core.schema import (
    K_BBOX,
    K_CLASS_NAME,
    K_FACES,
    K_GALLERY_ID,
    K_KEYPOINTS,
    K_LANDMARKS,
    K_SCORE,
    K_TRACK_ID,
    K_SEG,
)

# -------------------------
# Stable color helpers (gtrack-style)
# -------------------------

def _hash_to_color(key: str | int, s: float = 0.9, v: float = 0.95) -> Tuple[int, int, int]:
    """Stable BGR color from an id using HSV hashing (SHA1-based)."""
    import hashlib

    if not isinstance(key, str):
        key = str(key)
    h = int(hashlib.sha1(key.encode("utf-8")).hexdigest(), 16)
    hue = (h % 360) / 360.0

    i = int(hue * 6)
    f = hue * 6 - i
    p = v * (1 - s)
    q = v * (1 - f * s)
    t = v * (1 - (1 - f) * s)
    i = i % 6
    if i == 0:
        r, g, b = v, t, p
    elif i == 1:
        r, g, b = q, v, p
    elif i == 2:
        r, g, b = p, v, t
    elif i == 3:
        r, g, b = p, q, v
    elif i == 4:
        r, g, b = t, p, v
    else:
        r, g, b = v, p, q
    return int(b * 255), int(g * 255), int(r * 255)


def _legible_text_color(bgr: Tuple[int, int, int]) -> Tuple[int, int, int]:
    b, g, r = bgr
    y = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return (0, 0, 0) if y > 160 else (255, 255, 255)


# -------------------------
# Drawing primitives
# -------------------------

def draw_box_label(
    img: np.ndarray,
    xyxy: Sequence[float],
    color: Tuple[int, int, int],
    text: Optional[str] = None,
    thickness: int = 2,
) -> None:
    x1, y1, x2, y2 = [int(round(v)) for v in xyxy[:4]]
    x1, y1 = max(0, x1), max(0, y1)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)
    if text:
        tf = max(thickness - 1, 1)
        ts = 0.5 + 0.1 * (thickness - 2)
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, ts, tf)
        th = th + 4
        cv2.rectangle(img, (x1, y1 - th), (x1 + tw + 6, y1), color, -1)
        cv2.putText(
            img,
            text,
            (x1 + 3, y1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            ts,
            _legible_text_color(color),
            tf,
            cv2.LINE_AA,
        )


def draw_keypoints(
    img: np.ndarray,
    kpts: Sequence[Sequence[float]],
    color: Tuple[int, int, int],
    radius: int = 3,
) -> None:
    for item in kpts:
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            continue
        x, y = int(item[0]), int(item[1])
        cv2.circle(img, (x, y), radius, color, -1, cv2.LINE_AA)


def draw_face_landmarks(
    img: np.ndarray,
    lms: Sequence[Sequence[float]],
    color: Tuple[int, int, int],
    size: int = 3,
    thickness: int = 1,
) -> None:
    """Draw face landmarks as small crosshairs (distinct from body kpts)."""
    for item in lms:
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            continue
        x, y = int(item[0]), int(item[1])
        cv2.line(img, (x - size, y), (x + size, y), color, thickness, cv2.LINE_AA)
        cv2.line(img, (x, y - size), (x, y + size), color, thickness, cv2.LINE_AA)


def draw_polygon(
    img: np.ndarray,
    points: Sequence[Sequence[float]] | None,
    color: Tuple[int, int, int],
    alpha: float = 0.2,
    thickness: int = 2,
) -> None:
    if not points:
        return
    poly = np.asarray(points, dtype=np.int32).reshape(-1, 1, 2)
    overlay = img.copy()
    cv2.fillPoly(overlay, [poly], color)
    cv2.addWeighted(overlay, float(alpha), img, 1 - float(alpha), 0, dst=img)
    cv2.polylines(img, [poly], isClosed=True, color=color, thickness=thickness, lineType=cv2.LINE_AA)


# -------------------------
# High-level draw
# -------------------------

def draw_frame(
    img: np.ndarray,
    persons: List[Dict[str, Any]],
    *,
    show_conf: bool = True,
) -> np.ndarray:
    out = img
    for det in persons:
        sid = str(det.get(K_GALLERY_ID) or det.get(K_TRACK_ID) or "?")
        base_color = _hash_to_color(sid)
        face_color = base_color

        bbox = det.get(K_BBOX)
        if bbox is not None and isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
            label = f"ID:{sid}"
            if show_conf and det.get(K_SCORE) is not None:
                try:
                    label += f" {float(det[K_SCORE]):.2f}"
                except Exception:
                    pass
            cname = det.get(K_CLASS_NAME)
            if cname:
                label += f" {cname}"
            draw_box_label(out, bbox, base_color, label)

        kps = det.get(K_KEYPOINTS)
        if kps:
            draw_keypoints(out, kps, base_color, radius=3)

        seg = det.get(K_SEG)
        if seg:
            draw_polygon(out, seg, base_color, alpha=0.2, thickness=2)

        faces = det.get(K_FACES) or []
        for fd in faces:
            fb = fd.get(K_BBOX)
            if fb is None or not isinstance(fb, (list, tuple)) or len(fb) < 4:
                continue
            flabel = "face"
            if show_conf and fd.get(K_SCORE) is not None:
                try:
                    flabel = f"face {float(fd[K_SCORE]):.2f}"
                except Exception:
                    pass
            draw_box_label(out, fb, face_color, flabel, thickness=2)
            lms = fd.get(K_LANDMARKS)
            if lms:
                draw_face_landmarks(out, lms, face_color, size=3, thickness=1)

    return out