from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, TypedDict

import numpy as np

# -------------------------
# Keys (aligned with your pipeline conventions)
# -------------------------

K_BBOX = "bbox"
K_SCORE = "score"
K_CLASS_ID = "class_id"
K_CLASS_NAME = "class_name"
K_DET_IND = "det_ind"
K_TRACK_ID = "track_id"
K_GALLERY_ID = "gallery_id"
K_KEYPOINTS = "keypoints"          # body kpts: list[[x,y,score], ...]
K_SEG = "segments"                 # optional segmentation field name
K_FRAME_INDEX = "frame_index"

# Face augmentation keys
K_FACES = "faces"                  # container for faces attached to a person det
K_LANDMARKS = "landmarks"          # face 5-point landmarks

__all__ = [
    # constants
    "K_BBOX",
    "K_SCORE",
    "K_CLASS_ID",
    "K_CLASS_NAME",
    "K_DET_IND",
    "K_TRACK_ID",
    "K_GALLERY_ID",
    "K_KEYPOINTS",
    "K_SEG",
    "K_FRAME_INDEX",
    "K_FACES",
    "K_LANDMARKS",
    # types
    "BBox",
    "Point",
    "KptTriplet",
    "FaceDet",
    "PersonDet",
    # geometry
    "ensure_xyxy",
    "bbox_area",
    "bbox_iou",
    "bbox_iou_matrix",
    "bbox_contains",
    "point_in_box",
    "clip_box_to_image",
    "expand_box",
    # keypoints
    "normalize_kpt_triplet",
    "kpts_inside_box",
]

# -------------------------
# Types
# -------------------------

BBox = Tuple[float, float, float, float]  # (x1, y1, x2, y2)
Point = Tuple[float, float]
KptTriplet = Tuple[float, float, float]   # (x, y, conf)


class FaceDet(TypedDict, total=False):
    bbox: List[float]
    score: float
    landmarks: List[List[float]]  # 5x2 [[x,y], ...]
    face_ind: int                 # per-person running index


class PersonDet(TypedDict, total=False):
    bbox: List[float]
    score: float
    class_id: int
    class_name: str
    det_ind: int
    track_id: str
    gallery_id: str
    keypoints: List[List[float]]  # Nx3 [x,y,score]
    segments: Any
    faces: List[FaceDet]


# -------------------------
# Geometry helpers
# -------------------------

def ensure_xyxy(b: Sequence[float]) -> BBox:
    """Return a valid (x1,y1,x2,y2) with sorted corners."""
    x1, y1, x2, y2 = float(b[0]), float(b[1]), float(b[2]), float(b[3])
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return (x1, y1, x2, y2)


def bbox_area(b: Sequence[float]) -> float:
    x1, y1, x2, y2 = ensure_xyxy(b)
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def bbox_iou(a: Sequence[float], b: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = ensure_xyxy(a)
    bx1, by1, bx2, by2 = ensure_xyxy(b)
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    ua = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    ub = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = ua + ub - inter
    if union <= 0.0:
        return 0.0
    return inter / union


def bbox_iou_matrix(a_boxes: np.ndarray, b_boxes: np.ndarray) -> np.ndarray:
    """Pairwise IoU between arrays of boxes (Na,4) and (Nb,4). Returns (Na,Nb) float32."""
    a = np.asarray(a_boxes, dtype=np.float32)
    b = np.asarray(b_boxes, dtype=np.float32)
    if a.size == 0 or b.size == 0:
        return np.zeros((a.shape[0], b.shape[0]), dtype=np.float32)
    if a.ndim != 2 or a.shape[1] != 4 or b.ndim != 2 or b.shape[1] != 4:
        raise ValueError("bbox_iou_matrix expects inputs of shape (N,4) and (M,4)")

    a_x1 = np.minimum(a[:, 0], a[:, 2]); a_y1 = np.minimum(a[:, 1], a[:, 3])
    a_x2 = np.maximum(a[:, 0], a[:, 2]); a_y2 = np.maximum(a[:, 1], a[:, 3])
    b_x1 = np.minimum(b[:, 0], b[:, 2]); b_y1 = np.minimum(b[:, 1], b[:, 3])
    b_x2 = np.maximum(b[:, 0], b[:, 2]); b_y2 = np.maximum(b[:, 1], b[:, 3])

    a_w = np.clip(a_x2 - a_x1, 0.0, None); a_h = np.clip(a_y2 - a_y1, 0.0, None)
    b_w = np.clip(b_x2 - b_x1, 0.0, None); b_h = np.clip(b_y2 - b_y1, 0.0, None)
    a_area = a_w * a_h
    b_area = b_w * b_h

    ix1 = np.maximum(a_x1[:, None], b_x1[None, :])
    iy1 = np.maximum(a_y1[:, None], b_y1[None, :])
    ix2 = np.minimum(a_x2[:, None], b_x2[None, :])
    iy2 = np.minimum(a_y2[:, None], b_y2[None, :])
    iw = np.clip(ix2 - ix1, 0.0, None)
    ih = np.clip(iy2 - iy1, 0.0, None)
    inter = iw * ih

    union = a_area[:, None] + b_area[None, :] - inter
    return np.where(union > 0.0, inter / union, 0.0).astype(np.float32)


def bbox_contains(outer: Sequence[float], inner: Sequence[float], *, inclusive: bool = True) -> bool:
    ox1, oy1, ox2, oy2 = ensure_xyxy(outer)
    ix1, iy1, ix2, iy2 = ensure_xyxy(inner)
    if inclusive:
        return (ix1 >= ox1) and (iy1 >= oy1) and (ix2 <= ox2) and (iy2 <= oy2)
    return (ix1 > ox1) and (iy1 > oy1) and (ix2 < ox2) and (iy2 < oy2)


def point_in_box(pt: Sequence[float], box: Sequence[float], *, inclusive: bool = True) -> bool:
    x, y = float(pt[0]), float(pt[1])
    x1, y1, x2, y2 = ensure_xyxy(box)
    if inclusive:
        return (x1 <= x <= x2) and (y1 <= y <= y2)
    return (x1 < x < x2) and (y1 < y < y2)


def clip_box_to_image(box: Sequence[float], size: Tuple[int, int]) -> BBox:
    x1, y1, x2, y2 = ensure_xyxy(box)
    W, H = int(size[0]), int(size[1])
    x1 = min(max(0.0, x1), float(W))
    x2 = min(max(0.0, x2), float(W))
    y1 = min(max(0.0, y1), float(H))
    y2 = min(max(0.0, y2), float(H))
    return (x1, y1, x2, y2)


def expand_box(box: Sequence[float], *, scale: float = 1.0) -> BBox:
    x1, y1, x2, y2 = ensure_xyxy(box)
    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)
    w = max(0.0, x2 - x1)
    h = max(0.0, y2 - y1)
    w2 = 0.5 * w * float(scale)
    h2 = 0.5 * h * float(scale)
    return (cx - w2, cy - h2, cx + w2, cy + h2)


# -------------------------
# Keypoint helpers
# -------------------------

def normalize_kpt_triplet(item: Sequence[float]) -> Optional[KptTriplet]:
    if item is None or len(item) < 2:
        return None
    x = float(item[0])
    y = float(item[1])
    c = float(item[2]) if len(item) >= 3 else 1.0
    return (x, y, c)


def kpts_inside_box(
    person: Mapping[str, Any],
    face_box: Sequence[float],
    indices: Sequence[int],
    *,
    conf_thresh: float = 0.3,
    inclusive: bool = True,
    flexible_kpt: Optional[int] = None,
) -> bool:
    """Return True if keypoint constraints pass.

    - If no keypoints exist: pass (caller should warn/disable if desired).
    - Only keypoints at requested indices with conf>=conf_thresh are considered.
    - If none survive: pass.
    - Default: require all surviving inside box.
    - flexible_kpt=N: if >=N survive -> require >=N inside; else require all inside.
    """
    kpts = person.get(K_KEYPOINTS)
    if not isinstance(kpts, Sequence):
        return True

    cache: Dict[int, KptTriplet] = {}
    for idx, item in enumerate(kpts):
        trip = normalize_kpt_triplet(item)  # type: ignore[arg-type]
        if trip is not None:
            cache[idx] = trip

    surviving: List[int] = []
    for idx in indices:
        if not isinstance(idx, int):
            continue
        trip = cache.get(idx)
        if trip is None:
            continue
        x, y, c = trip
        if c >= float(conf_thresh):
            surviving.append(idx)

    if not surviving:
        return True

    inside_count = 0
    for idx in surviving:
        x, y, _ = cache[idx]
        if point_in_box((x, y), face_box, inclusive=inclusive):
            inside_count += 1

    total_surviving = len(surviving)

    if not flexible_kpt or flexible_kpt <= 0:
        return inside_count == total_surviving

    if total_surviving >= flexible_kpt:
        return inside_count >= flexible_kpt
    return inside_count == total_surviving