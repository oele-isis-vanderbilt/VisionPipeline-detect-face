from __future__ import annotations

import math
import time
import warnings
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import cv2
import numpy as np

from ..detectors import get_face_detector
from ..detectors.base import FaceDetection, as_minimal_face_dict
from ..viz.draw import draw_frame
from .io import VideoReader, VideoWriter, load_json, plan_artifacts, save_json
from .normalize import NormalizedInput, default_associate_class_ids, normalize_device, normalize_input_payload
from .result import FaceResult
from .schema import (
    K_BBOX,
    K_CLASS_ID,
    K_CLASS_NAME,
    K_DET_IND,
    K_FACES,
    K_FRAME_INDEX,
    K_GALLERY_ID,
    K_KEYPOINTS,
    K_SCORE,
    K_TRACK_ID,
    bbox_contains,
    bbox_iou_matrix,
    ensure_xyxy,
    kpts_inside_box,
)

__all__ = ["detect_faces_video"]


# -------------------------
# Assignment helpers
# -------------------------

def _eligible_person_class(det: Dict[str, Any], associate_class_ids: Optional[Set[int]]) -> bool:
    if not associate_class_ids:
        return True
    cid = det.get(K_CLASS_ID)
    try:
        cid_i = int(cid)
    except Exception:
        return False
    return cid_i in associate_class_ids


def _eligible_person_gallery(det: Dict[str, Any], *, gallery_filter: bool) -> bool:
    if not gallery_filter:
        return True
    tid = det.get(K_TRACK_ID)
    gid = det.get(K_GALLERY_ID)
    if tid is None or gid is None:
        return False
    return str(tid) == str(gid)


class _Workspace:
    """Reusable buffers for per-frame assignment."""
    def __init__(self) -> None:
        self.F_cap = 0
        self.P_cap = 0
        self.f_arr: Optional[np.ndarray] = None  # (F,4)
        self.p_arr: Optional[np.ndarray] = None  # (P,4)
        self.iou: Optional[np.ndarray] = None    # (F,P)
        self.cost: Optional[np.ndarray] = None   # (F,P)
        self.cand: Optional[np.ndarray] = None   # (F,P) bool

    def ensure(self, F: int, P: int) -> "_Workspace":
        if F > self.F_cap or P > self.P_cap or self.f_arr is None:
            self.F_cap = max(self.F_cap, F)
            self.P_cap = max(self.P_cap, P)
            self.f_arr = np.empty((self.F_cap, 4), dtype=np.float32)
            self.p_arr = np.empty((self.P_cap, 4), dtype=np.float32)
            self.iou = np.empty((self.F_cap, self.P_cap), dtype=np.float32)
            self.cost = np.empty((self.F_cap, self.P_cap), dtype=np.float32)
            self.cand = np.empty((self.F_cap, self.P_cap), dtype=bool)
        return self


_WS = _Workspace()


def _hungarian_min_cost(C: np.ndarray) -> List[Tuple[int, int]]:
    """Min-cost assignment. SciPy if available, else deterministic greedy."""
    try:
        from scipy.optimize import linear_sum_assignment  # type: ignore

        r_ind, c_ind = linear_sum_assignment(C)
        out: List[Tuple[int, int]] = []
        for r, c in zip(r_ind.tolist(), c_ind.tolist()):
            if math.isfinite(float(C[r, c])):
                out.append((r, c))
        return out
    except Exception:
        pass

    # Greedy fallback
    F, P = C.shape
    candidates: List[Tuple[float, int, int]] = []
    for r in range(F):
        for c in range(P):
            v = float(C[r, c])
            if math.isfinite(v):
                candidates.append((v, r, c))
    candidates.sort(key=lambda t: t[0])  # low cost = high IoU
    used_r = set()
    used_c = set()
    out: List[Tuple[int, int]] = []
    for v, r, c in candidates:
        if r in used_r or c in used_c:
            continue
        out.append((r, c))
        used_r.add(r)
        used_c.add(c)
    return out


def _assign_faces_to_persons(
    persons: List[Dict[str, Any]],
    faces: List[FaceDetection],
    *,
    iou_thresh: float,
    containment: bool,
    associate_class_ids: Optional[Set[int]],
    gallery_filter: bool,
    kpt_indices: Optional[Sequence[int]],
    kpt_conf: float,
    inclusive: bool,
    flexible_kpt: Optional[int],
) -> List[Tuple[int, int]]:
    """Return list of (face_index, person_index) assignments.

    Optimization: compute IoU/cost only against eligible persons (class + gallery filter).
    """
    F = len(faces)
    P = len(persons)
    if F == 0 or P == 0:
        return []

    # Collect eligible person indices first
    elig_idx: List[int] = []
    for j, pd in enumerate(persons):
        if not _eligible_person_class(pd, associate_class_ids):
            continue
        if not _eligible_person_gallery(pd, gallery_filter=gallery_filter):
            continue
        elig_idx.append(j)

    if not elig_idx:
        return []

    P2 = len(elig_idx)
    ws = _WS.ensure(F, P2)
    assert ws.f_arr is not None and ws.p_arr is not None and ws.iou is not None and ws.cost is not None and ws.cand is not None

    f_arr = ws.f_arr[:F]
    p_arr = ws.p_arr[:P2]
    iou = ws.iou[:F, :P2]
    cost = ws.cost[:F, :P2]
    cand = ws.cand[:F, :P2]

    # Fill face boxes
    for i, fd in enumerate(faces):
        x1, y1, x2, y2 = ensure_xyxy(fd.bbox)
        f_arr[i, 0] = x1
        f_arr[i, 1] = y1
        f_arr[i, 2] = x2
        f_arr[i, 3] = y2

    # Fill eligible person boxes
    for jj, j in enumerate(elig_idx):
        pd = persons[j]
        x1, y1, x2, y2 = ensure_xyxy(pd.get(K_BBOX, (0, 0, 0, 0)))
        p_arr[jj, 0] = x1
        p_arr[jj, 1] = y1
        p_arr[jj, 2] = x2
        p_arr[jj, 3] = y2

    # Quick AABB overlap mask (cheap rejection before IoU)
    px1, py1, px2, py2 = p_arr.T
    fx1, fy1, fx2, fy2 = f_arr.T
    no_x = (fx2[:, None] <= px1[None, :]) | (px2[None, :] <= fx1[:, None])
    no_y = (fy2[:, None] <= py1[None, :]) | (py2[None, :] <= fy1[:, None])
    np.logical_not(np.logical_or(no_x, no_y), out=cand)

    # IoU only for eligible persons
    iou[:, :] = bbox_iou_matrix(f_arr, p_arr)

    # Candidate mask: AABB overlap AND IoU>=thresh
    np.logical_and(cand, iou >= float(iou_thresh), out=cand)

    # Fill costs
    cost.fill(np.inf)
    np.subtract(1.0, iou, out=cost, where=cand)

    # Hard rules (containment / keypoints) only for candidate pairs
    if containment or (kpt_indices and len(kpt_indices) > 0):
        for fi in range(F):
            cols = np.nonzero(cand[fi])[0]
            if cols.size == 0:
                continue
            fbox = ensure_xyxy(faces[fi].bbox)
            for jj in cols.tolist():
                j = elig_idx[jj]
                pdet = persons[j]
                pbox = ensure_xyxy(pdet.get(K_BBOX, (0, 0, 0, 0)))

                if containment and not bbox_contains(pbox, fbox, inclusive=inclusive):
                    cost[fi, jj] = np.inf
                    continue

                if kpt_indices and len(kpt_indices) > 0:
                    ok = kpts_inside_box(
                        pdet,
                        fbox,
                        kpt_indices,
                        conf_thresh=float(kpt_conf),
                        inclusive=bool(inclusive),
                        flexible_kpt=flexible_kpt,
                    )
                    if not ok:
                        cost[fi, jj] = np.inf

    matches_small = _hungarian_min_cost(cost.astype(float, copy=True))

    # Map back to original person indices
    matches: List[Tuple[int, int]] = []
    for fi, jj in matches_small:
        if 0 <= jj < P2:
            matches.append((fi, elig_idx[jj]))
    return matches


# -------------------------
# Public API
# -------------------------

def detect_faces_video(
    *,
    json_in: str | Path,
    video: str | Path,
    detector: str = "retinaface",
    variant: str = "resnet50_2020-07-20",
    max_size: int = 1024,
    device: str = "auto",
    conf_thresh: float = 0.5,
    nms_thresh: float = 0.3,
    iou_thresh: float = 0.0,
    containment: bool = True,
    inclusive: bool = True,
    associate_class_ids: Optional[Sequence[int]] = None,
    gallery_filter: bool = False,
    kpt_indices: Optional[Sequence[int]] = None,
    kpt_conf: float = 0.3,
    flexible_kpt: Optional[int] = None,
    # artifacts (all off by default)
    save_json_flag: bool = False,
    save_frames: bool = False,
    save_video: Optional[str | Path] = None,
    out_dir: str | Path = "out",
    run_name: Optional[str] = None,
    fourcc: str = "mp4v",
    display: bool = False,
    no_progress: bool = False,
) -> FaceResult:
    """Augment input JSON (det-v1 or track-v1) with face detections assigned to persons.

    Behavior:
    - Accepts det-v1 or track-v1 (or compatible) JSON.
    - Adds `faces` array to each eligible person detection.
    - Adds root-level `face_augment` metadata.
    - Writes no files unless save flags are enabled.
    """
    payload = load_json(json_in)
    norm: NormalizedInput = normalize_input_payload(payload)

    # Default class association heuristic if not provided
    if associate_class_ids is None:
        associate_class_ids = default_associate_class_ids(norm.payload)

    # Precompute associate_class_set for fast membership tests
    associate_class_set: Optional[Set[int]] = None
    if associate_class_ids:
        try:
            associate_class_set = set(int(x) for x in associate_class_ids)
        except Exception:
            associate_class_set = None

    # Keypoint filtering: warn & disable if requested but no keypoints exist
    if kpt_indices and len(kpt_indices) > 0 and not norm.has_any_keypoints:
        warnings.warn(
            "Keypoint filtering was requested (kpt_indices) but no keypoints exist in the input JSON. "
            "Skipping keypoint constraints for this run.",
            RuntimeWarning,
        )
        kpt_indices = None

    # Plan artifacts (no side-effects unless enabled)
    plan = plan_artifacts(
        out_dir=out_dir,
        run_name=run_name,
        save_json_flag=save_json_flag,
        save_frames=save_frames,
        save_video=save_video,
    )

    # Detector
    device_norm = normalize_device(device)
    det = get_face_detector(name=detector, variant=variant, max_size=int(max_size), device=device_norm)

    # Video I/O
    reader = VideoReader(video)
    W, H = reader.size
    writer: Optional[VideoWriter] = None
    if plan.video_path is not None:
        writer = VideoWriter(plan.video_path, fps=reader.fps, size=(W, H), fourcc=fourcc)

    # Progress
    frames = norm.frames
    if not no_progress:
        try:
            from tqdm import trange  # type: ignore
            it = trange(len(frames), desc="detect-face")
        except Exception:
            it = range(len(frames))  # type: ignore
    else:
        it = range(len(frames))  # type: ignore

    t0 = time.time()
    processed = 0
    faces_total = 0
    assigned_total = 0
    frames_with_faces = 0
    frames_with_assignments = 0

    for idx in it:  # type: ignore
        ok, frame = reader.read()
        if not ok or frame is None:
            break
        processed += 1

        fr = frames[idx]
        frame_index = int(fr.get(K_FRAME_INDEX, idx))
        persons: List[Dict[str, Any]] = fr.get("detections", [])

        # Run detector over full frame
        faces: List[FaceDetection] = det.predict(
            frame,
            conf_thresh=float(conf_thresh),
            nms_thresh=float(nms_thresh),
        )
        faces_total += len(faces)
        if faces:
            frames_with_faces += 1

        # Clear existing faces (avoid stale duplicates)
        for p in persons:
            if isinstance(p, dict):
                p.pop(K_FACES, None)

        # Assign and attach
        matches = _assign_faces_to_persons(
            persons,
            faces,
            iou_thresh=float(iou_thresh),
            containment=bool(containment),
            associate_class_ids=associate_class_set,
            gallery_filter=bool(gallery_filter),
            kpt_indices=list(kpt_indices) if kpt_indices else None,
            kpt_conf=float(kpt_conf),
            inclusive=bool(inclusive),
            flexible_kpt=flexible_kpt,
        )
        assigned_total += len(matches)
        if matches:
            frames_with_assignments += 1

        # Attach minimal faces per matched person
        # Ensure per-person face_ind increments in the order they are attached
        for fi, pj in matches:
            if not (0 <= pj < len(persons)):
                continue
            pdet = persons[pj]
            arr = pdet.setdefault(K_FACES, [])
            face_dict = as_minimal_face_dict(faces[fi])
            face_dict["face_ind"] = len(arr)
            arr.append(face_dict)

        # Viz / artifacts
        if display or writer is not None or plan.frames_dir is not None:
            vis = frame.copy()
            vis = draw_frame(vis, persons, show_conf=True)

            if display:
                cv2.imshow("detect-face", vis)
                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # ESC
                    display = False

            if writer is not None:
                writer.write(vis)

            if plan.frames_dir is not None:
                outp = plan.frames_dir / f"frame_{frame_index:06d}.jpg"
                cv2.imwrite(str(outp), vis)

    # Cleanup
    if writer is not None:
        writer.release()
    reader.release()
    if display:
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass

    # Metadata injection (minimal + stable)
    face_meta = {
        "version": "face-v1",
        "parent_schema_version": norm.payload.get("schema_version", norm.schema_version),
        "detector": {
            "name": str(detector),
            "variant": str(variant),
            "max_size": int(max_size),
            "conf_thresh": float(conf_thresh),
            "nms_thresh": float(nms_thresh),
            "device": device_norm,
        },
        "association": {
            "associate_class_ids": list(associate_class_ids) if associate_class_ids else None,
            "gallery_filter": bool(gallery_filter),
            "iou_thresh": float(iou_thresh),
            "containment": bool(containment),
            "inclusive": bool(inclusive),
            "kpt_indices": list(kpt_indices) if kpt_indices else None,
            "kpt_conf": float(kpt_conf),
            "flexible_kpt": int(flexible_kpt) if flexible_kpt is not None else None,
        },
    }
    frames = norm.payload.pop("frames", None)
    norm.payload["face_augment"] = face_meta
    if frames is not None:
        norm.payload["frames"] = frames

    # Save JSON only if enabled
    paths: Optional[Dict[str, str]] = None
    if plan.json_path is not None:
        save_json(norm.payload, plan.json_path)
        paths = paths or {}
        paths["json"] = str(plan.json_path)

    if plan.frames_dir is not None:
        paths = paths or {}
        paths["frames_dir"] = str(plan.frames_dir)

    if plan.video_path is not None:
        paths = paths or {}
        paths["video"] = str(plan.video_path)

    dt = max(1e-6, time.time() - t0)
    stats = {
        "frames": int(processed),
        "faces_total": int(faces_total),
        "assigned_total": int(assigned_total),
        "frames_with_faces": int(frames_with_faces),
        "frames_with_assignments": int(frames_with_assignments),
        "fps": float(processed / dt),
    }

    return FaceResult(payload=norm.payload, stats=stats, paths=paths)