from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence


class _Fmt(argparse.ArgumentDefaultsHelpFormatter, argparse.RawDescriptionHelpFormatter):
    """Argparse formatter: preserves newlines AND shows defaults."""

from ..detectors import available_detectors, available_variants
from ..core.pipeline import detect_faces_video


def _parse_int_list(s: str) -> list[int]:
    # supports "0,1,2" or "0;1;2" or "0 1 2"
    parts = []
    for chunk in s.replace(";", ",").replace(" ", ",").split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        parts.append(int(chunk))
    return parts


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        "detect_faces",
        formatter_class=_Fmt,
        description=(
            "Augment det-v1 / track-v1 JSON with face detections and attach them to person detections.\n\n"
            "Input:\n"
            "  - --json-in: det-v1 (no tracking) OR track-v1 (with track_id/gallery_id).\n"
            "  - --video:   source video used to produce the JSON (frames must align).\n\n"
            "By default no files are written (in-memory only). Enable artifacts with --json / --frames / --save-video."
        ),
        epilog=(
            "Examples:\n"
            "  # List backends / variants\n"
            "  python -m detect_face.cli.detect_faces --list-detectors\n"
            "  python -m detect_face.cli.detect_faces --detector retinaface --list-variants\n\n"
            "  # Minimal run (no outputs written)\n"
            "  python -m detect_face.cli.detect_faces --json-in tracked.json --video in.mp4\n\n"
            "  # Save JSON only (tracking-style run folder)\n"
            "  python -m detect_face.cli.detect_faces --json-in tracked.json --video in.mp4 \\\n"
            "    --json --out-dir out --run-name demo\n\n"
            "  # Restrict face association to class_id=0 (commonly 'person')\n"
            "  python -m detect_face.cli.detect_faces --json-in detections.json --video in.mp4 \\\n"
            "    --associate-classes 0 --json --out-dir out --run-name cls0\n\n"
            "  # Full artifacts\n"
            "  python -m detect_face.cli.detect_faces --json-in tracked.json --video in.mp4 \\\n"
            "    --json --frames --save-video annotated.mp4 --out-dir out --run-name full\n"
        ),
    )

    # Inputs
    p.add_argument("--json-in", required=False, help="Input det-v1 or track-v1 JSON")
    p.add_argument("--video", required=False, help="Source video path (same as used for detection/tracking)")

    # Discovery
    p.add_argument("--list-detectors", action="store_true", help="List available face detector backends and exit")
    p.add_argument(
        "--list-variants",
        action="store_true",
        help="List supported named variants for --detector and exit",
    )

    # Detector
    p.add_argument("--detector", default="retinaface", help="Face detector backend name")
    p.add_argument("--variant", default="resnet50_2020-07-20", help="Backend model variant (if applicable)")
    p.add_argument(
        "--max-size",
        type=int,
        default=1024,
        help=(
            "Cap the longer image side for the face detector input. "
            "Lower = faster but may miss small faces; higher = slower but more accurate on small faces."
        ),
    )
    p.add_argument(
        "--device",
        default="auto",
        help=(
            "Compute device. Supported: auto (default), cpu, mps, cuda, cuda:0, cuda:1, or a CUDA index like 0/1. "
            "auto resolves to cuda:0 if available, else mps, else cpu."
        ),
    )

    p.add_argument(
        "--conf-thresh",
        type=float,
        default=0.5,
        help=(
            "Minimum face confidence. Increase to reduce false positives (fewer faces); "
            "decrease to detect more faces (more false positives)."
        ),
    )
    p.add_argument(
        "--nms-thresh",
        type=float,
        default=0.3,
        help=(
            "Non-maximum suppression threshold inside the face detector. "
            "Lower = more aggressive suppression (fewer overlapping faces)."
        ),
    )

    # Association rules
    p.add_argument(
        "--iou-thresh",
        type=float,
        default=0.0,
        help=(
            "IoU threshold for person-face pairing candidates. "
            "0.0 considers any overlap; 0.1-0.3 is stricter and can reduce wrong assignments."
        ),
    )
    p.add_argument(
        "--containment",
        dest="containment",
        action="store_true",
        help="Require the face box to lie fully inside the person box (safer, fewer bad matches).",
    )
    p.add_argument(
        "--no-containment",
        dest="containment",
        action="store_false",
        help="Disable containment (more permissive; may increase mismatches on crowded frames).",
    )
    p.set_defaults(containment=True)

    p.add_argument(
        "--associate-classes",
        default=None,
        help=(
            "Class IDs eligible for face association (comma/semicolon/space separated). "
            "Example: '0' (person). If omitted, tries to auto-detect class_name=='person'; "
            "if not found, all classes are eligible."
        ),
    )
    p.add_argument(
        "--gallery-filter",
        action="store_true",
        help=(
            "Track-v1 only: only assign faces to detections where track_id == gallery_id (when both exist). "
            "Useful when you want faces only for confirmed identities."
        ),
    )

    p.add_argument(
        "--kpt-indices",
        default=None,
        help=(
            "Body keypoint indices to require inside the face box (comma/semicolon/space separated). "
            "Only used if input JSON has keypoints; otherwise a warning is emitted and keypoint rules are skipped."
        ),
    )
    p.add_argument(
        "--kpt-conf",
        type=float,
        default=0.3,
        help="Minimum confidence for keypoints used by --kpt-indices.",
    )
    p.add_argument(
        "--flexible-kpt",
        type=int,
        default=None,
        help="If set to N: require at least N keypoints inside the face box when >=N survive conf thresh; "
             "if fewer survive, require all surviving inside.",
    )

    # Artifacts (all opt-in)
    p.add_argument(
        "--json",
        dest="save_json_flag",
        action="store_true",
        help="Write augmented JSON to <run>/faces.json (nothing is written unless a save flag is enabled).",
    )
    p.add_argument(
        "--frames",
        dest="save_frames",
        action="store_true",
        help="Save annotated JPEG frames to <run>/frames/ (can be large).",
    )
    p.add_argument(
        "--save-video",
        default=None,
        help="Save an annotated video to <run>/<name>.mp4 (provide filename, e.g. annotated.mp4).",
    )
    p.add_argument("--out-dir", default="out", help="Output root (used only when saving artifacts)")
    p.add_argument(
        "--run-name",
        default=None,
        help=(
            "Run folder name under --out-dir. If omitted, artifacts go directly under --out-dir. "
            "Note: no run folder is created unless a save flag is enabled."
        ),
    )
    p.add_argument("--fourcc", default="mp4v", help="FourCC codec for saved video")
    p.add_argument("--display", action="store_true", help="Display live annotated preview (press ESC to stop)")

    # UX
    p.add_argument("--no-progress", action="store_true", help="Disable progress bar (clean logs).")

    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = _parse_args(argv)

    if args.list_detectors:
        print("\n".join(available_detectors()))
        return

    if args.list_variants:
        print("\n".join(available_variants(args.detector)))
        return

    if not args.json_in or not args.video:
        raise SystemExit(
            "error: --json-in and --video are required unless --list-detectors or --list-variants is used"
        )

    associate_class_ids = _parse_int_list(args.associate_classes) if args.associate_classes else None
    kpt_indices = _parse_int_list(args.kpt_indices) if args.kpt_indices else None

    res = detect_faces_video(
        json_in=args.json_in,
        video=args.video,
        detector=args.detector,
        variant=args.variant,
        max_size=args.max_size,
        device=args.device,
        conf_thresh=args.conf_thresh,
        nms_thresh=args.nms_thresh,
        iou_thresh=args.iou_thresh,
        containment=args.containment,
        associate_class_ids=associate_class_ids,
        gallery_filter=args.gallery_filter,
        kpt_indices=kpt_indices,
        kpt_conf=args.kpt_conf,
        flexible_kpt=args.flexible_kpt,
        save_json_flag=args.save_json_flag,
        save_frames=args.save_frames,
        save_video=args.save_video,
        out_dir=args.out_dir,
        run_name=args.run_name,
        fourcc=args.fourcc,
        display=args.display,
        no_progress=args.no_progress,
    )

    print(json.dumps({"detect_face": res.stats, "paths": res.paths}, indent=2))


if __name__ == "__main__":
    main()