# detect-face-lib

**Minimum Python:** `>=3.10`

**detect-face-lib** is a modular **face detection + JSON augmentation** toolkit that attaches face detections to **det-v1** (detections) or **track-v1** (tracked) payloads produced by earlier stages in the Vision Pipeline.

This is the **Face Augmentation stage** of the Vision Pipeline.

Detectors included:
- **retinaface**: `retinaface-pytorch` backend (named variant registry)

> By default, `detect-face-lib` **does not write any files**. You opt-in to saving JSON, frames, or annotated video via flags.

---

## Vision Pipeline

```
Original Video (.mp4)
        │
        ▼
  detect-lib
  (Detection Stage)
        │
        └── detections.json (det-v1)
                   │
                   ▼
                track-lib
           (Tracking + ReID)
                   │
                   └── tracked.json (track-v1)
                           │
                           ▼
                      detect-face-lib
                    (Face Augmentation)
                           │
                           └── faces.json (augmented; face-v1 meta)

Note: Each stage consumes the original video + the upstream JSON from the previous stage.
```

Stage 1 (Detection):
- PyPI: https://pypi.org/project/detect-lib/
- GitHub: https://github.com/Surya-Rayala/VideoPipeline-detection

Stage 2 (Tracking + ReID):
- PyPI: https://pypi.org/project/gallery-track-lib/
- GitHub: https://github.com/Surya-Rayala/VisionPipeline-gallery-track

---

## Output: augmented det-v1 / track-v1 (returned + optionally saved)

`detect-face-lib` returns an **augmented JSON payload** in-memory that preserves the upstream schema (det-v1 or track-v1) and adds:

- `face_augment`: metadata about the face detector + association rules (versioned)
- `detections[*].faces`: for each frame, matched faces are attached under the matched person detection

### What gets attached to a person

Each face entry is intentionally minimal:
- `bbox`: face box `[x1,y1,x2,y2]`
- `score`: face confidence
- `landmarks`: 5-point landmarks `[[x,y] x5]` when provided by the backend

---

## Minimal schema example

This example assumes the upstream JSON is **track-v1**; the same structure applies for **det-v1** (it will simply lack tracker fields).

```json
{
  "schema_version": "track-v1",
  "parent_schema_version": "det-v1",
  "video": {
    "path": "in.mp4",
    "fps": 30.0,
    "frame_count": 120,
    "width": 1920,
    "height": 1080
  },
  "tracker": {
    "name": "gallery_hybrid"
  },
  "face_augment": {
    "version": "face-v1",
    "parent_schema_version": "track-v1",
    "detector": {
      "name": "retinaface",
      "variant": "resnet50_2020-07-20",
      "max_size": 1024,
      "conf_thresh": 0.5,
      "nms_thresh": 0.3,
      "device": "mps"
    },
    "association": {
      "associate_class_ids": [0],
      "gallery_filter": false,
      "iou_thresh": 0.0,
      "containment": true,
      "inclusive": true,
      "kpt_indices": null,
      "kpt_conf": 0.3,
      "flexible_kpt": null
    }
  },
  "frames": [
    {
      "frame_index": 0,
      "detections": [
        {
          "bbox": [100.0, 50.0, 320.0, 240.0],
          "score": 0.91,
          "class_id": 0,
          "class_name": "person",
          "track_id": "3",
          "gallery_id": "person_A",
          "faces": [
            {
              "bbox": [140.0, 70.0, 210.0, 150.0],
              "score": 0.98,
              "landmarks": [[160.0, 95.0], [195.0, 95.0], [178.0, 110.0], [165.0, 130.0], [192.0, 130.0]]
            }
          ]
        }
      ]
    }
  ]
}
```

### Returned vs saved

- **Returned (always):** the augmented payload is available as `FaceResult.payload` (Python) and is always produced in-memory.
- **Saved (opt-in):** nothing is written unless you enable artifacts:
  - `--json` writes `<run>/faces.json`
  - `--frames` writes annotated frames under `<run>/frames/`
  - `--save-video` writes an annotated video under `<run>/...`

When no artifacts are enabled, no output directory/run folder is created.

---

## Install with `pip` (PyPI)

> Use this if you want to install and use the tool without cloning the repo.
> Requires **Python >= 3.10**.

### CUDA note (optional)

If you want GPU acceleration on NVIDIA CUDA, install a **CUDA-matching** build of **torch** and **torchvision**.

If you installed CPU-only wheels by accident, uninstall and reinstall the correct CUDA wheels (use the official PyTorch selector for your CUDA version).

```bash
pip uninstall -y torch torchvision
# then install the CUDA-matching wheels for your system
# (see: https://pytorch.org/get-started/locally/)
```

### Install

```bash
pip install detect-face-lib
```

> Note: the PyPI package name is `detect-face-lib`, but the Python module/import name remains `detect_face`.

---

## CLI usage (pip)

Global help:

```bash
python -m detect_face.cli.detect_faces -h
```

List detectors:

```bash
python -m detect_face.cli.detect_faces --list-detectors
```

List variants for a detector:

```bash
python -m detect_face.cli.detect_faces --detector retinaface --list-variants
```

---

## Face augmentation CLI: `detect_face.cli.detect_faces`

### Quick start (track-v1 input)

```bash
python -m detect_face.cli.detect_faces \
  --json-in tracked.json \
  --video in.mp4
```

### Quick start (det-v1 input)

```bash
python -m detect_face.cli.detect_faces \
  --json-in detections.json \
  --video in.mp4
```

### Save artifacts (opt-in)

```bash
python -m detect_face.cli.detect_faces \
  --json-in tracked.json \
  --video in.mp4 \
  --json \
  --frames \
  --save-video annotated.mp4 \
  --out-dir out --run-name demo
```

---

## CLI arguments

### Required (for running augmentation)

- `--json-in <path>`: Path to the **det-v1** or **track-v1** JSON to augment.
- `--video <path>`: Path to the original source video used to generate the JSON. Frame order must align.

### Discovery

- `--list-detectors`: Print available face detector backends and exit.
- `--list-variants`: Print supported named variants for `--detector` and exit.

### Detector selection

- `--detector <name>`: Face backend to use (default: `retinaface`).
- `--variant <name>`: Backend model variant (named variant). For `retinaface`, this is typically `resnet50_2020-07-20`.
- `--max-size <int>`: Cap longer side for detector input.
  - Lower → faster, may miss small faces.
  - Higher → slower, better for small faces.
- `--device <auto|cpu|mps|cuda|cuda:0|0|1...>`: Compute device.
  - `auto` resolves to `cuda:0` if available, else `mps`, else `cpu`.

### Detector thresholds

- `--conf-thresh <float>`: Minimum face confidence.
  - Increase → fewer false positives (fewer faces).
  - Decrease → more faces (more false positives).
- `--nms-thresh <float>`: Non-maximum suppression threshold inside the face detector.
  - Lower → more aggressive suppression.

### Association rules

- `--associate-classes <ids>`: Class IDs eligible for face association (e.g., `0` for person). If omitted, the tool tries to auto-detect class_name=='person'; if not found, all classes are eligible.
- `--iou-thresh <float>`: IoU threshold for candidate person-face pairing.
  - `0.0` considers any overlap.
  - `0.1–0.3` is stricter and can reduce wrong assignments.
- `--containment` / `--no-containment`: Require (or not) that the face box lies fully inside the person box.
  - Containment on → safer (fewer bad matches).
  - Containment off → more permissive (may increase mismatches in crowded frames).
- `--gallery-filter`: Track-v1 only: only assign faces where `track_id == gallery_id` (when both exist).

### Keypoint constraints (optional)

- `--kpt-indices <ids>`: Body keypoint indices to require inside the face box.
  - Only used if the input JSON contains `keypoints`. If keypoints are missing, a warning is emitted and the rule is skipped.
- `--kpt-conf <float>`: Minimum confidence for keypoints used by `--kpt-indices`.
- `--flexible-kpt <N>`: If set, requires at least `N` keypoints inside the face box when `>=N` survive the confidence threshold; if fewer survive, requires all surviving keypoints inside.

### Artifact saving (all opt-in)

- `--json`: Write augmented JSON to `<run>/faces.json`.
- `--frames`: Save annotated frames under `<run>/frames/` (can be large).
- `--save-video <name.mp4>`: Save an annotated video under `<run>/<name.mp4>`.
- `--out-dir <dir>`: Output root used only when saving artifacts (default: `out`).
- `--run-name <name>`: Run folder name under `--out-dir`. If omitted, artifacts go directly under `--out-dir`.
- `--fourcc <fourcc>`: FourCC codec for saved video (default: `mp4v`).
- `--display`: Show a live annotated preview (press ESC to stop). Does not write files unless saving flags are set.

### UX

- `--no-progress`: Disable progress bar.

---

## Python usage (import)

You can use `detect-face-lib` as a library after installing it with pip.

### Quick sanity check

```bash
python -c "import detect_face; print(detect_face.available_detectors())"
```

### Python API reference (keywords)

#### `detect_face.detect_faces_video(...)`

**Required**
- `json_in`: Path to det-v1 or track-v1 JSON.
- `video`: Path to the source video.

**Detector**
- `detector`: Backend name (e.g., `"retinaface"`).
- `variant`: Variant name for the backend.
- `max_size`: Cap longer side.
- `device`: `"auto"`, `"cpu"`, `"mps"`, `"cuda"`, `"cuda:0"`, or an index like `"0"`.
- `conf_thresh`, `nms_thresh`: Detector thresholds.

**Association**
- `associate_class_ids`: List of class_ids eligible for association.
- `iou_thresh`, `containment`: Pairing rules.
- `gallery_filter`: Track-v1 only.

**Keypoints (optional)**
- `kpt_indices`, `kpt_conf`, `flexible_kpt`: Keypoint-based validation (used only when keypoints exist).

**Artifacts (all off by default)**
- `save_json_flag`: Write `<run>/faces.json`.
- `save_frames`: Write `<run>/frames/*.jpg`.
- `save_video`: Filename for annotated video under the run folder.
- `out_dir`, `run_name`, `fourcc`, `display`, `no_progress`.

Returns a `FaceResult` with `payload` (augmented JSON), `paths` (only populated when saving), and `stats`.

### Run face augmentation from a Python file

Create `run_faces.py`:

```python
from detect_face import detect_faces_video

res = detect_faces_video(
    json_in="tracked.json",
    video="in.mp4",
    detector="retinaface",
    variant="resnet50_2020-07-20",
    device="auto",
)

print(res.stats)
print("face_augment" in res.payload)
print(res.paths)  # populated only if you enable saving artifacts
```

Run:

```bash
python run_faces.py
```

---

## Install from GitHub (uv)

Use this if you are developing locally or want reproducible project environments.

Install uv:
https://docs.astral.sh/uv/getting-started/installation/#standalone-installer

Verify:

```bash
uv --version
```

### CUDA note (optional)

For best performance on NVIDIA GPUs, make sure **torch** and **torchvision** are installed with a build that matches your CUDA toolkit / driver stack.

If you added CPU-only builds earlier, remove them and add the correct CUDA wheels, then re-sync.

```bash
uv remove torch torchvision
# then add the CUDA-matching wheels for your system
# (see: https://pytorch.org/get-started/locally/)
uv add torch torchvision
uv sync
```

### Install dependencies

```bash
git clone https://github.com/Surya-Rayala/VisionPipeline-detect-face.git
cd VisionPipeline-detect-face
uv sync
```

---

## CLI usage (uv)

Global help:

```bash
uv run python -m detect_face.cli.detect_faces -h
```

List detectors / variants:

```bash
uv run python -m detect_face.cli.detect_faces --list-detectors
uv run python -m detect_face.cli.detect_faces --detector retinaface --list-variants
```

Basic command (track-v1 input):

```bash
uv run python -m detect_face.cli.detect_faces \
  --json-in tracked.json \
  --video in.mp4
```

Basic command (det-v1 input):

```bash
uv run python -m detect_face.cli.detect_faces \
  --json-in detections.json \
  --video in.mp4
```

Save artifacts (opt-in):

```bash
uv run python -m detect_face.cli.detect_faces \
  --json-in tracked.json \
  --video in.mp4 \
  --json --frames --save-video annotated.mp4 \
  --out-dir out --run-name demo
```

---

# License

This project is licensed under the **MIT License**. See `LICENSE`.