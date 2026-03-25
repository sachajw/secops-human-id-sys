# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Real-Time Human Identification System (RHIDS) — a deep learning pipeline that detects persons in video/image input and assigns each a persistent unique identifier across frames. Composes three open-source components: YOLOv6 (detection), ByteTrack/YOLOX (tracking), and OSNet-AIN (re-identification via feature extraction + cosine similarity).

## Setup

Model weights and demo videos are stored via Git LFS. The `models/` directory contains LFS pointers (~134 bytes), not actual weights.

```bash
git lfs install && git lfs pull                  # Download actual model files
pip install -r requirements.txt                   # Top-level deps (PyTorch 1.13.1, OpenCV, etc.)
cd deep_person_reid && pip install -r requirements.txt && python setup.py install
cd ByteTrack && pip install -r requirements.txt && python setup.py develop
cd YOLOv6 && pip install -r requirements.txt
```

## Running

The main pipeline lives in `RHIDS.ipynb` — execute cells sequentially:
- Cells 1-8: Load models (YOLOv6 detector + OSNet-AIN feature extractor)
- Cells 9-14: Known-human identification mode (user selects ROI, names target)
- Cells 15-20: Multi-person tracking (v1: YOLOv6; v2: ByteTrack YOLOX with center-displacement constraints)

ByteTrack standalone demo:
```bash
cd ByteTrack
python tools/demo_track.py video -f exps/example/mot/yolox_x_mix_det.py -c models/bytetrack_x_mot17.pth.tar --path video.mp4
```

## Architecture

```
Video Frame
  -> YOLOv6 DetectBackend (640x640, letterbox, NMS @ IoU 0.45)
    -> Bounding boxes filtered for "person" class (conf > 0.25)
      -> Crop person images
        -> OSNet-AIN FeatureExtractor (512-d vectors)
          -> Cosine similarity matrix (queries vs gallery)
            -> Match to existing ID (threshold ~0.6-0.7) or assign new ID
              -> Annotated output (bbox + ID + similarity score)
```

**Three detector options:** YOLOv6-Large (primary), ByteTrack YOLOX-X (v2 tracker), OSNet-AIN-X1.0 (feature extractor, not a detector).

**Two tracking modes in notebook:**
1. `identify_known_humans()` — targeted: user selects ROI, system tracks that person only
2. `multi_person_tracking()` / `multi_person_tracking_v2()` — assigns unique IDs to all detected persons; v2 adds 50px center-displacement constraint for track stability

## Key Directories

| Path | Purpose |
|------|---------|
| `RHIDS.ipynb` | Main pipeline notebook |
| `ByteTrack/` | ByteTrack submodule — YOLOX tracker (`yolox/tracker/byte_tracker.py`) |
| `YOLOv6/` | YOLOv6 submodule — detector (`yolov6/core/inferer.py`, `hubconf.py`) |
| `deep_person_reid/` | OSNet-AIN submodule — feature extractor (`torchreid/utils/feature_extractor.py`) |
| `models/` | Pretrained weights (LFS: `yolov6l.pt`, `osnet_ain.pth.tar-50`, `bytetrack_x_mot17.pth.tar`) |

## No Tests or Build System

This project has no test suite, Makefile, or CI/CD pipeline. It is notebook-driven research code. No linting or formatting tooling is configured at the project level (flake8/isort/yapf are in requirements but not wired up).
