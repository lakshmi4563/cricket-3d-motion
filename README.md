# Cricket3D_Motion

## 1. Project Goal
The objective of this research is **motion-aware keyframe selection and lightweight 3D cricket-player pose estimation** from short broadcast/field video clips. The system is designed to identify critical motion phases (backlift, impact/execution, and follow-through) and estimate 3D player kinematics while strictly operating within resource-constrained consumer hardware.

---

## 2. Planned Pipeline
```
Short Video (.avi)
       │
       ▼
[Motion-Aware Frame Selection]
  - Temporal optical flow magnitude / frame-differencing energy
  - Adaptive sampling to isolate key stroke phases (stance, impact, follow-through)
       │
       ▼
[Lightweight 2D Player Detection & Pose Estimation]
  - Crop player region using a lightweight detector (e.g., YOLOv8-nano / MediaPipe)
  - Extract 2D body joints (17/33 keypoints)
       │
       ▼
[Monocular 2D-to-3D Pose Lifting]
  - Lightweight temporal/GCN/MLP lifter (e.g., lightweight VideoPose3D, MotionBERT-lite, or MediaPipe World Landmarks)
  - Low memory footprint suitable for 4 GB VRAM
       │
       ▼
[Kinematic Metric Extraction & Visualisation]
  - Joint angles (elbow flexion, shoulder rotation, knee bend)
  - 3D pose trajectory export and skeletal motion rendering
```

---

## 3. Dataset Classes & Archive Summary
The project utilizes the **CricShot10k** dataset located in `CricShot10k_ Shot Dataset/` comprising 10 distinct shot classes packaged as RAR archives (total size ~1.7 GB):

| Shot Class Archive | Size (MB) | Format | Extracted Status |
| :--- | :--- | :--- | :--- |
| `Cover Drive.rar` | ~264.9 MB | `.avi` clips | Not extracted (original archive intact) |
| `Defensive.rar` | ~205.2 MB | `.avi` clips | Not extracted (original archive intact) |
| `Flick.rar` | ~322.8 MB | `.avi` clips | Not extracted (original archive intact) |
| `Hook.rar` | ~196.1 MB | `.avi` clips | Not extracted (original archive intact) |
| `Late Cut.rar` | ~107.1 MB | `.avi` clips | Not extracted (original archive intact) |
| `Pull.rar` | ~280.9 MB | `.avi` clips | Not extracted (original archive intact) |
| `Reverse Sweep.rar`| ~95.0 MB | `.avi` clips | Not extracted (original archive intact) |
| `Scoop.rar` | ~106.2 MB | `.avi` clips | Not extracted (original archive intact) |
| `Straight Drive.rar`| ~143.6 MB| `.avi` clips | Not extracted (original archive intact) |
| `Upper Cut.rar` | ~98.9 MB | `.avi` clips | Not extracted (original archive intact) |

---

## 4. Hardware Constraints & VRAM Budget
- **GPU**: NVIDIA GeForce RTX 3050 Laptop GPU (Dedicated VRAM: **4096 MiB / 4 GB**)
- **Driver / CUDA**: Driver 592.00, CUDA 13.1 support
- **Design Rules**:
  - Keep model batch size = 1 or use sequence sliding windows <= 16 frames.
  - Employ FP16 mixed precision or ONNX Runtime inference where applicable.
  - Avoid heavy transformer backbones (e.g., standard ViTPose-huge, heavy SMPL meshes) that exceed 3.5 GB peak VRAM.
  - Offload video decoding and frame buffering to host RAM.

---

## 5. Limitations of Monocular 3D Pose Estimation
Estimating 3D kinematics from uncalibrated monocular broadcast cricket videos introduces specific scientific and technical challenges:
1. **Depth Ambiguity**: Multiple 3D skeletal configurations can project to the exact same 2D pixel coordinates.
2. **Severe Self-Occlusion & Equipment Interaction**: Batting stances routinely occlude the trailing arm, front pad, or wrists during the downswing; bats and pads frequently obscure joints.
3. **Dynamic Camera Motion & Motion Blur**: Broadcast camera pans, zooms, and fast bat/hand motions during shot execution degrade edge sharpness and joint detection confidence.
4. **Lack of Ground-Truth 3D Data**: Markerless field videos lack 3D motion-capture ground truth; validation relies on relative 2D reprojection error, anatomical plausibility constraints, and temporal smoothness metrics.

---

## 6. Directory Layout
- `CricShot10k_ Shot Dataset/` : Source RAR archives of cricket shot clips.
- `outputs/sample_clips/` : Small AVI samples for pose analysis and matching MP4 previews for browser playback.
- `outputs/` : Intermediate keypoint representations and 3D pose results.
- `reports/` : Experimental benchmarks, ablation notes, and progress reports.
