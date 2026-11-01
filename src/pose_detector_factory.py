"""
Shared pose detector factory and clip pose extraction.

Both app.py and evaluate_reconstruction.py import from here so they always
use identical detector settings.  The settings below must NOT be changed
without re-running src/evaluate_reconstruction.py and confirming that the
numbers in reports/reconstruction_evaluation.json stay the same.

Detector settings (MediaPipe PoseLandmarker, IMAGE mode)
---------------------------------------------------------
  min_pose_detection_confidence : 0.5  (MediaPipe default)
  min_pose_presence_confidence  : 0.5  (MediaPipe default)
  min_tracking_confidence       : 0.5  (MediaPipe default)
  num_poses                     : 1

These settings are printed at startup so they are always visible in logs.
"""

from __future__ import annotations

from pathlib import Path

# ── Constants ──────────────────────────────────────────────────────────────────
# These match MediaPipe defaults and are what evaluate_reconstruction.py uses.
DETECTOR_SETTINGS = {
    "min_pose_detection_confidence": 0.5,
    "min_pose_presence_confidence":  0.5,
    "min_tracking_confidence":       0.5,
    "num_poses":                     1,
    "running_mode":                  "IMAGE",
}


def print_detector_settings() -> None:
    """Print the shared detector settings to stdout."""
    print("[pose_detector_factory] Detector settings in use:")
    for k, v in DETECTOR_SETTINGS.items():
        print(f"  {k}: {v}")


def create_pose_detector(model_path: str | Path):
    """
    Create and return a MediaPipe PoseLandmarker (IMAGE mode) with the
    shared settings defined in DETECTOR_SETTINGS.

    Parameters
    ----------
    model_path : str | Path
        Absolute path to the .task model file.

    Returns
    -------
    mediapipe.tasks.vision.PoseLandmarker  — caller is responsible for .close()
    """
    import mediapipe as mp

    model_path = str(model_path)
    options = mp.tasks.vision.PoseLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=model_path),
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        min_pose_detection_confidence=DETECTOR_SETTINGS["min_pose_detection_confidence"],
        min_pose_presence_confidence=DETECTOR_SETTINGS["min_pose_presence_confidence"],
        min_tracking_confidence=DETECTOR_SETTINGS["min_tracking_confidence"],
        num_poses=DETECTOR_SETTINGS["num_poses"],
    )
    return mp.tasks.vision.PoseLandmarker.create_from_options(options)


def extract_poses_from_frames(frames: list, detector) -> list[dict]:
    """
    Run the given MediaPipe detector on a list of BGR frames.

    Parameters
    ----------
    frames   : list of np.ndarray (BGR)
    detector : PoseLandmarker in IMAGE mode

    Returns
    -------
    list of dicts:  {"detected": bool, "landmarks": [{x,y,z,visibility,presence}, ...]}
    """
    import cv2
    import mediapipe as mp

    poses: list[dict] = []
    _warned = False
    for frame_bgr in frames:
        try:
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            mp_image  = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            result    = detector.detect(mp_image)
            if result.pose_landmarks and len(result.pose_landmarks) > 0:
                lm_list = [
                    {
                        "x":          float(pt.x),
                        "y":          float(pt.y),
                        "z":          float(pt.z),
                        "visibility": float(getattr(pt, "visibility", 0.0) or 0.0),
                        "presence":   float(getattr(pt, "presence",   0.0) or 0.0),
                    }
                    for pt in result.pose_landmarks[0]
                ]
                poses.append({"detected": True, "landmarks": lm_list})
            else:
                poses.append({"detected": False, "landmarks": []})
        except Exception as exc:
            if not _warned:
                print(f"    [WARN] pose detection error: {exc!r}")
                _warned = True
            poses.append({"detected": False, "landmarks": []})
    return poses

