"""
3D Cricket Player Pose Visualization from Monocular Video.
Extracts 3D pose_world_landmarks using MediaPipe Tasks PoseLandmarker (CPU-only)
and renders multi-view 3D skeletal snapshots (front, side, isometric, and combined tri-view).
Also provides an optional interactive Matplotlib 3D viewer with mouse rotation and frame scrubbing.

Scientific & Methodological Note:
These 3D coordinates represent estimated monocular body poses output by MediaPipe's
pose_world_landmarks neural network. They are rooted at the player's hip midpoint in approximate
metric units. They do NOT constitute ground-truth optical motion capture or calibrated metric 3D reconstruction.
"""

from pathlib import Path
import argparse
import sys
import numpy as np

try:
    import cv2
except ImportError:
    print("[ERROR] OpenCV (cv2) is not installed.")
    sys.exit(1)

try:
    import matplotlib
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
except ImportError:
    print("[ERROR] Matplotlib is not installed.")
    sys.exit(1)

try:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision
except ImportError as e:
    print(f"[ERROR] MediaPipe Tasks import failed: {e}")
    sys.exit(1)


# Full 33-landmark anatomical skeletal connectivity
SKELETAL_CONNECTIONS = [
    # Torso
    (11, 12, "torso"),  # Left shoulder to Right shoulder
    (11, 23, "torso"),  # Left shoulder to Left hip
    (12, 24, "torso"),  # Right shoulder to Right hip
    (23, 24, "torso"),  # Left hip to Right hip
    # Left Arm
    (11, 13, "left_arm"),   # Shoulder to Elbow
    (13, 15, "left_arm"),   # Elbow to Wrist
    (15, 17, "left_hand"),  # Wrist to Pinky
    (15, 19, "left_hand"),  # Wrist to Index
    (15, 21, "left_hand"),  # Wrist to Thumb
    (17, 19, "left_hand"),  # Pinky to Index
    # Right Arm
    (12, 14, "right_arm"),  # Shoulder to Elbow
    (14, 16, "right_arm"),  # Elbow to Wrist
    (16, 18, "right_hand"), # Wrist to Pinky
    (16, 20, "right_hand"), # Wrist to Index
    (16, 22, "right_hand"), # Wrist to Thumb
    (18, 20, "right_hand"), # Pinky to Index
    # Left Leg
    (23, 25, "left_leg"),   # Hip to Knee
    (25, 27, "left_leg"),   # Knee to Ankle
    (27, 29, "left_foot"),  # Ankle to Heel
    (27, 31, "left_foot"),  # Ankle to Toe
    (29, 31, "left_foot"),  # Heel to Toe
    # Right Leg
    (24, 26, "right_leg"),  # Hip to Knee
    (26, 28, "right_leg"),  # Knee to Ankle
    (28, 30, "right_foot"), # Ankle to Heel
    (28, 32, "right_foot"), # Ankle to Toe
    (30, 32, "right_foot"), # Heel to Toe
    # Head / Face
    (0, 1, "head"), (1, 2, "head"), (2, 3, "head"), (3, 7, "head"), # Left eye/ear
    (0, 4, "head"), (4, 5, "head"), (5, 6, "head"), (6, 8, "head"), # Right eye/ear
    (9, 10, "head"), # Mouth
]

COLOR_MAP = {
    "torso": "#00ADB5",       # Cyan
    "left_arm": "#FF5722",    # Deep Orange
    "right_arm": "#4CAF50",   # Green
    "left_hand": "#FF9800",   # Amber
    "right_hand": "#8BC34A",  # Light Green
    "left_leg": "#E91E63",    # Pink / Magenta
    "right_leg": "#2196F3",   # Blue
    "left_foot": "#F06292",   # Soft Pink
    "right_foot": "#64B5F6",  # Soft Blue
    "head": "#FFC107",        # Gold
}


def create_pose_detector(model_path: Path):
    """Initializes MediaPipe Tasks PoseLandmarker on CPU."""
    base_options = mp_python.BaseOptions(model_asset_path=str(model_path))
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        min_pose_detection_confidence=0.3,
        min_pose_presence_confidence=0.3,
        min_tracking_confidence=0.3,
        num_poses=1,
    )
    return vision.PoseLandmarker.create_from_options(options)


def extract_single_frame_world_landmarks(video_path: Path, frame_idx: int, detector):
    """
    Extracts frame image and 33 pose_world_landmarks for a given frame index.
    Returns:
        frame_bgr: numpy array
        world_landmarks: list of dicts with x, y, z, visibility
        detected: bool
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open {video_path}")

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame_bgr = cap.read()
    cap.release()

    if not ret or frame_bgr is None:
        return None, [], False

    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    result = detector.detect(mp_image)

    if result.pose_world_landmarks and len(result.pose_world_landmarks) > 0:
        wlm = result.pose_world_landmarks[0]
        pts = []
        for p in wlm:
            # MediaPipe world landmarks: Y points downwards in camera frame; negate Y so Y is upright
            pts.append({
                "x": float(p.x),
                "y": -float(p.y),
                "z": float(p.z),
                "visibility": float(getattr(p, "visibility", 0.0)),
            })
        return frame_bgr, pts, True
    else:
        return frame_bgr, [], False


def plot_skeleton_on_axis(ax, landmarks, view_type="isometric"):
    """
    Renders 3D skeleton on a given 3D Matplotlib axis with equal scaling and specific camera angle.
    """
    if not landmarks:
        ax.text(0, 0, 0, "No Pose Detected", color="red", ha="center")
        return

    xs = [pt["x"] for pt in landmarks]
    ys = [pt["z"] for pt in landmarks]  # Map depth to horizontal depth axis
    zs = [pt["y"] for pt in landmarks]  # Upright elevation

    # Plot bones
    for start_idx, end_idx, limb_type in SKELETAL_CONNECTIONS:
        if start_idx < len(landmarks) and end_idx < len(landmarks):
            p1, p2 = landmarks[start_idx], landmarks[end_idx]
            v1, v2 = p1["visibility"], p2["visibility"]
            if v1 >= 0.3 and v2 >= 0.3:
                color = COLOR_MAP.get(limb_type, "#00ADB5")
                alpha = 0.9 if (v1 >= 0.5 and v2 >= 0.5) else 0.4
                lw = 2.5 if (v1 >= 0.5 and v2 >= 0.5) else 1.2
                ax.plot(
                    [p1["x"], p2["x"]],
                    [p1["z"], p2["z"]],
                    [p1["y"], p2["y"]],
                    color=color,
                    alpha=alpha,
                    linewidth=lw,
                )

    # Plot landmark joints
    joint_colors = []
    for pt in landmarks:
        if pt["visibility"] >= 0.5:
            joint_colors.append("#00FF66")  # Green for high confidence
        elif pt["visibility"] >= 0.3:
            joint_colors.append("#FFAA00")  # Orange for moderate confidence
        else:
            joint_colors.append("#555555")  # Dark grey for low confidence

    ax.scatter(xs, ys, zs, c=joint_colors, s=18, depthshade=True, edgecolors="#111111", linewidths=0.5)

    # Enforce strictly equal aspect ratio
    max_range = np.array([
        max(xs) - min(xs),
        max(ys) - min(ys),
        max(zs) - min(zs)
    ]).max() / 2.0
    mid_x = (max(xs) + min(xs)) * 0.5
    mid_y = (max(ys) + min(ys)) * 0.5
    mid_z = (max(zs) + min(zs)) * 0.5

    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    # Dark aesthetic theme
    ax.set_facecolor("#181818")
    ax.grid(True, linestyle="--", alpha=0.25, color="#888888")
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False

    ax.set_xlabel("X (m)", color="#CCCCCC", fontsize=8, labelpad=2)
    ax.set_ylabel("Z (m)", color="#CCCCCC", fontsize=8, labelpad=2)
    ax.set_zlabel("Y (m, Up)", color="#CCCCCC", fontsize=8, labelpad=2)
    ax.tick_params(colors="#888888", labelsize=7)

    # Camera angles: elevation (elev), azimuth (azim)
    if view_type == "front":
        ax.view_init(elev=5, azim=-90)
        ax.set_title("Front View (XZ)", color="#E0E0E0", fontsize=10, pad=4)
    elif view_type == "side":
        ax.view_init(elev=5, azim=0)
        ax.set_title("Side View (YZ)", color="#E0E0E0", fontsize=10, pad=4)
    elif view_type == "isometric":
        ax.view_init(elev=20, azim=-45)
        ax.set_title("Isometric 3D View", color="#E0E0E0", fontsize=10, pad=4)


def render_tri_view_snapshot(frame_bgr, landmarks, clip_class, clip_name, frame_idx, out_path: Path):
    """
    Renders a 4-panel publication card:
    - Panel 1: Original RGB 2D Video Frame
    - Panel 2: Front View 3D Skeleton
    - Panel 3: Side View 3D Skeleton
    - Panel 4: Isometric 3D Skeleton
    """
    fig = plt.figure(figsize=(16, 4.8), facecolor="#121212")

    # Panel 1: Original 2D Frame
    ax_img = fig.add_subplot(1, 4, 1)
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    ax_img.imshow(frame_rgb)
    ax_img.set_title(f"Source Frame #{frame_idx}", color="#E0E0E0", fontsize=10, pad=4)
    ax_img.axis("off")

    # Panel 2: Front 3D
    ax_front = fig.add_subplot(1, 4, 2, projection="3d")
    plot_skeleton_on_axis(ax_front, landmarks, view_type="front")

    # Panel 3: Side 3D
    ax_side = fig.add_subplot(1, 4, 3, projection="3d")
    plot_skeleton_on_axis(ax_side, landmarks, view_type="side")

    # Panel 4: Isometric 3D
    ax_iso = fig.add_subplot(1, 4, 4, projection="3d")
    plot_skeleton_on_axis(ax_iso, landmarks, view_type="isometric")

    # Master title banner with explicit scientific disclaimer
    header_text = (
        f"Cricket 3D Pose Estimation  |  Shot: {clip_class} ({clip_name})  |  Frame #{frame_idx}\n"
        "[Estimated Monocular Pose via MediaPipe World Landmarks (m) — Not Optical Motion-Capture Ground Truth]"
    )
    fig.suptitle(header_text, color="#FFFFFF", fontsize=10.5, y=0.98, weight="bold")

    plt.tight_layout(rect=[0, 0.02, 1, 0.94])
    plt.savefig(str(out_path), dpi=150, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)


def render_single_view_snapshot(landmarks, clip_class, clip_name, frame_idx, view_type, out_path: Path):
    """Renders a standalone high-resolution single-view 3D skeleton plot."""
    fig = plt.figure(figsize=(6, 6), facecolor="#121212")
    ax = fig.add_subplot(1, 1, 1, projection="3d")
    plot_skeleton_on_axis(ax, landmarks, view_type=view_type)

    title = f"{clip_class} ({clip_name}) - Frame #{frame_idx} [{view_type.upper()} 3D]\n(Estimated Monocular 3D Pose)"
    ax.set_title(title, color="#FFFFFF", fontsize=9, pad=8)

    plt.tight_layout()
    plt.savefig(str(out_path), dpi=150, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)


def run_interactive_viewer(video_path: Path, detector):
    """
    Advanced dual-panel interactive 3D Matplotlib viewer with widget controls.
    Features:
      - Left Panel: Original Video Frame with shot class, frame index, and status badge.
      - Right Panel: 3D Skeleton when pose is detected; clear 'No Pose Detected' overlay when missed.
      - Slider: Interactive timeline slider to scrub to any frame in real-time.
      - Play/Pause Button: Automatically advances playback through the clip at ~10-15 FPS.
      - Reset View Button: Restores default isometric camera perspective (elev=20, azim=-45).
      - Camera Angle Preservation: Retains user's custom mouse rotation and zoom across frame changes.
      - Missing Pose Safety: Never persists or reuses a previous frame's skeleton.
      - Keyboard Shortcuts: Left/Right = scrub, Space = play/pause, R = reset view.
    """
    from matplotlib.widgets import Slider, Button

    clip_name = video_path.name
    shot_class = video_path.parent.name
    print(f"\n[INFO] Initializing interactive 3D pose viewer for: {shot_class} ({clip_name})")

    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    if total_frames <= 0:
        raise RuntimeError(f"Cannot read video: {video_path}")

    # Pre-extract frames and world landmarks into memory for fluid interactive scrubbing
    print(f"[INFO] Pre-caching {total_frames} frames for zero-latency scrubbing & playback...")
    cache = []
    cap = cv2.VideoCapture(str(video_path))
    for f_idx in range(total_frames):
        ret, frame_bgr = cap.read()
        if not ret or frame_bgr is None:
            cache.append((None, [], False))
            continue
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        result = detector.detect(mp_image)
        if result.pose_world_landmarks and len(result.pose_world_landmarks) > 0:
            pts = [
                {
                    "x": float(p.x),
                    "y": -float(p.y),
                    "z": float(p.z),
                    "visibility": float(getattr(p, "visibility", 0.0)),
                }
                for p in result.pose_world_landmarks[0]
            ]
            cache.append((frame_bgr, pts, True))
        else:
            cache.append((frame_bgr, [], False))
    cap.release()
    print(f"[INFO] Caching complete. Valid poses: {sum(1 for c in cache if c[2])}/{total_frames}")

    # Set up Matplotlib figure with control widget margins
    fig = plt.figure(figsize=(14, 7.2), facecolor="#141414")
    fig.canvas.manager.set_window_title(f"Cricket 3D Pose Viewer — {shot_class} ({clip_name})")

    # Leave bottom margin for Slider & Buttons
    ax_img = fig.add_axes([0.05, 0.20, 0.42, 0.70])
    ax_3d = fig.add_axes([0.52, 0.18, 0.45, 0.74], projection="3d")

    # Widget Axes
    ax_slider = fig.add_axes([0.15, 0.07, 0.52, 0.04], facecolor="#242424")
    ax_play = fig.add_axes([0.70, 0.065, 0.12, 0.05], facecolor="#2C3E50")
    ax_reset = fig.add_axes([0.84, 0.065, 0.11, 0.05], facecolor="#34495E")

    # Interactive Widgets
    slider = Slider(
        ax=ax_slider,
        label="Frame  ",
        valmin=0,
        valmax=total_frames - 1,
        valinit=12 if total_frames > 12 else 0,
        valstep=1,
        color="#00ADB5",
        initcolor="none"
    )
    slider.label.set_color("#FFFFFF")
    slider.label.set_fontsize(10)
    slider.valtext.set_color("#00FFCC")
    slider.valtext.set_fontsize(10)

    btn_play = Button(ax_play, "Play ►", color="#1E2A38", hovercolor="#2E4053")
    btn_play.label.set_color("#FFFFFF")
    btn_play.label.set_fontsize(10)
    btn_play.label.set_weight("bold")

    btn_reset = Button(ax_reset, "Reset View ⟲", color="#1E2A38", hovercolor="#2E4053")
    btn_reset.label.set_color("#FFFFFF")
    btn_reset.label.set_fontsize(9.5)

    # State variables
    current_frame = int(slider.val)
    is_playing = False
    timer = None
    cam_elev = 20.0
    cam_azim = -45.0

    def render_frame(f_idx, reset_camera=False):
        nonlocal cam_elev, cam_azim
        f_idx = int(f_idx)
        frame_bgr, wlm, detected = cache[f_idx]

        # Preserve user's interactive rotation unless explicit reset is requested
        if not reset_camera and ax_3d.azim is not None and ax_3d.elev is not None:
            cam_elev = ax_3d.elev
            cam_azim = ax_3d.azim

        ax_img.cla()
        ax_3d.cla()

        # 1. Left Panel: Original Video Frame
        if frame_bgr is not None:
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            ax_img.imshow(frame_rgb)
        ax_img.axis("off")

        status_str = "POSE DETECTED (33 Landmarks)" if detected else "NO POSE DETECTED"
        status_col = "#00FF66" if detected else "#FF3333"
        ax_img.set_title(
            f"Shot: {shot_class} | Frame #{f_idx} of {total_frames - 1}\nDetection Status: {status_str}",
            color=status_col,
            fontsize=10.5,
            weight="bold",
            pad=8
        )

        # 2. Right Panel: 3D Pose
        if detected and wlm:
            plot_skeleton_on_axis(ax_3d, wlm, view_type="isometric")
            ax_3d.view_init(elev=cam_elev, azim=cam_azim)
            ax_3d.set_title(
                f"Estimated Monocular 3D Skeleton (m)\nFrame #{f_idx} (Elev: {int(cam_elev)}°, Azim: {int(cam_azim)}°)",
                color="#FFFFFF",
                fontsize=10,
                pad=8
            )
        else:
            # Completely clear 3D space with prominent warning badge
            ax_3d.set_facecolor("#181818")
            ax_3d.grid(True, linestyle="--", alpha=0.15, color="#888888")
            ax_3d.xaxis.pane.fill = False
            ax_3d.yaxis.pane.fill = False
            ax_3d.zaxis.pane.fill = False
            ax_3d.set_xlim(-1, 1)
            ax_3d.set_ylim(-1, 1)
            ax_3d.set_zlim(-1, 1)
            ax_3d.view_init(elev=cam_elev, azim=cam_azim)
            ax_3d.text(
                0, 0, 0,
                f"NO POSE DETECTED\nFrame #{f_idx} / {total_frames - 1}\n(Zero-Persistence Mode Active)",
                color="#FF5555",
                ha="center",
                va="center",
                fontsize=11.5,
                weight="bold"
            )
            ax_3d.set_title(
                f"3D Skeleton — No Pose Detected (Frame #{f_idx})",
                color="#FF5555",
                fontsize=10,
                pad=8
            )

        # Master Super Title
        fig.suptitle(
            f"Cricket 3D Pose Viewer — {shot_class} ({clip_name})  |  [Space] Play/Pause  [Arrows] Step  [Drag] Rotate 3D  [R] Reset View",
            color="#EAEAEA",
            fontsize=11,
            y=0.97
        )
        fig.canvas.draw_idle()

    # Callback: Slider scrub
    def on_slider_change(val):
        nonlocal current_frame
        current_frame = int(val)
        render_frame(current_frame)

    slider.on_changed(on_slider_change)

    # Callback: Play/Pause
    def toggle_play(event=None):
        nonlocal is_playing, timer
        is_playing = not is_playing
        if is_playing:
            btn_play.label.set_text("Pause ❚❚")
            btn_play.color = "#8A2B2B"
            # Start timer at ~10 FPS (100 ms step)
            timer = fig.canvas.new_timer(interval=100)
            timer.add_callback(advance_frame)
            timer.start()
        else:
            btn_play.label.set_text("Play ►")
            btn_play.color = "#1E2A38"
            if timer:
                timer.stop()
                timer = None
        fig.canvas.draw_idle()

    def advance_frame():
        nonlocal current_frame
        next_frame = (current_frame + 1) % total_frames
        current_frame = next_frame
        slider.set_val(current_frame)

    btn_play.on_clicked(toggle_play)

    # Callback: Reset View
    def on_reset_view(event=None):
        nonlocal cam_elev, cam_azim
        cam_elev = 20.0
        cam_azim = -45.0
        render_frame(current_frame, reset_camera=True)

    btn_reset.on_clicked(on_reset_view)

    # Callback: Keyboard shortcuts
    def on_key(event):
        nonlocal current_frame
        if event.key in ["right", "up"]:
            current_frame = min(current_frame + 1, total_frames - 1)
            slider.set_val(current_frame)
        elif event.key in ["left", "down"]:
            current_frame = max(current_frame - 1, 0)
            slider.set_val(current_frame)
        elif event.key == " ":
            toggle_play()
        elif event.key in ["r", "R"]:
            on_reset_view()

    fig.canvas.mpl_connect("key_press_event", on_key)

    # Initial render
    render_frame(current_frame, reset_camera=True)
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Cricket 3D Pose Visualizer")
    parser.add_argument("--interactive", action="store_true", help="Launch interactive 3D Matplotlib viewer")
    parser.add_argument("--clip", type=str, default=None, help="Relative clip path for interactive viewer (e.g. outputs/sample_clips/Pull/vid100_29.avi)")
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parents[1]
    model_path = base_dir / "models" / "pose_landmarker_lite.task"
    out_dir = base_dir / "outputs" / "3d_pose_visualizations"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not model_path.exists():
        raise FileNotFoundError(f"Model file missing: {model_path}")

    # Target representative cricket shots and stroke keyframes:
    # 1. Cover Drive (downswing/impact drive)
    # 2. Pull Shot (horizontal pull shot mid-execution)
    # 3. Scoop Shot (innovative low ramp execution)
    sample_targets = [
        ("Cover Drive", "outputs/sample_clips/Cover Drive/vid100_41.avi", 14),
        ("Pull", "outputs/sample_clips/Pull/vid100_29.avi", 12),
        ("Scoop", "outputs/sample_clips/Scoop/vid100_28.avi", 15),
    ]

    print("=" * 86)
    print("3D CRICKET POSE VISUALIZATION (CPU-ONLY)")
    print(f"Generating 3D multi-view skeletal visualizations into: {out_dir.relative_to(base_dir)}/")
    print("=" * 86)

    detector = create_pose_detector(model_path)
    created_files = []

    try:
        for shot_class, rel_path, frame_idx in sample_targets:
            video_file = base_dir / rel_path
            clip_name = video_file.name
            safe_class = shot_class.replace(" ", "_")

            print(f"\nProcessing {shot_class} ({clip_name}), Frame #{frame_idx}...")
            frame_bgr, landmarks, detected = extract_single_frame_world_landmarks(
                video_file, frame_idx, detector
            )

            if not detected:
                print(f"  [WARNING] Pose not detected for {shot_class} at frame {frame_idx}")
                continue

            # 1. Combined Tri-View Snapshot (Source 2D + Front + Side + Isometric)
            tri_name = f"3d_pose_{safe_class}_{video_file.stem}_frame{frame_idx}_triview.png"
            tri_path = out_dir / tri_name
            render_tri_view_snapshot(frame_bgr, landmarks, shot_class, clip_name, frame_idx, tri_path)
            created_files.append(tri_name)
            print(f"  [SAVED] Combined Tri-View: {tri_name}")

            # 2. Standalone Views (Front, Side, Isometric)
            for view in ["front", "side", "isometric"]:
                view_name = f"3d_pose_{safe_class}_{video_file.stem}_frame{frame_idx}_{view}.png"
                view_path = out_dir / view_name
                render_single_view_snapshot(landmarks, shot_class, clip_name, frame_idx, view, view_path)
                created_files.append(view_name)
                print(f"  [SAVED] {view.capitalize()} View: {view_name}")

        print("\n" + "=" * 86)
        print(f"[SUCCESS] Generated {len(created_files)} 3D skeletal visualization snapshots.")
        print(f"Destination Folder: {out_dir.relative_to(base_dir)}/")
        print("=" * 86)

        if args.interactive:
            # Choose clip from CLI argument or default to Pull shot (vid100_29.avi)
            if args.clip:
                target_clip_path = base_dir / args.clip
            else:
                target_clip_path = base_dir / sample_targets[1][1]  # Default to Pull shot where frame 12 is valid

            run_interactive_viewer(target_clip_path, detector)

    finally:
        detector.close()
        print("MediaPipe PoseLandmarker closed cleanly.")


if __name__ == "__main__":
    main()
