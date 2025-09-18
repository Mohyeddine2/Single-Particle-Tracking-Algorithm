import cv2
import numpy as np

def ensure_bgr_format(frame):
    """
    Ensure the frame is in BGR format for OpenCV operations.
    """
    if len(frame.shape) == 2:  # Grayscale
        return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    elif len(frame.shape) == 3:
        if frame.shape[2] == 1:  # Grayscale with single channel dimension
            frame_2d = frame.squeeze(axis=2)  # Remove the channel dimension
            return cv2.cvtColor(frame_2d, cv2.COLOR_GRAY2BGR)
        elif frame.shape[2] == 3:  # Already 3 channels
            return frame
        else:
            raise ValueError(f"Unexpected number of channels: {frame.shape[2]}")
    else:
        raise ValueError(f"Unexpected frame shape: {frame.shape}")
def draw_tracks_on_video(tracks, frames, all_detections, masks, output_path, fps=10):
    """
    Generates an annotated video with trajectories, detections, and masks.

    Args:
        tracks: list of trajectories [(frame, id, (x, y)), ...] grouped by id
        frames: list of images (grayscale)
        all_detections: list per frame of np.ndarray Nx2 with detected positions
        masks: list of masks (or a single mask), or None
        output_path: output path for the video
        fps: frames per second
    """
    height, width = frames[0].shape[:2]
    size = (width, height)

    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    out = cv2.VideoWriter(output_path, fourcc, fps, size, isColor=True)

    # Generation of unique colors per trajectory
    rng = np.random.default_rng(0)
    colors = {}
    for i in range(len(tracks)):
        hue = int(179 * i / len(tracks))
        hsv_color = np.uint8([[[hue, 255, 200]]])
        rgb_color = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2BGR)[0, 0]
        colors[i] = tuple(int(c) for c in rgb_color)

    for t, frame in enumerate(frames):
        frame_bgr = ensure_bgr_format(frame)

        # Draw trajectories
        for i, trk in enumerate(tracks):
            points = [p for p in trk if p[0] <= t]
            if len(points) >= 2:
                for j in range(1, len(points)):
                    pt1 = tuple(map(int, points[j - 1][2][:2]))
                    pt2 = tuple(map(int, points[j][2][:2]))
                    cv2.line(frame_bgr, pt1, pt2, colors[i], 1)

        # Draw current detections
        # Red circles on detections at time t
        if len(all_detections[t]) > 0:
            cx, cy = all_detections[t][:, 0], all_detections[t][:, 1]
            for k in range(len(cx)):
                cv2.circle(frame_bgr, (int(cx[k]), int(cy[k])), 2, (0, 0, 255), -1)
        # Text for the current frame
        cv2.putText(
            frame_bgr,
            f"t{t} --- Traj: {len(tracks)} --- Detections: {len(all_detections[t])}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
            cv2.LINE_AA
        )
        # Add mask
        if masks is not None:
            mask = masks[t] if isinstance(masks, list) and t < len(masks) else masks[0]
            mask_uint8 = mask.astype(np.uint8)  # 🔧 conversion au bon type
            added = cv2.add(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY), mask_uint8)
            frame_bgr = cv2.cvtColor(added, cv2.COLOR_GRAY2BGR)
        out.write(frame_bgr)
    out.release()
    print(f" Vidéo enregistrée dans {output_path}")


def draw_detections_on_video(frames, all_detections, masks, output_path, fps=10):
    """
    Generates a video annotated with trajectories, detections, and masks.

    Args:
        tracks: list of trajectories [(frame, id, (x, y)), ...] grouped by id
        frames: list of images (grayscale)
        all_detections: list per frame of np.ndarray Nx2 with detected positions
        masks: list of masks (or a single mask), or None
        output_path: output path for the video
        fps: frames per second
    """

    height, width = frames[0].shape[:2]
    size = (width, height)
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    out = cv2.VideoWriter(output_path, fourcc, fps, size, isColor=True)
    for t, frame in enumerate(frames):
        frame_bgr = ensure_bgr_format(frame)
        # Draw current detections
        # Red circles on detections at time t
        if len(all_detections[t]) > 0:
            cx, cy = all_detections[t][:, 0], all_detections[t][:, 1]
            for k in range(len(cx)):
                cv2.circle(frame_bgr, (int(cx[k]), int(cy[k])), 2, (0, 0, 255), -1)
        # Text for the current frame
        cv2.putText(
            frame_bgr,
            f"t{t} --- Detections: {len(all_detections[t])}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
            cv2.LINE_AA
        )
        out.write(frame_bgr)
    out.release()
    print(f" Vidéo enregistrée dans {output_path}")

