"""
Enhanced Data-Level Fusion Engine - 2D Spatiotemporal Image Generation (Habib et al. 2025).
"""

from typing import List, Tuple, Dict
import numpy as np
import cv2

from sigil.core.interfaces import IFusionRenderer, ResampledSequence, ViewOrientation

# Default 6 View Orientations from Paper (Fig. 3 & Table 2)
DEFAULT_VIEW_ORIENTATIONS = [
    ViewOrientation(name="custom", elevation=30.0, azimuth=45.0),
    ViewOrientation(name="top_down", elevation=90.0, azimuth=0.0),
    ViewOrientation(name="front_away", elevation=0.0, azimuth=0.0),
    ViewOrientation(name="side_left", elevation=0.0, azimuth=-90.0),
    ViewOrientation(name="front_to", elevation=0.0, azimuth=180.0),
    ViewOrientation(name="side_right", elevation=0.0, azimuth=90.0),
]

# Color Palette for 5 Fingers (BGR)
FINGER_COLORS_BGR: Dict[str, Tuple[int, int, int]] = {
    "THUMB": (0, 255, 0),       # Green
    "INDEX": (255, 255, 0),     # Cyan
    "MIDDLE": (255, 0, 0),      # Blue
    "RING": (255, 0, 255),      # Magenta
    "PINKY": (0, 255, 255),     # Yellow
    "WRIST": (255, 255, 255),   # White
}

# MediaPipe Fingertip Joint Indices
FINGERTIP_INDICES = [4, 8, 12, 16, 20]  # Thumb, Index, Middle, Ring, Pinky

# MediaPipe Hand Bone Connections: (joint_a, joint_b, finger_group)
HAND_CONNECTIONS = [
    (0, 1, "THUMB"), (1, 2, "THUMB"), (2, 3, "THUMB"), (3, 4, "THUMB"),
    (0, 5, "INDEX"), (5, 6, "INDEX"), (6, 7, "INDEX"), (7, 8, "INDEX"),
    (0, 9, "MIDDLE"), (9, 10, "MIDDLE"), (10, 11, "MIDDLE"), (11, 12, "MIDDLE"),
    (0, 13, "RING"), (13, 14, "RING"), (14, 15, "RING"), (15, 16, "RING"),
    (0, 17, "PINKY"), (17, 18, "PINKY"), (18, 19, "PINKY"), (19, 20, "PINKY"),
    (5, 9, "WRIST"), (9, 13, "WRIST"), (13, 17, "WRIST")
]


def rotate_3d_points(points: np.ndarray, elevation_deg: float, azimuth_deg: float) -> np.ndarray:
    """Rotate 3D points by elevation and azimuth angles."""
    el = np.radians(elevation_deg)
    az = np.radians(azimuth_deg)

    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(el), -np.sin(el)],
        [0, np.sin(el), np.cos(el)]
    ], dtype=np.float32)

    Ry = np.array([
        [np.cos(az), 0, np.sin(az)],
        [0, 1, 0],
        [-np.sin(az), 0, np.cos(az)]
    ], dtype=np.float32)

    R = Ry @ Rx
    return points @ R.T


class SpatiotemporalFusionRenderer(IFusionRenderer):
    """Renders static 2D spatiotemporal RGB images from 3D skeleton sequences."""

    def render_views(
        self,
        resampled_seq: ResampledSequence,
        view_orientations: List[ViewOrientation] = None,
        image_size: Tuple[int, int] = (224, 224)
    ) -> List[np.ndarray]:
        """Render static 2D RGB spatiotemporal images across view orientations."""
        if view_orientations is None:
            view_orientations = DEFAULT_VIEW_ORIENTATIONS[:3]

        coords = resampled_seq.coordinates  # Shape: (T=250, N_joints=21, 3)
        T, N, _ = coords.shape
        w, h = image_size
        zoom = max(resampled_seq.zoom_level, 1e-5)

        rendered_images: List[np.ndarray] = []

        for vo in view_orientations:
            canvas = np.zeros((h, w, 3), dtype=np.uint8)
            rotated_coords = rotate_3d_points(coords, vo.elevation, vo.azimuth)

            # Map zero-centered coordinates [-zoom/2, +zoom/2] -> [0, w] and [0, h]
            proj_x = ((rotated_coords[:, :, 0] / zoom) + 0.5) * (w - 1)
            proj_y = ((rotated_coords[:, :, 1] / zoom) + 0.5) * (h - 1)

            # 1. Render Temporal Channel: Fingertip motion trails with alpha decay (tau = 1 ... T-1)
            overlay = canvas.copy()
            # Step size to draw continuous smooth motion trails
            step = max(1, T // 50)
            for tau in range(0, T - step, step):
                alpha = 0.20 + 0.80 * (tau / float(T))
                for f_idx, tip_idx in enumerate(FINGERTIP_INDICES):
                    finger_name = list(FINGER_COLORS_BGR.keys())[f_idx]
                    color = FINGER_COLORS_BGR[finger_name]

                    pt_curr = (int(proj_x[tau, tip_idx]), int(proj_y[tau, tip_idx]))
                    pt_next = (int(proj_x[tau + step, tip_idx]), int(proj_y[tau + step, tip_idx]))

                    if 0 <= pt_curr[0] < w and 0 <= pt_curr[1] < h and 0 <= pt_next[0] < w and 0 <= pt_next[1] < h:
                        thickness = 2 if tau < T // 2 else 3
                        cv2.line(overlay, pt_curr, pt_next, color, thickness=thickness, lineType=cv2.LINE_AA)

            cv2.addWeighted(overlay, 0.8, canvas, 0.2, 0, canvas)

            # 2. Render Spatial Channel: Final Hand Pose at frame T-1
            final_frame_x = proj_x[-1]
            final_frame_y = proj_y[-1]

            # Draw bones
            for idx_a, idx_b, finger_grp in HAND_CONNECTIONS:
                if idx_a < N and idx_b < N:
                    pt_a = (int(final_frame_x[idx_a]), int(final_frame_y[idx_a]))
                    pt_b = (int(final_frame_x[idx_b]), int(final_frame_y[idx_b]))
                    color = FINGER_COLORS_BGR.get(finger_grp, (255, 255, 255))
                    cv2.line(canvas, pt_a, pt_b, color, thickness=3, lineType=cv2.LINE_AA)

            # Draw joint dots
            for j_idx in range(N):
                pt = (int(final_frame_x[j_idx]), int(final_frame_y[j_idx]))
                cv2.circle(canvas, pt, radius=3, color=(255, 255, 255), thickness=-1, lineType=cv2.LINE_AA)

            rendered_images.append(canvas)

        return rendered_images
