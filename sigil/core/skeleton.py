"""
Skeleton Provider implementations with strict Hand Isolation filtering (excluding shoulders/face)
supporting MediaPipe and OpenCV Contour Fallback.
"""

import time
from typing import List, Optional
import numpy as np

from sigil.core.interfaces import ISkeletonProvider, SkeletonFrame, Joint3D

# MediaPipe Hand Joint Names (21 Joints)
MEDIAPIPE_JOINT_NAMES = [
    "WRIST",
    "THUMB_CMC", "THUMB_MCP", "THUMB_IP", "THUMB_TIP",
    "INDEX_FINGER_MCP", "INDEX_FINGER_PIP", "INDEX_FINGER_DIP", "INDEX_FINGER_TIP",
    "MIDDLE_FINGER_MCP", "MIDDLE_FINGER_PIP", "MIDDLE_FINGER_DIP", "MIDDLE_FINGER_TIP",
    "RING_FINGER_MCP", "RING_FINGER_PIP", "RING_FINGER_DIP", "RING_FINGER_TIP",
    "PINKY_MCP", "PINKY_PIP", "PINKY_DIP", "PINKY_TIP"
]


class MediaPipeSkeletonProvider(ISkeletonProvider):
    """Skeleton provider with strict hand geometry & palm isolation filtering."""

    def __init__(
        self,
        static_image_mode: bool = False,
        max_num_hands: int = 1,
        min_detection_confidence: float = 0.6,
        min_tracking_confidence: float = 0.6
    ):
        self.frame_counter = 0
        self.api_type = None
        self.detector = None

        # 1. MediaPipe Solutions API (Strictest Palm & Hand Isolation)
        try:
            import mediapipe as mp
            if hasattr(mp, "solutions") and hasattr(mp.solutions, "hands"):
                self.detector = mp.solutions.hands.Hands(
                    static_image_mode=static_image_mode,
                    max_num_hands=max_num_hands,
                    min_detection_confidence=min_detection_confidence,
                    min_tracking_confidence=min_tracking_confidence
                )
                self.api_type = "solutions"
        except Exception:
            pass

        if self.api_type is None:
            self.api_type = "opencv"

    def process_frame(self, frame: np.ndarray) -> Optional[SkeletonFrame]:
        """Extract 3D joint coordinates from image frame with strict hand isolation."""
        self.frame_counter += 1
        timestamp = time.time()

        if frame is None or frame.size == 0:
            return SkeletonFrame(frame_index=self.frame_counter, timestamp=timestamp, joints=[], is_detected=False)

        import cv2

        # 1. MediaPipe Solutions API (Strict Palm Detector)
        if self.api_type == "solutions" and self.detector is not None:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.detector.process(rgb_frame)
            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                joints: List[Joint3D] = []
                for idx, lm in enumerate(hand_landmarks.landmark):
                    name = MEDIAPIPE_JOINT_NAMES[idx] if idx < len(MEDIAPIPE_JOINT_NAMES) else f"JOINT_{idx}"
                    joints.append(Joint3D(x=float(lm.x), y=float(lm.y), z=float(lm.z), name=name))
                return SkeletonFrame(frame_index=self.frame_counter, timestamp=timestamp, joints=joints, is_detected=True)

        # 2. Advanced OpenCV Hand Isolation Fallback
        # Filters out faces, shoulders, and background body parts using geometric hand constraints
        h, w, _ = frame.shape
        img_area = float(w * h)

        ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
        lower_skin = np.array([0, 135, 80], dtype=np.uint8)
        upper_skin = np.array([255, 170, 125], dtype=np.uint8)
        mask = cv2.inRange(ycrcb, lower_skin, upper_skin)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.erode(mask, kernel, iterations=1)
        mask = cv2.dilate(mask, kernel, iterations=2)
        mask = cv2.GaussianBlur(mask, (5, 5), 0)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return SkeletonFrame(frame_index=self.frame_counter, timestamp=timestamp, joints=[], is_detected=False)

        valid_hand_contours = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            # Hand size constraint: Between 1.0% and 15% of image area (excludes small noise and large faces/torso)
            area_ratio = area / img_area
            if 0.010 <= area_ratio <= 0.15:
                hull = cv2.convexHull(cnt)
                hull_area = cv2.contourArea(hull)
                solidity = float(area) / (hull_area + 1e-5)
                # Hands with extended fingers have lower solidity (< 0.85), whereas faces/shoulders have high solidity (> 0.85)
                if 0.35 <= solidity <= 0.82:
                    valid_hand_contours.append((cnt, area))

        if not valid_hand_contours:
            return SkeletonFrame(frame_index=self.frame_counter, timestamp=timestamp, joints=[], is_detected=False)

        # Select most dynamic hand-like contour
        best_contour, _ = max(valid_hand_contours, key=lambda item: item[1])

        M = cv2.moments(best_contour)
        if M["m00"] == 0:
            return SkeletonFrame(frame_index=self.frame_counter, timestamp=timestamp, joints=[], is_detected=False)

        cx = float(M["m10"] / M["m00"]) / w
        cy = float(M["m01"] / M["m00"]) / h

        hull = cv2.convexHull(best_contour, returnPoints=True)
        hull_pts = [pt[0] for pt in hull]
        hull_pts.sort(key=lambda p: p[1])
        top_pts = hull_pts[:5]

        joints: List[Joint3D] = []
        joints.append(Joint3D(x=cx, y=min(cy + 0.15, 1.0), z=0.0, name="WRIST"))

        base_fingers = [
            (cx - 0.10, cy - 0.05),
            (cx - 0.05, cy - 0.12),
            (cx, cy - 0.15),
            (cx + 0.05, cy - 0.12),
            (cx + 0.10, cy - 0.05)
        ]

        for f_idx in range(5):
            target_pt = top_pts[f_idx % len(top_pts)]
            tx, ty = float(target_pt[0]) / w, float(target_pt[1]) / h
            bx, by = base_fingers[f_idx]

            for seg in range(1, 5):
                alpha = seg / 4.0
                jx = bx + (tx - bx) * alpha
                jy = by + (ty - by) * alpha
                name_idx = 1 + f_idx * 4 + (seg - 1)
                name = MEDIAPIPE_JOINT_NAMES[name_idx]
                joints.append(Joint3D(x=jx, y=jy, z=0.0, name=name))

        return SkeletonFrame(
            frame_index=self.frame_counter,
            timestamp=timestamp,
            joints=joints,
            is_detected=True
        )

    def reset(self) -> None:
        self.frame_counter = 0


class SyntheticSkeletonProvider(ISkeletonProvider):
    """Synthetic skeleton generator for testing, offline evaluation, and demos."""

    def __init__(self, gesture_type: str = "swipe_right"):
        self.gesture_type = gesture_type
        self.frame_counter = 0

    def process_frame(self, frame: Optional[np.ndarray] = None) -> SkeletonFrame:
        self.frame_counter += 1
        timestamp = time.time()

        t = self.frame_counter * 0.05
        dx = np.sin(t) * 0.2 if "swipe" in self.gesture_type else 0.0
        dy = np.cos(t) * 0.2 if "circle" in self.gesture_type else 0.0

        base_joints_2d = [
            (0.5, 0.8),
            (0.42, 0.72), (0.38, 0.65), (0.35, 0.58), (0.32, 0.52),
            (0.46, 0.60), (0.45, 0.48), (0.44, 0.38), (0.43, 0.30),
            (0.50, 0.58), (0.50, 0.45), (0.50, 0.35), (0.50, 0.26),
            (0.54, 0.60), (0.55, 0.48), (0.56, 0.38), (0.57, 0.30),
            (0.58, 0.65), (0.60, 0.55), (0.61, 0.48), (0.62, 0.42)
        ]

        joints: List[Joint3D] = []
        for idx, (bx, by) in enumerate(base_joints_2d):
            name = MEDIAPIPE_JOINT_NAMES[idx]
            is_tip = "TIP" in name
            multiplier = 1.5 if is_tip else 0.5
            x = bx + dx * multiplier
            y = by + dy * multiplier
            z = 0.05 * np.sin(t + idx * 0.1)
            joints.append(Joint3D(x=float(x), y=float(y), z=float(z), name=name))

        return SkeletonFrame(
            frame_index=self.frame_counter,
            timestamp=timestamp,
            joints=joints,
            is_detected=True
        )

    def reset(self) -> None:
        self.frame_counter = 0
