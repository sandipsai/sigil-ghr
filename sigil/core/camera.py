"""
Camera abstraction module for video frame acquisition.
"""

from typing import Optional, Generator, Tuple
import cv2
import numpy as np


class CameraCapture:
    """Handles camera input streaming and frame capture."""

    def __init__(self, device_id: int = 0, frame_width: int = 640, frame_height: int = 480, fps: int = 30):
        self.device_id = device_id
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.fps = fps
        self._cap: Optional[cv2.VideoCapture] = None

    def start(self) -> bool:
        """Initialize camera device."""
        self._cap = cv2.VideoCapture(self.device_id)
        if not self._cap.isOpened():
            return False
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
        self._cap.set(cv2.CAP_PROP_FPS, self.fps)
        return True

    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Read a single frame (BGR format)."""
        if self._cap is None or not self._cap.isOpened():
            return False, None
        ret, frame = self._cap.read()
        return ret, frame

    def stream(self) -> Generator[np.ndarray, None, None]:
        """Generator streaming BGR frames continuously."""
        if not self.start():
            raise RuntimeError(f"Could not open camera device {self.device_id}")
        try:
            while True:
                ret, frame = self.read_frame()
                if not ret or frame is None:
                    break
                yield frame
        finally:
            self.stop()

    def stop(self) -> None:
        """Release camera hardware resource."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None
