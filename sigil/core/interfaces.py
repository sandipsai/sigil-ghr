"""
SigilGHR Interfaces - Core abstraction definitions following SOLID principles.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
import numpy as np


@dataclass
class Joint3D:
    """3D Joint coordinate representation."""
    x: float
    y: float
    z: float
    name: str = ""


@dataclass
class SkeletonFrame:
    """Represents hand skeleton joints for a single temporal frame."""
    frame_index: int
    timestamp: float
    joints: List[Joint3D]  # 21 joints for standard MediaPipe hand format
    is_detected: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ResampledSequence:
    """Resampled and normalized skeleton gesture sequence (T frames)."""
    raw_frames: List[SkeletonFrame]
    coordinates: np.ndarray  # Shape: (T, N_joints, 3)
    sequence_length: int
    mean_center: np.ndarray
    spatial_extent: float
    zoom_level: float


@dataclass
class ViewOrientation:
    """Camera view orientation parameters."""
    name: str
    elevation: float  # Degrees
    azimuth: float    # Degrees


@dataclass
class GesturePrediction:
    """Prediction outcome from model inference."""
    gesture_id: int
    gesture_name: str
    confidence: float
    stream_probabilities: List[np.ndarray]
    tuner_probabilities: np.ndarray


class ISkeletonProvider(ABC):
    """Interface for hand tracking and 3D skeleton providers (SRP/DIP)."""

    @abstractmethod
    def process_frame(self, frame: np.ndarray) -> Optional[SkeletonFrame]:
        """Extract 3D hand joint coordinates from an image frame."""
        pass

    @abstractmethod
    def reset(self) -> None:
        """Reset internal tracking state."""
        pass


class ITemporalEncoder(ABC):
    """Interface for temporal sequence resampling and normalization (Eq. 1-4)."""

    @abstractmethod
    def encode_sequence(self, sequence: List[SkeletonFrame]) -> ResampledSequence:
        """Normalize, center, scale, and resample sequence to target T frames."""
        pass


class IFusionRenderer(ABC):
    """Interface for data-level fusion rendering (2D spatiotemporal images)."""

    @abstractmethod
    def render_views(
        self,
        resampled_seq: ResampledSequence,
        view_orientations: List[ViewOrientation],
        image_size: Tuple[int, int] = (224, 224)
    ) -> List[np.ndarray]:
        """Render static 2D RGB spatiotemporal images across specified VOs."""
        pass


class IGestureRecognizer(ABC):
    """Interface for model inference and gesture recognition."""

    @abstractmethod
    def predict(self, view_images: List[np.ndarray]) -> GesturePrediction:
        """Classify gesture from multi-view spatiotemporal images."""
        pass


class IPlugin(ABC):
    """Interface for application macro plugins."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Plugin identifier."""
        pass

    @abstractmethod
    def get_supported_gestures(self) -> List[str]:
        """Return list of supported gesture names."""
        pass

    @abstractmethod
    def execute(self, gesture_name: str, confidence: float) -> bool:
        """Execute macro action associated with the gesture."""
        pass
