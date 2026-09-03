"""
Temporal Sequence Resampling and Normalization Engine (Habib et al. 2025, Eq. 1-4).
"""

from typing import List
import numpy as np

from sigil.core.interfaces import ITemporalEncoder, SkeletonFrame, ResampledSequence


class TemporalEncoder(ITemporalEncoder):
    """Encodes raw variable-length gesture sequences into standardized 250-frame representations."""

    def __init__(self, target_frames: int = 250, padding_gamma: float = 0.125, image_length: float = 1.0):
        self.target_frames = target_frames
        self.padding_gamma = padding_gamma
        self.image_length = image_length

    def encode_sequence(self, sequence: List[SkeletonFrame]) -> ResampledSequence:
        """Process raw frame sequence through temporal interpolation and spatial centering."""
        valid_frames = [f for f in sequence if f.is_detected and len(f.joints) > 0]
        if not valid_frames:
            raise ValueError("Sequence contains no valid skeleton frames.")

        num_joints = len(valid_frames[0].joints)
        num_raw_frames = len(valid_frames)

        # Build coordinate array: Shape (T_raw, N_joints, 3)
        raw_coords = np.zeros((num_raw_frames, num_joints, 3), dtype=np.float32)
        for t_idx, frame in enumerate(valid_frames):
            for j_idx, joint in enumerate(frame.joints):
                raw_coords[t_idx, j_idx, :] = [joint.x, joint.y, joint.z]

        # 1. Temporal Interpolation to target_frames (T = 250) [Equation 1]
        if num_raw_frames == 1:
            resampled_coords = np.repeat(raw_coords, self.target_frames, axis=0)
        else:
            orig_timeline = np.linspace(0.0, 1.0, num_raw_frames)
            target_timeline = np.linspace(0.0, 1.0, self.target_frames)
            resampled_coords = np.zeros((self.target_frames, num_joints, 3), dtype=np.float32)

            for j in range(num_joints):
                for c in range(3):
                    resampled_coords[:, j, c] = np.interp(
                        target_timeline,
                        orig_timeline,
                        raw_coords[:, j, c]
                    )

        # 2. Spatial Centering [Equation 2]: g_i = g_i - mean(g_i)
        mean_center = np.mean(resampled_coords, axis=(0, 1))  # Zero-centered in 3D
        centered_coords = resampled_coords - mean_center

        # 3. Dynamic Zoom Level [Equation 4]: Z_i = max(g_i) - min(g_i) + gamma
        coord_max = np.max(centered_coords)
        coord_min = np.min(centered_coords)
        spatial_extent = float(coord_max - coord_min)
        zoom_level = spatial_extent + self.padding_gamma

        return ResampledSequence(
            raw_frames=valid_frames,
            coordinates=centered_coords,
            sequence_length=self.target_frames,
            mean_center=mean_center,
            spatial_extent=spatial_extent,
            zoom_level=zoom_level
        )
