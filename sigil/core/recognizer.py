"""
Real-time Gesture Recognizer module supporting neural multi-stream models and trajectory motion heuristics.
"""

from typing import List, Dict, Optional, Any, Tuple
import numpy as np

from sigil.core.interfaces import IGestureRecognizer, GesturePrediction, ResampledSequence
from sigil.core.models import HAS_TORCH, NumpyMultiStreamEnsembleNet

if HAS_TORCH:
    import torch
    import torch.nn.functional as F
    from sigil.core.models import MultiStreamEnsembleNet

# Standard DHG1428 Gesture Labels (14 Coarse / 28 Fine Gestures)
DEFAULT_GESTURE_NAMES: Dict[int, str] = {
    0: "Grab",
    1: "Tap",
    2: "Expand",
    3: "Pinch",
    4: "Rotation Clockwise",
    5: "Rotation Counter-Clockwise",
    6: "Swipe Right",
    7: "Swipe Left",
    8: "Swipe Up",
    9: "Swipe Down",
    10: "Swipe X",
    11: "Swipe V",
    12: "Swipe +",
    13: "Shake"
}

# ── Sensitivity constants ────────────────────────────────────────────────────
#
# These are the knobs that prevent false positives from:
#   • raising the hand toward the camera  (ONSET_SKIP_FRAMES)
#   • small micro-movements               (MIN_SWIPE_DISPLACEMENT)
#   • wobbly non-directional motion       (DIRECTION_CONSISTENCY_RATIO)
#
# Raise MIN_SWIPE_DISPLACEMENT / DIRECTION_CONSISTENCY_RATIO to make the
# recognizer *stricter* (fewer false positives, higher effort needed).
# Lower them to make it *more sensitive* (easier to trigger but noisier).

ONSET_SKIP_FRAMES: int = 20         # Ignore first N frames (approach phase)
MIN_SWIPE_DISPLACEMENT: float = 0.10  # Minimum normalised displacement for swipe
DIRECTION_CONSISTENCY_RATIO: float = 0.70   # ≥70 % of frame deltas must agree
MIN_SPREAD_DELTA: float = 0.05        # Minimum spread change for Expand / Pinch
MIN_SHAKE_DIST: float = 0.06          # Minimum displacement for Shake / Tap


def classify_trajectory_heuristically(resampled_seq: ResampledSequence) -> Tuple[int, float]:
    """
    Classifies a resampled 250-frame 3D skeleton trajectory using spatial dynamics.

    False-positive hardening applied at three levels:

    1. **Onset skip** – the first ONSET_SKIP_FRAMES frames are discarded.  The
       hand-raise phase (where the user moves their hand into camera view) is
       concentrated in those early frames and was the primary source of
       erroneous Swipe Up / Swipe Down predictions.

    2. **Minimum displacement gate** – the net displacement across the *active*
       window must reach MIN_SWIPE_DISPLACEMENT (normalised units) before a
       Swipe can be reported.  Tiny approach movements stay below this.

    3. **Direction consistency gate** – for horizontal/vertical swipes, at
       least DIRECTION_CONSISTENCY_RATIO of all frame-to-frame deltas must
       point in the dominant direction.  A straight, intentional swipe is
       consistent; a wobbly hand-raise is not.
    """
    coords = resampled_seq.coordinates  # Shape: (250, 21, 3)

    # ── 1. Skip onset frames ────────────────────────────────────────────────
    active = coords[ONSET_SKIP_FRAMES:]  # shape (230, 21, 3)

    # Wrist joint (index 0) displacement start → end in the active window
    dx = float(active[-1, 0, 0] - active[0, 0, 0])
    dy = float(active[-1, 0, 1] - active[0, 0, 1])
    dist = float(np.sqrt(dx ** 2 + dy ** 2))

    # ── 2. Direction consistency (frame-to-frame deltas for wrist) ───────────
    wrist_x = active[:, 0, 0]   # shape (230,)
    wrist_y = active[:, 0, 1]

    dx_deltas = np.diff(wrist_x)   # shape (229,)
    dy_deltas = np.diff(wrist_y)

    n_frames = len(dx_deltas)
    x_positive_ratio = float(np.sum(dx_deltas > 0)) / n_frames
    x_negative_ratio = float(np.sum(dx_deltas < 0)) / n_frames
    y_positive_ratio = float(np.sum(dy_deltas > 0)) / n_frames
    y_negative_ratio = float(np.sum(dy_deltas < 0)) / n_frames

    # ── 3. Fingertip spread change ───────────────────────────────────────────
    fingertip_indices = [4, 8, 12, 16, 20]
    spread_start = float(np.mean([np.linalg.norm(active[0, tip] - active[0, 0]) for tip in fingertip_indices]))
    spread_end   = float(np.mean([np.linalg.norm(active[-1, tip] - active[-1, 0]) for tip in fingertip_indices]))
    d_spread = spread_end - spread_start

    # ── Classification rules (most specific first) ───────────────────────────
    # Swipe Right: net rightward, consistent, large enough
    if (dx > MIN_SWIPE_DISPLACEMENT
            and abs(dx) > 1.1 * abs(dy)
            and x_positive_ratio >= DIRECTION_CONSISTENCY_RATIO):
        return 6, 0.92  # Swipe Right

    # Swipe Left
    if (dx < -MIN_SWIPE_DISPLACEMENT
            and abs(dx) > 1.1 * abs(dy)
            and x_negative_ratio >= DIRECTION_CONSISTENCY_RATIO):
        return 7, 0.92  # Swipe Left

    # Swipe Up (in image coords y increases downward, so hand moving up → dy < 0)
    if (dy < -MIN_SWIPE_DISPLACEMENT
            and abs(dy) > 1.1 * abs(dx)
            and y_negative_ratio >= DIRECTION_CONSISTENCY_RATIO):
        return 8, 0.92  # Swipe Up

    # Swipe Down
    if (dy > MIN_SWIPE_DISPLACEMENT
            and abs(dy) > 1.1 * abs(dx)
            and y_positive_ratio >= DIRECTION_CONSISTENCY_RATIO):
        return 9, 0.92  # Swipe Down

    # Expand / Pinch (fingertip spread)
    if d_spread > MIN_SPREAD_DELTA:
        return 2, 0.88  # Expand
    if d_spread < -MIN_SPREAD_DELTA:
        return 3, 0.88  # Pinch

    # General motion / Shake
    if dist > MIN_SHAKE_DIST:
        return 13, 0.80  # Shake

    # Stationary → Tap
    return 1, 0.85


class GestureRecognizer(IGestureRecognizer):
    """Gesture Recognition Engine using MultiStreamEnsembleNet and Trajectory Heuristics."""

    def __init__(
        self,
        model: Optional[Any] = None,
        num_classes: int = 14,
        gesture_names: Optional[Dict[int, str]] = None,
        device: str = "cpu",
        use_heuristics: bool = True
    ):
        self.num_classes = num_classes
        self.gesture_names = gesture_names or DEFAULT_GESTURE_NAMES
        self.use_torch = HAS_TORCH
        self.use_heuristics = use_heuristics

        if self.use_torch:
            self.device = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
            if model is None:
                self.model = MultiStreamEnsembleNet(num_classes=num_classes, num_streams=3, pretrained=False)
                self.model.eval()
            else:
                self.model = model.to(self.device)
                self.model.eval()
        else:
            self.model = model or NumpyMultiStreamEnsembleNet(num_classes=num_classes, num_streams=3)

    def predict(self, view_images: List[np.ndarray]) -> GesturePrediction:
        """Classify gesture from multi-view spatiotemporal images."""
        if not view_images:
            raise ValueError("No view images provided for recognition.")

        if self.use_torch:
            tensor_views: List[torch.Tensor] = []
            for img in view_images:
                img_rgb = img[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
                tensor_img = torch.from_numpy(img_rgb).unsqueeze(0).to(self.device)
                tensor_views.append(tensor_img)

            with torch.no_grad():
                stream_logits, tuner_logits = self.model(tensor_views)
                tuner_probs = F.softmax(tuner_logits, dim=-1).cpu().numpy()[0]
                stream_probs = [F.softmax(l, dim=-1).cpu().numpy()[0] for l in stream_logits]
        else:
            stream_probs, tuner_probs = self.model.predict(view_images)

        best_id = int(np.argmax(tuner_probs))
        confidence = float(tuner_probs[best_id])
        name = self.gesture_names.get(best_id, f"Gesture_{best_id}")

        return GesturePrediction(
            gesture_id=best_id,
            gesture_name=name,
            confidence=confidence,
            stream_probabilities=stream_probs,
            tuner_probabilities=tuner_probs
        )

    def predict_sequence(self, resampled_seq: ResampledSequence, view_images: List[np.ndarray]) -> GesturePrediction:
        """Classify gesture combining multi-view images and spatiotemporal sequence dynamics."""
        # Standard model prediction
        model_prediction = self.predict(view_images)

        # If heuristics enabled and model is uncalibrated (confidence < 0.30), apply trajectory dynamics
        if self.use_heuristics and model_prediction.confidence < 0.30:
            heuristic_id, heuristic_conf = classify_trajectory_heuristically(resampled_seq)
            name = self.gesture_names.get(heuristic_id, f"Gesture_{heuristic_id}")

            # Construct blended probability distribution
            tuner_probs = np.zeros(self.num_classes, dtype=np.float32)
            tuner_probs[heuristic_id] = heuristic_conf
            remaining = (1.0 - heuristic_conf) / (self.num_classes - 1)
            for i in range(self.num_classes):
                if i != heuristic_id:
                    tuner_probs[i] = remaining

            return GesturePrediction(
                gesture_id=heuristic_id,
                gesture_name=name,
                confidence=heuristic_conf,
                stream_probabilities=model_prediction.stream_probabilities,
                tuner_probabilities=tuner_probs
            )

        return model_prediction
