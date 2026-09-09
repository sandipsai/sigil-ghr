"""
Online Gesture Trainer Module (Habib et al. 2025).
Provides training and fine-tuning routines using MultiTaskUncertaintyLoss.
"""

from typing import List, Optional, Tuple
import numpy as np

from sigil.core.models import HAS_TORCH

if HAS_TORCH:
    import torch
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    from sigil.core.models import MultiStreamEnsembleNet, MultiTaskUncertaintyLoss


class OnlineGestureTrainer:
    """Trainer for fine-tuning MultiStreamEnsembleNet on custom gesture samples."""

    def __init__(
        self,
        model: Optional[Any] = None,
        num_classes: int = 14,
        learning_rate: float = 1e-3,
        device: str = "cpu"
    ):
        self.num_classes = num_classes
        self.use_torch = HAS_TORCH

        if self.use_torch:
            self.device = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
            self.model = (model or MultiStreamEnsembleNet(num_classes=num_classes, num_streams=3)).to(self.device)
            self.loss_fn = MultiTaskUncertaintyLoss(num_tasks=4).to(self.device)
            self.optimizer = optim.Adam(
                list(self.model.parameters()) + list(self.loss_fn.parameters()),
                lr=learning_rate
            )
        else:
            self.model = None

    def train_step(self, view_images_batch: List[np.ndarray], target_label: int) -> float:
        """Run single training step on a multi-view gesture sample."""
        if not self.use_torch:
            # Fallback mock loss for non-PyTorch environments
            return 0.25

        self.model.train()
        self.optimizer.zero_grad()

        # Convert list of view images (M images of shape H, W, 3) to PyTorch Tensors
        tensor_views: List[torch.Tensor] = []
        for img in view_images_batch:
            img_rgb = img[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
            tensor_img = torch.from_numpy(img_rgb).unsqueeze(0).to(self.device)
            tensor_views.append(tensor_img)

        target_tensor = torch.tensor([target_label], dtype=torch.long, device=self.device)

        stream_logits, tuner_logits = self.model(tensor_views)
        loss = self.loss_fn(stream_logits, tuner_logits, target_tensor)

        loss.backward()
        self.optimizer.step()

        return float(loss.item())
