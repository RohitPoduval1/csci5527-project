"""Grad-CAM helpers for the FER2013 vanilla CNN notebook.

This module is intentionally notebook-friendly: import the pieces you need and
use them directly with ``model_vanilla`` or ``model_best``.
"""

from __future__ import annotations

from typing import Iterable, Optional, Sequence

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


def get_last_conv_layer(model: nn.Module) -> nn.Conv2d:
    """Return the last convolutional layer in a model."""
    last_conv: Optional[nn.Conv2d] = None
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            last_conv = module
    if last_conv is None:
        raise ValueError("Grad-CAM needs at least one nn.Conv2d layer.")
    return last_conv


def _to_class_index_tensor(
    class_idx: Optional[Sequence[int] | torch.Tensor | int],
    batch_size: int,
    device: torch.device,
) -> Optional[torch.Tensor]:
    """Normalize class indices into a 1D tensor on the right device."""
    if class_idx is None:
        return None
    if isinstance(class_idx, int):
        return torch.full((batch_size,), class_idx, device=device, dtype=torch.long)
    if isinstance(class_idx, torch.Tensor):
        return class_idx.to(device=device, dtype=torch.long).view(-1)
    if isinstance(class_idx, Iterable):
        return torch.tensor(list(class_idx), device=device, dtype=torch.long).view(-1)
    raise TypeError("class_idx must be None, an int, a tensor, or a sequence of ints.")


class GradCAM:
    """Minimal Grad-CAM implementation for CNN classifiers."""

    def __init__(self, model: nn.Module, target_layer: Optional[nn.Module] = None) -> None:
        self.model = model
        self.target_layer = target_layer if target_layer is not None else get_last_conv_layer(model)
        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None
        self._handles = [self.target_layer.register_forward_hook(self._forward_hook)]

    def _save_gradient(self, grad: torch.Tensor) -> None:
        self.gradients = grad.detach()

    def _forward_hook(self, module: nn.Module, inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
        if not isinstance(output, torch.Tensor):
            raise TypeError("Grad-CAM target layer must return a tensor output.")
        self.activations = output.detach()
        output.register_hook(self._save_gradient)

    def close(self) -> None:
        """Remove hooks and release references."""
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self.activations = None
        self.gradients = None

    def __enter__(self) -> "GradCAM":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def generate(
        self,
        input_tensor: torch.Tensor,
        class_idx: Optional[Sequence[int] | torch.Tensor | int] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute Grad-CAM heatmaps.

        Returns:
            cam_map: shape ``[batch, height, width]`` normalized to ``[0, 1]``
            class_idx: shape ``[batch]``
            logits: detached model outputs
        """
        if input_tensor.ndim != 4:
            raise ValueError("input_tensor must have shape [batch, channels, height, width].")

        was_training = self.model.training
        self.model.eval()
        self.model.zero_grad(set_to_none=True)

        logits = self.model(input_tensor)
        class_tensor = _to_class_index_tensor(class_idx, logits.size(0), logits.device)
        if class_tensor is None:
            class_tensor = logits.argmax(dim=1)
        if class_tensor.numel() != logits.size(0):
            raise ValueError("class_idx must provide one class index per batch element.")

        target_scores = logits.gather(1, class_tensor.view(-1, 1)).sum()
        target_scores.backward()

        if self.activations is None or self.gradients is None:
            raise RuntimeError("Grad-CAM hooks did not capture activations and gradients.")

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam_map = (weights * self.activations).sum(dim=1, keepdim=True)
        cam_map = F.relu(cam_map)
        cam_map = F.interpolate(
            cam_map,
            size=input_tensor.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        cam_min = cam_map.amin(dim=(2, 3), keepdim=True)
        cam_max = cam_map.amax(dim=(2, 3), keepdim=True)
        cam_map = (cam_map - cam_min) / (cam_max - cam_min + 1e-8)

        if was_training:
            self.model.train()

        return cam_map.squeeze(1).detach(), class_tensor.detach(), logits.detach()


def denormalize_grayscale(
    image_tensor: torch.Tensor,
    mean: float = 0.5,
    std: float = 0.5,
) -> np.ndarray:
    """Undo the FER normalization and return a 2D numpy image in [0, 1]."""
    image = image_tensor.detach().cpu().float()
    if image.ndim == 4:
        image = image[0]
    if image.ndim == 3:
        image = image[0]
    image = (image * std) + mean
    return image.clamp(0.0, 1.0).numpy()


def overlay_cam_on_grayscale(
    image: np.ndarray,
    cam_map: np.ndarray,
    alpha: float = 0.35,
    cmap: str = "jet",
) -> np.ndarray:
    """Blend a Grad-CAM heatmap onto a grayscale image."""
    image = np.asarray(image, dtype=np.float32)
    cam_map = np.asarray(cam_map, dtype=np.float32)

    image_rgb = np.repeat(image[..., None], 3, axis=2)
    color_map = cm.get_cmap(cmap)(cam_map)[..., :3]
    overlay = ((1.0 - alpha) * image_rgb) + (alpha * color_map)
    return np.clip(overlay, 0.0, 1.0)


def visualize_gradcam_samples(
    model: nn.Module,
    dataset,
    class_names: Sequence[str],
    device: torch.device | str,
    indices: Optional[Sequence[int]] = None,
    num_samples: int = 6,
    target_layer: Optional[nn.Module] = None,
    mean: float = 0.5,
    std: float = 0.5,
    alpha: float = 0.35,
    cmap: str = "jet",
) -> plt.Figure:
    """Plot original images, Grad-CAM maps, and overlays for dataset samples."""
    if indices is None:
        num_samples = min(num_samples, len(dataset))
        indices = list(range(num_samples))
    if not indices:
        raise ValueError("indices must contain at least one dataset index.")

    device = torch.device(device)
    was_training = model.training
    model.eval()

    with GradCAM(model, target_layer=target_layer) as gradcam:
        fig, axes = plt.subplots(len(indices), 3, figsize=(12, 4 * len(indices)))
        if len(indices) == 1:
            axes = np.expand_dims(axes, axis=0)

        for row, index in enumerate(indices):
            image, label = dataset[index]
            if torch.is_tensor(label):
                label_idx = int(label.item())
            else:
                label_idx = int(label)

            input_tensor = image.unsqueeze(0).to(device)
            cam_map, pred_idx, logits = gradcam.generate(input_tensor)
            probs = logits.softmax(dim=1)[0]

            pred_idx_int = int(pred_idx.item())
            pred_prob = float(probs[pred_idx_int].item())
            image_np = denormalize_grayscale(image, mean=mean, std=std)
            cam_np = cam_map[0].cpu().numpy()
            overlay_np = overlay_cam_on_grayscale(image_np, cam_np, alpha=alpha, cmap=cmap)

            axes[row, 0].imshow(image_np, cmap="gray")
            axes[row, 0].set_title(f"Original\nTrue: {class_names[label_idx]}")
            axes[row, 0].axis("off")

            axes[row, 1].imshow(cam_np, cmap=cmap)
            axes[row, 1].set_title(
                f"Grad-CAM\nPred: {class_names[pred_idx_int]} ({pred_prob:.1%})"
            )
            axes[row, 1].axis("off")

            axes[row, 2].imshow(overlay_np)
            axes[row, 2].set_title("Overlay")
            axes[row, 2].axis("off")

        plt.tight_layout()

    if was_training:
        model.train()

    return fig


__all__ = [
    "GradCAM",
    "denormalize_grayscale",
    "get_last_conv_layer",
    "overlay_cam_on_grayscale",
    "visualize_gradcam_samples",
]

