import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import torch
import torch.nn.functional as F


def get_default_target_layer(model):
    """Return the last convolution layer in a CNN."""
    last_conv = None
    for module in model.modules():
        if isinstance(module, torch.nn.Conv2d):
            last_conv = module
    if last_conv is None:
        raise ValueError("Grad-CAM needs at least one Conv2d layer.")
    return last_conv

def gradcam_one_image(model, image_tensor, target_layer, class_names=None, device="cuda"):
    """
    Minimal Grad-CAM for a FER-style CNN.

    Args:
        model: trained CNN
        image_tensor: single image, shape [1, H, W] or [C, H, W]
        target_layer: the conv layer to hook
        class_names: optional list of class names
        device: "cuda" or "cpu"

    Returns:
        cam: heatmap in [0,1], shape [H, W]
        pred_idx: predicted class index
        pred_prob: predicted probability
        image_np: denormalized grayscale image in [0,1]
    """
    model.eval()
    model.to(device)

    # Ensure batch dimension
    if image_tensor.ndim == 3:
        input_tensor = image_tensor.unsqueeze(0).to(device)
    else:
        raise ValueError("image_tensor should have shape [C, H, W]")

    activations = None
    gradients = None

    def forward_hook(module, inp, out):
        nonlocal activations
        activations = out

    def backward_hook(module, grad_in, grad_out):
        nonlocal gradients
        gradients = grad_out[0]

    h1 = target_layer.register_forward_hook(forward_hook)
    h2 = target_layer.register_full_backward_hook(backward_hook)

    # Forward
    logits = model(input_tensor)
    probs = torch.softmax(logits, dim=1)
    pred_idx = logits.argmax(dim=1).item()
    pred_prob = probs[0, pred_idx].item()

    # Backward on predicted class
    model.zero_grad()
    logits[0, pred_idx].backward()

    # Grad-CAM
    weights = gradients.mean(dim=(2, 3), keepdim=True)          # [1, C, 1, 1]
    cam = (weights * activations).sum(dim=1, keepdim=True)      # [1, 1, h, w]
    cam = F.relu(cam)
    cam = F.interpolate(cam, size=input_tensor.shape[-2:], mode="bilinear", align_corners=False)
    cam = cam.squeeze().detach().cpu()

    # Normalize to [0,1]
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    cam = cam.numpy()

    # FER images are often normalized with mean=0.5, std=0.5
    image_np = input_tensor[0, 0].detach().cpu() * 0.5 + 0.5
    image_np = image_np.clamp(0, 1).numpy()

    h1.remove()
    h2.remove()

    if class_names is not None:
        print(f"Prediction: {class_names[pred_idx]} ({pred_prob:.3f})")
    else:
        print(f"Prediction: class {pred_idx} ({pred_prob:.3f})")

    return cam, pred_idx, pred_prob, image_np


def show_gradcam(image_np, cam, alpha=0.35, cmap="jet"):
    """Overlay Grad-CAM heatmap on grayscale image."""
    image_rgb = np.repeat(image_np[..., None], 3, axis=2)
    heatmap = cm.get_cmap(cmap)(cam)[..., :3]
    overlay = (1 - alpha) * image_rgb + alpha * heatmap
    overlay = np.clip(overlay, 0, 1)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].imshow(image_np, cmap="gray")
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(cam, cmap=cmap)
    axes[1].set_title("Grad-CAM")
    axes[1].axis("off")

    axes[2].imshow(overlay)
    axes[2].set_title("Overlay")
    axes[2].axis("off")

    plt.tight_layout()
    plt.show()


def visualize_gradcam_samples(
    model,
    dataset,
    class_names,
    device,
    indices=None,
    num_samples=6,
    target_layer=None,
    alpha=0.35,
    cmap="jet",
):
    """Plot original images, Grad-CAM heatmaps, and overlays for sample indices."""
    if target_layer is None:
        target_layer = get_default_target_layer(model)

    if indices is None:
        indices = list(range(min(num_samples, len(dataset))))

    fig, axes = plt.subplots(len(indices), 3, figsize=(12, 4 * len(indices)))
    if len(indices) == 1:
        axes = np.expand_dims(axes, axis=0)

    for row, index in enumerate(indices):
        image, label = dataset[index]
        cam, pred_idx, pred_prob, image_np = gradcam_one_image(
            model=model,
            image_tensor=image,
            target_layer=target_layer,
            class_names=class_names,
            device=device,
        )

        image_rgb = np.repeat(image_np[..., None], 3, axis=2)
        heatmap = cm.get_cmap(cmap)(cam)[..., :3]
        overlay = np.clip((1 - alpha) * image_rgb + alpha * heatmap, 0, 1)

        label_idx = int(label.item()) if torch.is_tensor(label) else int(label)

        axes[row, 0].imshow(image_np, cmap="gray")
        axes[row, 0].set_title(f"Original\nTrue: {class_names[label_idx]}")
        axes[row, 0].axis("off")

        axes[row, 1].imshow(cam, cmap=cmap)
        axes[row, 1].set_title(f"Grad-CAM\nPred: {class_names[pred_idx]} ({pred_prob:.1%})")
        axes[row, 1].axis("off")

        axes[row, 2].imshow(overlay)
        axes[row, 2].set_title("Overlay")
        axes[row, 2].axis("off")

    plt.tight_layout()
    return fig


def compare_gradcam_methods(
    model,
    dataset,
    class_names,
    device,
    indices=None,
    num_samples=3,
    target_layer=None,
    alpha=0.35,
    cmap="jet",
):
    """Compare this notebook's Grad-CAM against pytorch-grad-cam.

    Both methods explain the same predicted class on the same image and layer so
    the visual differences are easier to interpret.
    """
    try:
        from pytorch_grad_cam import GradCAM as TorchGradCAM
        from pytorch_grad_cam.utils.image import show_cam_on_image
        from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    except ImportError as exc:
        raise ImportError(
            "pytorch-grad-cam is not installed. Install it with `%pip install grad-cam` "
            "and rerun this comparison cell."
        ) from exc

    if target_layer is None:
        target_layer = get_default_target_layer(model)

    if indices is None:
        indices = list(range(min(num_samples, len(dataset))))

    cam_helper = TorchGradCAM(model=model, target_layers=[target_layer])
    fig, axes = plt.subplots(len(indices), 5, figsize=(18, 4 * len(indices)))
    if len(indices) == 1:
        axes = np.expand_dims(axes, axis=0)

    for row, index in enumerate(indices):
        image, label = dataset[index]
        custom_cam, pred_idx, pred_prob, image_np = gradcam_one_image(
            model=model,
            image_tensor=image,
            target_layer=target_layer,
            class_names=None,
            device=device,
        )

        input_tensor = image.unsqueeze(0).to(device)
        targets = [ClassifierOutputTarget(pred_idx)]
        torch_cam = cam_helper(input_tensor=input_tensor, targets=targets)[0]

        image_rgb = np.repeat(image_np[..., None], 3, axis=2).astype(np.float32)
        custom_heatmap = cm.get_cmap(cmap)(custom_cam)[..., :3]
        custom_overlay = np.clip((1 - alpha) * image_rgb + alpha * custom_heatmap, 0, 1)
        torch_overlay = show_cam_on_image(image_rgb, torch_cam.astype(np.float32), use_rgb=True)

        label_idx = int(label.item()) if torch.is_tensor(label) else int(label)

        axes[row, 0].imshow(image_np, cmap="gray")
        axes[row, 0].set_title(
            f"Original\nTrue: {class_names[label_idx]}\nPred: {class_names[pred_idx]} ({pred_prob:.1%})"
        )
        axes[row, 0].axis("off")

        axes[row, 1].imshow(custom_cam, cmap=cmap)
        axes[row, 1].set_title("Custom CAM")
        axes[row, 1].axis("off")

        axes[row, 2].imshow(custom_overlay)
        axes[row, 2].set_title("Custom Overlay")
        axes[row, 2].axis("off")

        axes[row, 3].imshow(torch_cam, cmap=cmap)
        axes[row, 3].set_title("pytorch-grad-cam")
        axes[row, 3].axis("off")

        axes[row, 4].imshow(torch_overlay)
        axes[row, 4].set_title("Library Overlay")
        axes[row, 4].axis("off")

    plt.tight_layout()
    return fig
