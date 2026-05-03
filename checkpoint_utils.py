"""Reusable save/load helpers for PyTorch model checkpoints."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import torch


def checkpoint_exists(checkpoint_path: str | Path) -> bool:
    """Return True when a checkpoint file is present on disk."""

    return Path(checkpoint_path).expanduser().exists()


def _resolve_target_module(model, module_attr: str | None):
    if module_attr is None:
        return model

    if not hasattr(model, module_attr):
        raise ValueError(f"Model does not have attribute {module_attr!r}.")
    return getattr(model, module_attr)


def _normalize_checkpoint_path(checkpoint_path: str | Path) -> Path:
    return Path(checkpoint_path).expanduser()


def _normalize_device(device: torch.device | str | None):
    if device is None:
        return None
    return torch.device(device)


def _extract_state_dict_and_metadata(payload: Any) -> tuple[Mapping[str, torch.Tensor], dict[str, Any]]:
    if isinstance(payload, Mapping) and "state_dict" in payload:
        metadata = dict(payload)
        state_dict = metadata.pop("state_dict")
        return state_dict, metadata

    if isinstance(payload, Mapping):
        return payload, {}

    raise TypeError("Checkpoint must be a state_dict mapping or a dict containing 'state_dict'.")


def save_model_checkpoint(
    model,
    checkpoint_path: str | Path,
    *,
    optimizer=None,
    epoch: int | None = None,
    history: dict[str, Any] | None = None,
    model_kwargs: dict[str, Any] | None = None,
    class_names: list[str] | tuple[str, ...] | None = None,
    module_attr: str | None = None,
    extra: dict[str, Any] | None = None,
) -> Path:
    """Save a model checkpoint with optional training metadata.

    Args:
        model: PyTorch model or wrapper object that owns the target module.
        checkpoint_path: Destination ``.pt`` or ``.pth`` path.
        optimizer: Optional optimizer to save for resume-training workflows.
        epoch: Optional epoch number to store.
        history: Optional training history dictionary.
        model_kwargs: Optional constructor kwargs for rebuilding the model later.
        class_names: Optional class name list for inference utilities.
        module_attr: Optional attribute name when the weights live inside a
            nested module, for example ``module_attr="model"`` on a Lightning module.
        extra: Optional free-form metadata dictionary.
    """

    checkpoint_path = _normalize_checkpoint_path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    target_module = _resolve_target_module(model, module_attr)

    checkpoint = {
        "state_dict": target_module.state_dict(),
        "epoch": epoch,
        "history": history,
        "model_kwargs": model_kwargs or {},
        "class_names": list(class_names) if class_names is not None else None,
        "module_attr": module_attr,
        "pytorch_version": str(torch.__version__),
        "extra": extra or {},
    }

    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()

    torch.save(checkpoint, checkpoint_path)
    return checkpoint_path


def load_checkpoint_metadata(
    checkpoint_path: str | Path,
    *,
    device: torch.device | str | None = "cpu",
) -> dict[str, Any]:
    """Load checkpoint contents without applying them to a model."""

    checkpoint_path = _normalize_checkpoint_path(checkpoint_path)
    payload = torch.load(checkpoint_path, map_location=_normalize_device(device))
    _, metadata = _extract_state_dict_and_metadata(payload)
    return metadata


def load_model_checkpoint(
    model,
    checkpoint_path: str | Path,
    *,
    device: torch.device | str | None = "cpu",
    optimizer=None,
    strict: bool = True,
    module_attr: str | None = None,
) -> dict[str, Any]:
    """Load weights from a checkpoint into an existing model instance.

    Returns a metadata dictionary with any saved training information plus
    ``missing_keys`` and ``unexpected_keys`` from ``load_state_dict``.
    """

    checkpoint_path = _normalize_checkpoint_path(checkpoint_path)
    resolved_device = _normalize_device(device)

    payload = torch.load(checkpoint_path, map_location=resolved_device)
    state_dict, metadata = _extract_state_dict_and_metadata(payload)

    resolved_module_attr = module_attr if module_attr is not None else metadata.get("module_attr")
    target_module = _resolve_target_module(model, resolved_module_attr)

    incompatible_keys = target_module.load_state_dict(state_dict, strict=strict)

    if resolved_device is not None:
        model.to(resolved_device)

    optimizer_state = metadata.get("optimizer_state_dict")
    if optimizer is not None and optimizer_state is not None:
        optimizer.load_state_dict(optimizer_state)

    result = dict(metadata)
    result["missing_keys"] = list(incompatible_keys.missing_keys)
    result["unexpected_keys"] = list(incompatible_keys.unexpected_keys)
    result["checkpoint_path"] = str(checkpoint_path)
    result["module_attr"] = resolved_module_attr
    return result


def build_model_from_checkpoint(
    model_factory: Callable[..., Any],
    checkpoint_path: str | Path,
    *,
    device: torch.device | str | None = "cpu",
    strict: bool = True,
    module_attr: str | None = None,
    override_model_kwargs: dict[str, Any] | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Instantiate a model from saved kwargs and load checkpoint weights."""

    metadata = load_checkpoint_metadata(checkpoint_path, device=device)
    model_kwargs = dict(metadata.get("model_kwargs") or {})
    if override_model_kwargs:
        model_kwargs.update(override_model_kwargs)

    model = model_factory(**model_kwargs)
    loaded_metadata = load_model_checkpoint(
        model,
        checkpoint_path,
        device=device,
        strict=strict,
        module_attr=module_attr,
    )
    return model, loaded_metadata


__all__ = [
    "build_model_from_checkpoint",
    "checkpoint_exists",
    "load_checkpoint_metadata",
    "load_model_checkpoint",
    "save_model_checkpoint",
]
