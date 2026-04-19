"""Cross-platform runtime helpers for PyTorch notebooks and scripts.

The goal is to make a single notebook work cleanly across:
- Windows and Linux with CUDA
- macOS with Apple Silicon / MPS
- CPU-only environments such as lightweight local runs
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
import os

import torch
from torch.utils.data import DataLoader


@dataclass(frozen=True)
class RuntimeConfig:
    """Bundle common runtime decisions in one place."""

    device: torch.device
    device_type: str
    device_name: str
    amp_enabled: bool
    non_blocking: bool
    num_workers: int
    pin_memory: bool
    persistent_workers: bool


def select_device() -> tuple[torch.device, str, str]:
    """Pick the best available device for the current machine."""
    if torch.cuda.is_available():
        return torch.device("cuda"), "cuda", torch.cuda.get_device_name(0)

    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend is not None and mps_backend.is_available():
        return torch.device("mps"), "mps", "Apple Silicon GPU (MPS)"

    return torch.device("cpu"), "cpu", "CPU"


def build_runtime_config(max_workers: int = 4) -> RuntimeConfig:
    """Create a runtime config tuned for the detected device."""
    device, device_type, device_name = select_device()
    amp_enabled = device_type == "cuda"
    non_blocking = device_type == "cuda"
    available_workers = os.cpu_count() or 1

    if device_type == "cuda":
        pin_memory = True
        if os.name == "nt":
            # Windows notebooks are often more stable with a single-process loader.
            num_workers = 0
            persistent_workers = False
        else:
            num_workers = max(1, min(max_workers, available_workers))
            persistent_workers = num_workers > 0
    else:
        # num_workers=0 is the safest default across notebooks, macOS, and CPU-only runs.
        num_workers = 0
        pin_memory = False
        persistent_workers = False

    return RuntimeConfig(
        device=device,
        device_type=device_type,
        device_name=device_name,
        amp_enabled=amp_enabled,
        non_blocking=non_blocking,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )


def build_dataloader_kwargs(
    runtime: RuntimeConfig,
    batch_size: int,
    shuffle: bool,
) -> dict:
    """Return DataLoader kwargs suitable for the detected runtime."""
    kwargs = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": runtime.num_workers,
        "pin_memory": runtime.pin_memory,
    }
    if runtime.num_workers > 0:
        kwargs["persistent_workers"] = runtime.persistent_workers
    return kwargs


def make_dataloader(dataset, batch_size: int, shuffle: bool, runtime: RuntimeConfig) -> DataLoader:
    """Build a DataLoader using runtime-aware defaults."""
    return DataLoader(dataset, **build_dataloader_kwargs(runtime, batch_size, shuffle))


def autocast_context(device_type: str, enabled: bool):
    """Return the correct autocast context manager for the current runtime."""
    if enabled and device_type == "cuda":
        return torch.amp.autocast(device_type="cuda")
    return nullcontext()


def make_grad_scaler(device_type: str, enabled: bool):
    """Return a GradScaler when it is supported and useful."""
    if enabled and device_type == "cuda":
        return torch.amp.GradScaler("cuda")
    return None


def clear_runtime_cache(device_type: str) -> None:
    """Release cached memory when the backend exposes a cache API."""
    if device_type == "cuda":
        torch.cuda.empty_cache()
        return

    if device_type == "mps" and hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
        torch.mps.empty_cache()


__all__ = [
    "RuntimeConfig",
    "autocast_context",
    "build_dataloader_kwargs",
    "build_runtime_config",
    "clear_runtime_cache",
    "make_dataloader",
    "make_grad_scaler",
    "select_device",
]
