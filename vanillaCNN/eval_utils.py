"""Helpers for ranking confident correct and incorrect FER predictions."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


@dataclass(frozen=True)
class PredictionRecord:
    """Metadata for a single prediction made on a dataset item."""

    dataset_index: int
    image_path: str | None
    true_idx: int
    true_name: str
    pred_idx: int
    pred_name: str
    correct: bool
    predicted_confidence: float
    true_class_confidence: float
    confidence_margin: float
    
import pandas as pd
from tqdm import tqdm

def run_inference_sweep(model, dataloader, device):
    model.eval()
    results = []

    # Pass 1: No gradients needed, just collect probabilities
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(tqdm(dataloader)):
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            probs = F.softmax(outputs, dim=1)
            max_probs, preds = torch.max(probs, dim=1)

            for i in range(len(labels)):
                # If your dataset returns file paths, save those instead of batch indices!
                results.append({
                    'batch_idx': batch_idx,
                    'img_idx_in_batch': i,
                    'true_label': labels[i].item(),
                    'pred_label': preds[i].item(),
                    'prob': max_probs[i].item()
                })

    return pd.DataFrame(results)


def _resolve_class_names(dataset, class_names: list[str] | None) -> list[str]:
    if class_names is not None:
        return list(class_names)

    dataset_classes = getattr(dataset, "classes", None)
    if dataset_classes is None:
        raise ValueError("Pass class_names explicitly when dataset.classes is unavailable.")
    return list(dataset_classes)


def _resolve_image_path(dataset, dataset_index: int) -> str | None:
    images = getattr(dataset, "images", None)
    if images is None or dataset_index >= len(images):
        return None

    image_path = images[dataset_index]
    if isinstance(image_path, Path):
        return str(image_path)
    return str(image_path)


def _sort_key(record: PredictionRecord, sort_by: str) -> float:
    sort_keys = {
        "predicted_confidence": record.predicted_confidence,
        "true_class_confidence": record.true_class_confidence,
        "confidence_margin": record.confidence_margin,
    }
    if sort_by not in sort_keys:
        valid = ", ".join(sort_keys)
        raise ValueError(f"sort_by must be one of: {valid}")
    return sort_keys[sort_by]


def _matches_class(record_value_idx: int, record_value_name: str, emotion: int | str | None) -> bool:
    if emotion is None:
        return True
    if isinstance(emotion, int):
        return record_value_idx == emotion
    return record_value_name == emotion


def rank_prediction_records(
    records: Iterable[PredictionRecord],
    correct: bool | None = None,
    top_k: int = 5,
    sort_by: str = "predicted_confidence",
    true_emotion: int | str | None = None,
    predicted_emotion: int | str | None = None,
    descending: bool = True,
) -> list[PredictionRecord]:
    """Return the highest- or lowest-ranked prediction records overall."""

    filtered_records = [
        record
        for record in records
        if (correct is None or record.correct == correct)
        and _matches_class(record.true_idx, record.true_name, true_emotion)
        and _matches_class(record.pred_idx, record.pred_name, predicted_emotion)
    ]

    return sorted(
        filtered_records,
        key=lambda record: _sort_key(record, sort_by),
        reverse=descending,
    )[:top_k]


def top_prediction_records(
    records: Iterable[PredictionRecord],
    correct: bool | None = None,
    top_k: int = 5,
    sort_by: str = "predicted_confidence",
    true_emotion: int | str | None = None,
    predicted_emotion: int | str | None = None,
) -> list[PredictionRecord]:
    """Return the most confident prediction records after optional filtering."""

    return rank_prediction_records(
        records=records,
        correct=correct,
        top_k=top_k,
        sort_by=sort_by,
        true_emotion=true_emotion,
        predicted_emotion=predicted_emotion,
        descending=True,
    )


def bottom_prediction_records(
    records: Iterable[PredictionRecord],
    correct: bool | None = None,
    top_k: int = 5,
    sort_by: str = "predicted_confidence",
    true_emotion: int | str | None = None,
    predicted_emotion: int | str | None = None,
) -> list[PredictionRecord]:
    """Return the least confident prediction records after optional filtering."""

    return rank_prediction_records(
        records=records,
        correct=correct,
        top_k=top_k,
        sort_by=sort_by,
        true_emotion=true_emotion,
        predicted_emotion=predicted_emotion,
        descending=False,
    )


def collect_prediction_records(
    model,
    dataset,
    device: torch.device | str,
    batch_size: int = 32,
    class_names: list[str] | None = None,
    num_workers: int = 0,
    pin_memory: bool = False,
    non_blocking: bool = False,
) -> list[PredictionRecord]:
    """Run inference on a dataset and keep enough metadata for per-class ranking.

    The helper builds its own DataLoader with ``shuffle=False`` so the
    ``dataset_index`` in each record always maps back to the original dataset.
    """

    resolved_class_names = _resolve_class_names(dataset, class_names)
    device = torch.device(device)

    loader_kwargs = {
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True

    loader = DataLoader(dataset, **loader_kwargs)

    was_training = model.training
    model = model.to(device)
    model.eval()

    records: list[PredictionRecord] = []
    dataset_index = 0

    with torch.inference_mode():
        for images, labels in loader:
            images = images.to(device, non_blocking=non_blocking)
            labels = labels.to(device, non_blocking=non_blocking)

            logits = model(images)
            probabilities = F.softmax(logits, dim=1)

            predicted_confidences, predicted_labels = probabilities.max(dim=1)
            true_class_confidences = probabilities.gather(1, labels.unsqueeze(1)).squeeze(1)
            top_probabilities = probabilities.topk(k=min(2, probabilities.shape[1]), dim=1).values

            if probabilities.shape[1] == 1:
                confidence_margins = top_probabilities[:, 0]
            else:
                confidence_margins = top_probabilities[:, 0] - top_probabilities[:, 1]

            for row in range(labels.shape[0]):
                true_idx = int(labels[row].item())
                pred_idx = int(predicted_labels[row].item())
                records.append(
                    PredictionRecord(
                        dataset_index=dataset_index,
                        image_path=_resolve_image_path(dataset, dataset_index),
                        true_idx=true_idx,
                        true_name=resolved_class_names[true_idx],
                        pred_idx=pred_idx,
                        pred_name=resolved_class_names[pred_idx],
                        correct=true_idx == pred_idx,
                        predicted_confidence=float(predicted_confidences[row].item()),
                        true_class_confidence=float(true_class_confidences[row].item()),
                        confidence_margin=float(confidence_margins[row].item()),
                    )
                )
                dataset_index += 1

    if was_training:
        model.train()

    return records


def top_samples_for_emotion(
    records: Iterable[PredictionRecord],
    emotion: int | str,
    correct: bool,
    top_k: int = 5,
    sort_by: str = "predicted_confidence",
    predicted_emotion: int | str | None = None,
) -> list[PredictionRecord]:
    """Return the most confident correct or incorrect samples for one true class.

    Args:
        records: Output from ``collect_prediction_records``.
        emotion: True class to filter on, either index or class name.
        correct: ``True`` for correct predictions, ``False`` for mistakes.
        top_k: Number of samples to keep.
        sort_by: One of ``predicted_confidence``, ``true_class_confidence``,
            or ``confidence_margin``.
        predicted_emotion: Optional predicted class filter for wrong examples.
    """

    return top_prediction_records(
        records=records,
        correct=correct,
        top_k=top_k,
        sort_by=sort_by,
        true_emotion=emotion,
        predicted_emotion=predicted_emotion,
    )


def bottom_samples_for_emotion(
    records: Iterable[PredictionRecord],
    emotion: int | str,
    correct: bool,
    top_k: int = 5,
    sort_by: str = "predicted_confidence",
    predicted_emotion: int | str | None = None,
) -> list[PredictionRecord]:
    """Return the least confident correct or incorrect samples for one true class."""

    return bottom_prediction_records(
        records=records,
        correct=correct,
        top_k=top_k,
        sort_by=sort_by,
        true_emotion=emotion,
        predicted_emotion=predicted_emotion,
    )


def rank_samples_by_true_class(
    records: Iterable[PredictionRecord],
    correct: bool,
    top_k: int = 5,
    sort_by: str = "predicted_confidence",
    descending: bool = True,
) -> dict[str, list[PredictionRecord]]:
    """Group the highest- or lowest-ranked samples by true emotion."""

    records = list(records)
    classes = sorted({(record.true_idx, record.true_name) for record in records}, key=lambda item: item[0])

    return {
        class_name: rank_prediction_records(
            records=records,
            correct=correct,
            top_k=top_k,
            sort_by=sort_by,
            true_emotion=class_name,
            descending=descending,
        )
        for _, class_name in classes
    }


def top_samples_by_true_class(
    records: Iterable[PredictionRecord],
    correct: bool,
    top_k: int = 5,
    sort_by: str = "predicted_confidence",
) -> dict[str, list[PredictionRecord]]:
    """Group the top confident correct or incorrect samples by true emotion."""

    return rank_samples_by_true_class(
        records=records,
        correct=correct,
        top_k=top_k,
        sort_by=sort_by,
        descending=True,
    )


def bottom_samples_by_true_class(
    records: Iterable[PredictionRecord],
    correct: bool,
    top_k: int = 5,
    sort_by: str = "predicted_confidence",
) -> dict[str, list[PredictionRecord]]:
    """Group the least confident correct or incorrect samples by true emotion."""

    return rank_samples_by_true_class(
        records=records,
        correct=correct,
        top_k=top_k,
        sort_by=sort_by,
        descending=False,
    )


def prediction_records_to_rows(records: Iterable[PredictionRecord]) -> list[dict]:
    """Convert records into plain dictionaries for pandas/DataFrame usage."""

    return [
        {
            "dataset_index": record.dataset_index,
            "image_path": record.image_path,
            "true_idx": record.true_idx,
            "true_name": record.true_name,
            "pred_idx": record.pred_idx,
            "pred_name": record.pred_name,
            "correct": record.correct,
            "predicted_confidence": record.predicted_confidence,
            "true_class_confidence": record.true_class_confidence,
            "confidence_margin": record.confidence_margin,
        }
        for record in records
    ]


def _to_display_image(image_tensor: torch.Tensor, mean, std):
    image_tensor = image_tensor.detach().cpu().float()

    if mean is not None and std is not None:
        mean_tensor = torch.as_tensor(mean, dtype=image_tensor.dtype).view(-1, 1, 1)
        std_tensor = torch.as_tensor(std, dtype=image_tensor.dtype).view(-1, 1, 1)

        if mean_tensor.shape[0] == 1 and image_tensor.shape[0] > 1:
            mean_tensor = mean_tensor.expand(image_tensor.shape[0], 1, 1)
            std_tensor = std_tensor.expand(image_tensor.shape[0], 1, 1)

        image_tensor = image_tensor * std_tensor + mean_tensor

    image_tensor = image_tensor.clamp(0, 1)

    if image_tensor.shape[0] == 1:
        return image_tensor[0].numpy(), "gray"
    return image_tensor.permute(1, 2, 0).numpy(), None


def plot_ranked_samples_by_class(
    dataset,
    grouped_records: dict[str, list[PredictionRecord]],
    title: str | None = None,
    denorm_mean=0.5,
    denorm_std=0.5,
):
    """Visualize ranked records, one row per true emotion class."""

    import matplotlib.pyplot as plt
    import numpy as np

    if not grouped_records:
        raise ValueError("grouped_records is empty.")

    class_names = list(grouped_records.keys())
    max_columns = max((len(records) for records in grouped_records.values()), default=0)
    if max_columns == 0:
        raise ValueError("grouped_records does not contain any samples to plot.")

    fig, axes = plt.subplots(
        len(class_names),
        max_columns,
        figsize=(3.2 * max_columns, 3.2 * len(class_names)),
        squeeze=False,
    )

    for row, class_name in enumerate(class_names):
        class_records = grouped_records[class_name]

        for column in range(max_columns):
            axis = axes[row, column]

            if column >= len(class_records):
                axis.axis("off")
                continue

            record = class_records[column]
            image_tensor, _ = dataset[record.dataset_index]
            image_np, cmap = _to_display_image(image_tensor, denorm_mean, denorm_std)

            axis.imshow(image_np, cmap=cmap)
            axis.set_title(
                "\n".join(
                    [
                        f"True: {record.true_name}",
                        f"Pred: {record.pred_name}",
                        f"Conf: {record.predicted_confidence:.1%}",
                    ]
                ),
                fontsize=10,
            )
            axis.set_xticks([])
            axis.set_yticks([])

        axes[row, 0].set_ylabel(class_name, rotation=90, fontsize=11)

    if title is not None:
        fig.suptitle(title, fontsize=14)
        fig.tight_layout(rect=(0, 0, 1, 0.98))
    else:
        fig.tight_layout()

    return fig, np.asarray(axes)


__all__ = [
    "bottom_prediction_records",
    "bottom_samples_by_true_class",
    "bottom_samples_for_emotion",
    "PredictionRecord",
    "collect_prediction_records",
    "plot_ranked_samples_by_class",
    "prediction_records_to_rows",
    "rank_prediction_records",
    "rank_samples_by_true_class",
    "top_prediction_records",
    "top_samples_by_true_class",
    "top_samples_for_emotion",
]
