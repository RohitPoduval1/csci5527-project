"""CLI for listing the most and least confident vanilla CNN predictions."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

from torchvision import transforms

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from checkpoint_utils import load_model_checkpoint
from fer_dataset import FERDataset
from vanillaCNN.device_utils import build_runtime_config
from vanillaCNN.eval_utils import bottom_prediction_records, collect_prediction_records, top_prediction_records
from vanillaCNN.model import VanillaCNN


def _version_sort_key(path: Path):
    return int(path.name) if path.name.isdigit() else path.name


def resolve_fer_path(user_path: str | None) -> Path:
    """Resolve the FER2013 root that contains the train/ and test/ folders."""

    candidates: list[Path] = []

    if user_path:
        candidates.append(Path(user_path).expanduser())

    env_path = os.environ.get("FER2013_PATH")
    if env_path:
        candidates.append(Path(env_path).expanduser())

    versions_root = Path.home() / ".cache" / "kagglehub" / "datasets" / "msambare" / "fer2013" / "versions"
    if versions_root.exists():
        version_dirs = sorted((path for path in versions_root.iterdir() if path.is_dir()), key=_version_sort_key, reverse=True)
        candidates.extend(version_dirs)

    for candidate in candidates:
        if (candidate / "train").is_dir() and (candidate / "test").is_dir():
            return candidate

    raise FileNotFoundError(
        "Could not find FER2013 data. Pass --fer-path or set FER2013_PATH to a directory "
        "that contains train/ and test/."
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fer-path", default=None, help="Path to the FER2013 dataset root containing train/ and test/.")
    parser.add_argument(
        "--checkpoint",
        default="vanillaCNN/checkpoints/vanilla_cnn_best.pt",
        help="Checkpoint to load for the vanilla CNN.",
    )
    parser.add_argument("--split", default="test", choices=("train", "test"), help="Dataset split to evaluate.")
    parser.add_argument("--top-k", type=int, default=5, help="How many examples to show per section.")
    parser.add_argument(
        "--sort-by",
        default="predicted_confidence",
        choices=("predicted_confidence", "true_class_confidence", "confidence_margin"),
        help="Score used to rank confidence.",
    )
    parser.add_argument("--batch-size", type=int, default=64, help="Inference batch size.")
    return parser


def _format_record_line(rank: int, record) -> str:
    image_name = Path(record.image_path).name if record.image_path else "<unknown>"
    return (
        f"{rank}. idx={record.dataset_index:<4} true={record.true_name:<8} pred={record.pred_name:<8} "
        f"pred_conf={record.predicted_confidence:>7.2%} true_conf={record.true_class_confidence:>7.2%} "
        f"margin={record.confidence_margin:>7.2%} file={image_name}"
    )


def _print_section(title: str, records) -> None:
    print()
    print(title)
    print("-" * len(title))
    if not records:
        print("No matching records found.")
        return

    for rank, record in enumerate(records, start=1):
        print(_format_record_line(rank, record))


def main() -> None:
    args = build_parser().parse_args()
    runtime = build_runtime_config()
    fer_path = resolve_fer_path(args.fer_path)
    checkpoint_path = Path(args.checkpoint).expanduser()

    dataset = FERDataset(
        fer_path=str(fer_path),
        split=args.split,
        transforms=transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5]),
            ]
        ),
    )

    model = VanillaCNN(num_classes=len(dataset.classes), dropout_rate=0.25, fc_dropout=0.5).to(runtime.device)
    metadata = load_model_checkpoint(model, checkpoint_path, device=runtime.device)

    records = collect_prediction_records(
        model=model,
        dataset=dataset,
        device=runtime.device,
        batch_size=args.batch_size,
        class_names=metadata.get("class_names") or dataset.classes,
        num_workers=runtime.num_workers,
        pin_memory=runtime.pin_memory,
        non_blocking=runtime.non_blocking,
    )

    most_confident_correct = top_prediction_records(records, correct=True, top_k=args.top_k, sort_by=args.sort_by)
    most_confident_incorrect = top_prediction_records(records, correct=False, top_k=args.top_k, sort_by=args.sort_by)
    least_confident_correct = bottom_prediction_records(records, correct=True, top_k=args.top_k, sort_by=args.sort_by)
    least_confident_incorrect = bottom_prediction_records(records, correct=False, top_k=args.top_k, sort_by=args.sort_by)

    print(f"FER path: {fer_path}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Split: {args.split}")
    print(f"Classes: {', '.join(dataset.classes)}")
    print(f"Samples: {len(records)}")
    print(f"Device: {runtime.device}")

    _print_section("Most confident correct predictions", most_confident_correct)
    _print_section("Most confident incorrect predictions", most_confident_incorrect)
    _print_section("Least confident correct predictions", least_confident_correct)
    _print_section("Least confident incorrect predictions", least_confident_incorrect)


if __name__ == "__main__":
    main()
