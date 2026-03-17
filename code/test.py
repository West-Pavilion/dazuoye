import argparse
import csv
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets
from tqdm import tqdm

from predict import build_eval_transform, load_class_indices, load_model_weights, resolve_class_indices_path
from train import create_model, has_images, locate_dataset_root, resolve_device


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    default_work_dir = project_root / "outputs" / "resnet34"

    parser = argparse.ArgumentParser(description="Evaluate a trained classification model.")
    parser.add_argument(
        "--data",
        type=str,
        default=str(default_work_dir / "prepared_data"),
        help="Path to dataset root or a specific split directory.",
    )
    parser.add_argument("--split", type=str, default="val", help="Split name when --data points to a dataset root.")
    parser.add_argument(
        "--weights",
        type=str,
        default=str(default_work_dir / "best_model.pth"),
        help="Path to best_model.pth or best_checkpoint.pth.",
    )
    parser.add_argument(
        "--class-indices",
        type=str,
        default="",
        help="Path to class_indices.json. Defaults to <weights_dir>/class_indices.json.",
    )
    parser.add_argument("--model", type=str, default="resnet34", help="Model name used during training.")
    parser.add_argument("--image-size", type=int, default=224, help="Input image size.")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size.")
    parser.add_argument("--num-workers", type=int, default=min(8, os.cpu_count() or 1), help="DataLoader workers.")
    parser.add_argument("--device", type=str, default="auto", help="Inference device.")
    parser.add_argument(
        "--save-dir",
        type=str,
        default="",
        help="Directory to save evaluation artifacts. Defaults to <weights_dir>/eval_<split>.",
    )
    parser.add_argument("--save-preds", action="store_true", help="Save per-image predictions as CSV.")
    return parser.parse_args()


def is_imagefolder_root(path: Path) -> bool:
    if not path.is_dir():
        return False
    subdirs = sorted([child for child in path.iterdir() if child.is_dir() and not child.name.startswith(".")])
    return bool(subdirs) and all(has_images(subdir) for subdir in subdirs)


def resolve_eval_root(data_arg: str, split: str) -> Path:
    data_path = Path(data_arg).expanduser().resolve()
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {data_path}")

    split_candidate = data_path / split
    if split_candidate.is_dir() and is_imagefolder_root(split_candidate):
        return split_candidate

    if is_imagefolder_root(data_path):
        return data_path

    dataset_root, is_split_root = locate_dataset_root(data_path)
    if is_split_root:
        target = dataset_root / split
        if not target.is_dir():
            raise FileNotFoundError(f"Split directory does not exist: {target}")
        return target

    return dataset_root


def build_dataloader(data_root: Path, image_size: int, batch_size: int, num_workers: int) -> Tuple[DataLoader, datasets.ImageFolder]:
    dataset = datasets.ImageFolder(root=str(data_root), transform=build_eval_transform(image_size))
    loader_kwargs = {
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": max(0, num_workers),
        "pin_memory": torch.cuda.is_available(),
    }
    if loader_kwargs["num_workers"] > 0:
        loader_kwargs["persistent_workers"] = True
    dataloader = DataLoader(dataset, **loader_kwargs)
    return dataloader, dataset


def save_confusion_matrix_csv(save_path: Path, class_names: List[str], confusion: torch.Tensor) -> None:
    with save_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["true/pred"] + class_names)
        for class_name, row in zip(class_names, confusion.tolist()):
            writer.writerow([class_name] + row)


def evaluate(
    model: torch.nn.Module,
    dataloader: DataLoader,
    dataset: datasets.ImageFolder,
    class_names: List[str],
    prediction_index_map: torch.Tensor,
    device: torch.device,
    save_predictions: bool,
) -> Tuple[Dict[str, object], List[Dict[str, object]], torch.Tensor]:
    criterion = nn.CrossEntropyLoss()
    confusion = torch.zeros((len(class_names), len(class_names)), dtype=torch.int64)
    prediction_rows: List[Dict[str, object]] = []

    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    sample_offset = 0

    model.eval()
    with torch.no_grad():
        progress = tqdm(dataloader, desc="Test", leave=False)
        for images, labels in progress:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(images)
            loss = criterion(outputs, labels)
            probabilities = torch.softmax(outputs, dim=1)
            confidences, predictions = torch.max(probabilities, dim=1)

            batch_size = labels.size(0)
            total_samples += batch_size
            total_loss += loss.item() * batch_size
            mapped_predictions = prediction_index_map[predictions.cpu()]
            total_correct += (mapped_predictions.to(labels.device) == labels).sum().item()

            labels_cpu = labels.cpu()
            predictions_cpu = mapped_predictions
            confidences_cpu = confidences.cpu()

            for true_label, pred_label in zip(labels_cpu.tolist(), predictions_cpu.tolist()):
                confusion[true_label, pred_label] += 1

            if save_predictions:
                batch_samples = dataset.samples[sample_offset : sample_offset + batch_size]
                for sample, true_label, pred_label, confidence in zip(
                    batch_samples,
                    labels_cpu.tolist(),
                    predictions_cpu.tolist(),
                    confidences_cpu.tolist(),
                ):
                    prediction_rows.append(
                        {
                            "image_path": sample[0],
                            "true_label": class_names[true_label],
                            "pred_label": class_names[pred_label],
                            "confidence": round(float(confidence), 6),
                            "correct": int(true_label == pred_label),
                        }
                    )
                sample_offset += batch_size

            progress.set_postfix(
                loss=f"{total_loss / total_samples:.4f}",
                acc=f"{total_correct / total_samples:.4f}",
            )

    per_class = []
    for class_index, class_name in enumerate(class_names):
        class_total = int(confusion[class_index, :].sum().item())
        class_correct = int(confusion[class_index, class_index].item())
        class_acc = class_correct / class_total if class_total else 0.0
        per_class.append(
            {
                "class_index": class_index,
                "class_name": class_name,
                "total": class_total,
                "correct": class_correct,
                "accuracy": class_acc,
            }
        )

    summary = {
        "samples": total_samples,
        "loss": total_loss / total_samples if total_samples else 0.0,
        "accuracy": total_correct / total_samples if total_samples else 0.0,
        "per_class": per_class,
    }
    return summary, prediction_rows, confusion


def save_prediction_rows(save_path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        return
    with save_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)

    weights_path = Path(args.weights).expanduser().resolve()
    eval_root = resolve_eval_root(args.data, args.split)
    save_dir = Path(args.save_dir).expanduser().resolve() if args.save_dir else weights_path.parent / f"eval_{eval_root.name}"
    save_dir.mkdir(parents=True, exist_ok=True)

    class_indices_path = resolve_class_indices_path(weights_path, args.class_indices)
    class_indices = load_class_indices(class_indices_path)

    dataloader, dataset = build_dataloader(eval_root, args.image_size, args.batch_size, args.num_workers)
    dataset_idx_to_class = {index: class_name for class_name, index in dataset.class_to_idx.items()}
    class_names = [dataset_idx_to_class[index] for index in range(len(dataset_idx_to_class))]
    try:
        prediction_index_map = torch.tensor(
            [dataset.class_to_idx[class_indices[index]] for index in range(len(class_indices))],
            dtype=torch.long,
        )
    except KeyError as exc:
        raise ValueError(f"class_indices.json contains a class not present in the evaluation dataset: {exc}") from exc

    if dataset_idx_to_class != class_indices:
        print("Warning: dataset folder order differs from class_indices.json. Predictions will be remapped automatically.")

    model = create_model(
        model_name=args.model,
        num_classes=len(class_indices),
        pretrained=False,
        freeze_backbone=False,
    )
    model.to(device)
    load_model_weights(model, weights_path, device)

    summary, prediction_rows, confusion = evaluate(
        model=model,
        dataloader=dataloader,
        dataset=dataset,
        class_names=class_names,
        prediction_index_map=prediction_index_map,
        device=device,
        save_predictions=args.save_preds,
    )

    confusion_path = save_dir / "confusion_matrix.csv"
    report_path = save_dir / "report.json"
    save_confusion_matrix_csv(confusion_path, class_names, confusion)

    payload = {
        "data_root": str(eval_root),
        "weights": str(weights_path),
        "device": str(device),
        "summary": summary,
        "confusion_matrix_csv": str(confusion_path),
    }
    with report_path.open("w", encoding="utf-8") as json_file:
        json.dump(payload, json_file, indent=2, ensure_ascii=False)

    if args.save_preds:
        save_prediction_rows(save_dir / "predictions.csv", prediction_rows)

    print(f"data_root: {eval_root}")
    print(f"weights: {weights_path}")
    print(f"device: {device}")
    print(f"samples: {summary['samples']}")
    print(f"loss: {summary['loss']:.4f}")
    print(f"accuracy: {summary['accuracy']:.4f}")
    print("per-class accuracy:")
    for item in summary["per_class"]:
        print(f"  {item['class_name']:<24} acc={item['accuracy']:.4f} ({item['correct']}/{item['total']})")
    print(f"saved report: {report_path}")
    print(f"saved confusion matrix: {confusion_path}")
    if args.save_preds:
        print(f"saved predictions: {save_dir / 'predictions.csv'}")


if __name__ == "__main__":
    main()
