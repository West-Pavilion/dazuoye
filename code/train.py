import argparse
import csv
import json
import os
import random
import shutil
import zipfile
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from tqdm import tqdm

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:  # pragma: no cover
    SummaryWriter = None


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
SUPPORTED_MODELS = {
    "resnet18": models.resnet18,
    "resnet34": models.resnet34,
    "resnet50": models.resnet50,
    "mobilenet_v3_small": models.mobilenet_v3_small,
    "mobilenet_v3_large": models.mobilenet_v3_large,
    "efficientnet_b0": models.efficientnet_b0,
}
WEIGHT_ENUMS = {
    "resnet18": "ResNet18_Weights",
    "resnet34": "ResNet34_Weights",
    "resnet50": "ResNet50_Weights",
    "mobilenet_v3_small": "MobileNet_V3_Small_Weights",
    "mobilenet_v3_large": "MobileNet_V3_Large_Weights",
    "efficientnet_b0": "EfficientNet_B0_Weights",
}


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    default_data = project_root / "dataset" / "dataset_classification.zip"
    default_work_dir = project_root / "outputs" / "resnet34"

    parser = argparse.ArgumentParser(description="Train an image classification model for AutoDL.")
    parser.add_argument(
        "--data",
        type=str,
        default=str(default_data),
        help="Path to dataset zip, dataset root, or an existing train/val directory.",
    )
    parser.add_argument(
        "--work-dir",
        type=str,
        default=str(default_work_dir),
        help="Directory used for logs, checkpoints, extracted files, and split data.",
    )
    parser.add_argument(
        "--split-dir",
        type=str,
        default="",
        help="Directory for generated train/val split. Defaults to <work-dir>/prepared_data.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="resnet34",
        choices=sorted(SUPPORTED_MODELS.keys()),
        help="Torchvision model name.",
    )
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size.")
    parser.add_argument("--lr", type=float, default=3e-4, help="Initial learning rate.")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay.")
    parser.add_argument("--image-size", type=int, default=224, help="Input image size.")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Validation ratio for auto split.")
    parser.add_argument("--num-workers", type=int, default=min(8, os.cpu_count() or 1), help="DataLoader workers.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--device", type=str, default="auto", help="Device, for example auto, cuda, cuda:0, cpu.")
    parser.add_argument(
        "--resume",
        type=str,
        default="",
        help="Resume training from a checkpoint saved by this script.",
    )
    parser.add_argument(
        "--pretrained",
        dest="pretrained",
        action="store_true",
        help="Use ImageNet pretrained weights.",
    )
    parser.add_argument(
        "--no-pretrained",
        dest="pretrained",
        action="store_false",
        help="Train from scratch.",
    )
    parser.set_defaults(pretrained=True)
    parser.add_argument(
        "--freeze-backbone",
        action="store_true",
        help="Freeze the backbone and train only the classifier head.",
    )
    parser.add_argument(
        "--amp",
        dest="amp",
        action="store_true",
        help="Enable mixed precision when CUDA is available.",
    )
    parser.add_argument(
        "--no-amp",
        dest="amp",
        action="store_false",
        help="Disable mixed precision.",
    )
    parser.set_defaults(amp=True)
    parser.add_argument(
        "--overwrite-split",
        action="store_true",
        help="Rebuild generated train/val split even if it already exists.",
    )
    parser.add_argument(
        "--overwrite-extract",
        action="store_true",
        help="Re-extract the zip dataset even if the extracted directory already exists.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    requested = torch.device(device_name)
    if requested.type == "cuda" and not torch.cuda.is_available():
        print("CUDA is not available, falling back to CPU.")
        return torch.device("cpu")
    return requested


def is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def has_images(path: Path) -> bool:
    return any(is_image_file(child) for child in path.iterdir())


def locate_dataset_root(path: Path) -> Tuple[Path, bool]:
    current = path
    for _ in range(4):
        train_dir = current / "train"
        val_dir = current / "val"
        if train_dir.is_dir() and val_dir.is_dir():
            return current, True

        subdirs = sorted([child for child in current.iterdir() if child.is_dir() and not child.name.startswith(".")])
        if subdirs and all(has_images(subdir) for subdir in subdirs):
            return current, False
        if len(subdirs) == 1:
            current = subdirs[0]
            continue
        break

    raise FileNotFoundError(
        "Unable to locate a valid dataset root. Expected a zip, a folder with class subdirectories, "
        "or a folder containing train/val."
    )


def extract_zip_if_needed(zip_path: Path, extract_root: Path, overwrite: bool) -> Path:
    target_dir = extract_root / zip_path.stem
    marker = target_dir / ".extract_done"
    if overwrite and target_dir.exists():
        shutil.rmtree(target_dir)
    if not marker.exists():
        target_dir.mkdir(parents=True, exist_ok=True)
        print("Extracting dataset zip...")
        with zipfile.ZipFile(zip_path, "r") as zip_file:
            zip_file.extractall(target_dir)
        marker.write_text("ok", encoding="utf-8")
    return target_dir


def link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def prepare_split_dataset(raw_root: Path, split_root: Path, val_ratio: float, seed: int, overwrite: bool) -> Path:
    train_root = split_root / "train"
    val_root = split_root / "val"
    meta_path = split_root / "split_meta.json"

    if overwrite and split_root.exists():
        shutil.rmtree(split_root)

    if train_root.is_dir() and val_root.is_dir() and meta_path.is_file() and not overwrite:
        print(f"Using existing split dataset: {split_root}")
        return split_root

    if split_root.exists():
        shutil.rmtree(split_root)
    split_root.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    classes = sorted([path for path in raw_root.iterdir() if path.is_dir() and not path.name.startswith(".")])
    if not classes:
        raise RuntimeError(f"No class folders found in {raw_root}")

    split_meta = {
        "source": str(raw_root),
        "val_ratio": val_ratio,
        "seed": seed,
        "classes": {},
    }

    print(f"Building train/val split under: {split_root}")
    for class_dir in classes:
        images = sorted([path for path in class_dir.iterdir() if is_image_file(path)])
        if not images:
            continue

        rng.shuffle(images)
        val_count = int(len(images) * val_ratio)
        if len(images) > 1:
            val_count = max(1, min(val_count, len(images) - 1))
        else:
            val_count = 0

        val_images = images[:val_count]
        train_images = images[val_count:]

        for image_path in train_images:
            link_or_copy(image_path, train_root / class_dir.name / image_path.name)
        for image_path in val_images:
            link_or_copy(image_path, val_root / class_dir.name / image_path.name)

        split_meta["classes"][class_dir.name] = {
            "train": len(train_images),
            "val": len(val_images),
            "total": len(images),
        }

    with meta_path.open("w", encoding="utf-8") as meta_file:
        json.dump(split_meta, meta_file, indent=2, ensure_ascii=False)

    return split_root


def resolve_data_root(args: argparse.Namespace, work_dir: Path) -> Path:
    data_path = Path(args.data).expanduser().resolve()
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {data_path}")

    if data_path.is_file():
        if data_path.suffix.lower() != ".zip":
            raise ValueError(f"Unsupported dataset file: {data_path}")
        extracted_root = extract_zip_if_needed(data_path, work_dir / "cache", args.overwrite_extract)
        dataset_root, is_split = locate_dataset_root(extracted_root)
    else:
        dataset_root, is_split = locate_dataset_root(data_path)

    if is_split:
        return dataset_root

    split_dir = Path(args.split_dir).expanduser().resolve() if args.split_dir else work_dir / "prepared_data"
    return prepare_split_dataset(dataset_root, split_dir, args.val_ratio, args.seed, args.overwrite_split)


def build_transforms(image_size: int) -> Dict[str, transforms.Compose]:
    resize_size = int(image_size * 256 / 224)
    return {
        "train": transforms.Compose(
            [
                transforms.RandomResizedCrop(image_size, scale=(0.75, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        ),
        "val": transforms.Compose(
            [
                transforms.Resize(resize_size),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        ),
    }


def build_dataloaders(data_root: Path, image_size: int, batch_size: int, num_workers: int) -> Tuple[DataLoader, DataLoader, int, Dict[str, int]]:
    data_transforms = build_transforms(image_size)
    train_dataset = datasets.ImageFolder(root=str(data_root / "train"), transform=data_transforms["train"])
    val_dataset = datasets.ImageFolder(root=str(data_root / "val"), transform=data_transforms["val"])

    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": max(0, num_workers),
        "pin_memory": torch.cuda.is_available(),
    }
    if loader_kwargs["num_workers"] > 0:
        loader_kwargs["persistent_workers"] = True

    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)
    return train_loader, val_loader, len(train_dataset.classes), train_dataset.class_to_idx


def create_model(model_name: str, num_classes: int, pretrained: bool, freeze_backbone: bool) -> nn.Module:
    builder = SUPPORTED_MODELS[model_name]
    model = None

    if pretrained:
        try:
            weights_name = WEIGHT_ENUMS.get(model_name)
            if weights_name and hasattr(models, weights_name):
                weights = getattr(models, weights_name).DEFAULT
                model = builder(weights=weights)
            else:
                model = builder(pretrained=True)
        except Exception as exc:
            print(f"Failed to load pretrained weights: {exc}")
            print("Falling back to randomly initialized weights.")

    if model is None:
        try:
            model = builder(weights=None)
        except TypeError:
            model = builder(pretrained=False)

    if model_name.startswith("resnet"):
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        classifier = model.fc
    else:
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
        classifier = model.classifier

    if freeze_backbone:
        for parameter in model.parameters():
            parameter.requires_grad = False
        for parameter in classifier.parameters():
            parameter.requires_grad = True

    return model


def save_class_indices(class_to_idx: Dict[str, int], save_path: Path) -> None:
    idx_to_class = {str(index): class_name for class_name, index in class_to_idx.items()}
    with save_path.open("w", encoding="utf-8") as json_file:
        json.dump(idx_to_class, json_file, indent=2, ensure_ascii=False)


def save_checkpoint(
    checkpoint_path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: CosineAnnealingLR,
    epoch: int,
    best_acc: float,
    args: argparse.Namespace,
    class_to_idx: Dict[str, int],
) -> None:
    checkpoint = {
        "epoch": epoch,
        "best_acc": best_acc,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "class_to_idx": class_to_idx,
        "args": vars(args),
    }
    torch.save(checkpoint, checkpoint_path)


def load_checkpoint(
    checkpoint_path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: CosineAnnealingLR,
    device: torch.device,
) -> Tuple[int, float]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if "model_state_dict" not in checkpoint:
        raise ValueError(f"Checkpoint does not contain optimizer/scheduler state: {checkpoint_path}")

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    start_epoch = int(checkpoint.get("epoch", -1)) + 1
    best_acc = float(checkpoint.get("best_acc", 0.0))
    return start_epoch, best_acc


def run_train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: GradScaler,
    amp_enabled: bool,
) -> Tuple[float, float]:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    progress = tqdm(dataloader, desc="Train", leave=False)
    for images, labels in progress:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=amp_enabled):
            outputs = model(images)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        batch_size = labels.size(0)
        total_samples += batch_size
        total_loss += loss.item() * batch_size
        total_correct += (outputs.argmax(dim=1) == labels).sum().item()

        progress.set_postfix(
            loss=f"{total_loss / total_samples:.4f}",
            acc=f"{total_correct / total_samples:.4f}",
        )

    return total_loss / total_samples, total_correct / total_samples


def run_val_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    progress = tqdm(dataloader, desc="Val", leave=False)
    with torch.no_grad():
        for images, labels in progress:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(images)
            loss = criterion(outputs, labels)

            batch_size = labels.size(0)
            total_samples += batch_size
            total_loss += loss.item() * batch_size
            total_correct += (outputs.argmax(dim=1) == labels).sum().item()

            progress.set_postfix(
                loss=f"{total_loss / total_samples:.4f}",
                acc=f"{total_correct / total_samples:.4f}",
            )

    return total_loss / total_samples, total_correct / total_samples


def append_history(csv_path: Path, row: Dict[str, object]) -> None:
    file_exists = csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    work_dir = Path(args.work_dir).expanduser().resolve()
    work_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)
    amp_enabled = bool(args.amp and device.type == "cuda")
    print(f"Using device: {device}")
    print(f"AMP enabled: {amp_enabled}")

    data_root = resolve_data_root(args, work_dir)
    print(f"Training data root: {data_root}")

    train_loader, val_loader, num_classes, class_to_idx = build_dataloaders(
        data_root=data_root,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    print(f"Classes: {num_classes}")
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")

    save_class_indices(class_to_idx, work_dir / "class_indices.json")

    model = create_model(
        model_name=args.model,
        num_classes=num_classes,
        pretrained=args.pretrained,
        freeze_backbone=args.freeze_backbone,
    )
    model.to(device)

    trainable_params = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))
    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler(enabled=amp_enabled)

    start_epoch = 0
    best_acc = 0.0
    if args.resume:
        resume_path = Path(args.resume).expanduser().resolve()
        if not resume_path.exists():
            raise FileNotFoundError(f"Resume checkpoint does not exist: {resume_path}")
        start_epoch, best_acc = load_checkpoint(resume_path, model, optimizer, scheduler, device)
        print(f"Resumed from {resume_path} at epoch {start_epoch}, best_acc={best_acc:.4f}")

    writer = SummaryWriter(str(work_dir / "tensorboard")) if SummaryWriter is not None else None
    if writer is None:
        print("TensorBoard is unavailable because torch.utils.tensorboard is not installed.")

    last_checkpoint = work_dir / "last_checkpoint.pth"
    best_checkpoint = work_dir / "best_checkpoint.pth"
    best_weights = work_dir / "best_model.pth"
    history_csv = work_dir / "history.csv"

    for epoch in range(start_epoch, args.epochs):
        print(f"\nEpoch [{epoch + 1}/{args.epochs}]")
        train_loss, train_acc = run_train_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            scaler=scaler,
            amp_enabled=amp_enabled,
        )
        val_loss, val_acc = run_val_epoch(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
        )
        scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]
        print(
            "train_loss={:.4f} train_acc={:.4f} val_loss={:.4f} val_acc={:.4f} lr={:.6f}".format(
                train_loss, train_acc, val_loss, val_acc, current_lr
            )
        )

        row = {
            "epoch": epoch + 1,
            "train_loss": f"{train_loss:.6f}",
            "train_acc": f"{train_acc:.6f}",
            "val_loss": f"{val_loss:.6f}",
            "val_acc": f"{val_acc:.6f}",
            "lr": f"{current_lr:.8f}",
        }
        append_history(history_csv, row)

        if writer is not None:
            writer.add_scalar("loss/train", train_loss, epoch + 1)
            writer.add_scalar("loss/val", val_loss, epoch + 1)
            writer.add_scalar("acc/train", train_acc, epoch + 1)
            writer.add_scalar("acc/val", val_acc, epoch + 1)
            writer.add_scalar("lr", current_lr, epoch + 1)

        if val_acc >= best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), best_weights)
            save_checkpoint(best_checkpoint, model, optimizer, scheduler, epoch, best_acc, args, class_to_idx)
            print(f"Saved new best model to: {best_weights}")

        save_checkpoint(last_checkpoint, model, optimizer, scheduler, epoch, best_acc, args, class_to_idx)

    summary = {
        "best_val_acc": best_acc,
        "best_model": str(best_weights),
        "last_checkpoint": str(last_checkpoint),
        "best_checkpoint": str(best_checkpoint),
        "class_indices": str(work_dir / "class_indices.json"),
        "data_root": str(data_root),
    }
    with (work_dir / "summary.json").open("w", encoding="utf-8") as summary_file:
        json.dump(summary, summary_file, indent=2, ensure_ascii=False)

    if writer is not None:
        writer.close()

    print("\nTraining finished.")
    print(f"Best val acc: {best_acc:.4f}")
    print(f"Outputs saved to: {work_dir}")


if __name__ == "__main__":
    main()
