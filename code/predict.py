import argparse
import json
from pathlib import Path
from typing import Dict, List

import torch
from PIL import Image
from torchvision import transforms

from train import IMAGENET_MEAN, IMAGENET_STD, create_model, resolve_device


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    default_work_dir = project_root / "outputs" / "resnet34"

    parser = argparse.ArgumentParser(description="Predict the class of a single image.")
    parser.add_argument("--image", type=str, required=True, help="Path to the image.")
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
    parser.add_argument("--device", type=str, default="auto", help="Inference device.")
    parser.add_argument("--topk", type=int, default=4, help="Number of classes to print.")
    parser.add_argument("--save-json", type=str, default="", help="Optional path to save prediction results as JSON.")
    return parser.parse_args()


def build_eval_transform(image_size: int) -> transforms.Compose:
    resize_size = int(image_size * 256 / 224)
    return transforms.Compose(
        [
            transforms.Resize(resize_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def resolve_class_indices_path(weights_path: Path, class_indices_arg: str) -> Path:
    if class_indices_arg:
        return Path(class_indices_arg).expanduser().resolve()
    return weights_path.resolve().parent / "class_indices.json"


def load_class_indices(json_path: Path) -> Dict[int, str]:
    if not json_path.exists():
        raise FileNotFoundError(f"class_indices.json does not exist: {json_path}")
    with json_path.open("r", encoding="utf-8") as json_file:
        class_indices = json.load(json_file)
    return {int(index): class_name for index, class_name in class_indices.items()}


def load_model_weights(model: torch.nn.Module, weights_path: Path, device: torch.device) -> None:
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights file does not exist: {weights_path}")

    checkpoint = torch.load(weights_path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)


def predict(image_path: Path, model: torch.nn.Module, device: torch.device, image_size: int) -> torch.Tensor:
    if not image_path.exists():
        raise FileNotFoundError(f"Image does not exist: {image_path}")

    transform = build_eval_transform(image_size)
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        logits = model(image_tensor)
        probabilities = torch.softmax(logits, dim=1).squeeze(0).cpu()
    return probabilities


def build_results(probabilities: torch.Tensor, class_indices: Dict[int, str], topk: int) -> List[Dict[str, float]]:
    topk = max(1, min(topk, len(class_indices)))
    values, indices = torch.topk(probabilities, k=topk)

    results = []
    for score, index in zip(values.tolist(), indices.tolist()):
        results.append(
            {
                "class_index": index,
                "class_name": class_indices[index],
                "probability": score,
            }
        )
    return results


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)

    image_path = Path(args.image).expanduser().resolve()
    weights_path = Path(args.weights).expanduser().resolve()
    class_indices_path = resolve_class_indices_path(weights_path, args.class_indices)
    class_indices = load_class_indices(class_indices_path)

    model = create_model(
        model_name=args.model,
        num_classes=len(class_indices),
        pretrained=False,
        freeze_backbone=False,
    )
    model.to(device)
    load_model_weights(model, weights_path, device)

    probabilities = predict(image_path, model, device, args.image_size)
    results = build_results(probabilities, class_indices, args.topk)

    print(f"image: {image_path}")
    print(f"weights: {weights_path}")
    print(f"device: {device}")
    print(f"prediction: {results[0]['class_name']} ({results[0]['probability']:.4f})")
    print("top-k results:")
    for item in results:
        print(f"  {item['class_name']:<24} prob={item['probability']:.4f} index={item['class_index']}")

    if args.save_json:
        save_path = Path(args.save_json).expanduser().resolve()
        save_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "image": str(image_path),
            "weights": str(weights_path),
            "device": str(device),
            "prediction": results[0],
            "topk": results,
        }
        with save_path.open("w", encoding="utf-8") as json_file:
            json.dump(payload, json_file, indent=2, ensure_ascii=False)
        print(f"saved json: {save_path}")


if __name__ == "__main__":
    main()
