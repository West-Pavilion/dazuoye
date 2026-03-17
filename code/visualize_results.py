import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
from PIL import Image

CLASS_NAME_TRANSLATIONS = {
    "Bacterial_leaf_blight": "白叶枯病",
    "Bacterial_leaf_streak": "细菌性条斑病",
    "healthy": "健康叶片",
    "Leaf_roller": "稻纵卷叶螟",
}

CHINESE_FONT_CANDIDATES = [
    "Noto Sans CJK SC",
    "Noto Sans CJK JP",
    "Noto Sans SC",
    "Source Han Sans CN",
    "WenQuanYi Zen Hei",
    "WenQuanYi Micro Hei",
    "SimHei",
    "Microsoft YaHei",
    "PingFang SC",
    "Arial Unicode MS",
]


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    default_work_dir = project_root / "outputs" / "resnet34"
    default_eval_dir = default_work_dir / "eval_val"

    parser = argparse.ArgumentParser(description="Generate report-friendly visualization figures.")
    parser.add_argument("--work-dir", type=str, default=str(default_work_dir), help="Training output directory.")
    parser.add_argument("--eval-dir", type=str, default=str(default_eval_dir), help="Evaluation output directory.")
    parser.add_argument("--save-dir", type=str, default="", help="Directory used to save generated figures.")
    parser.add_argument("--max-gallery", type=int, default=8, help="Maximum number of images per gallery.")
    parser.add_argument("--dpi", type=int, default=220, help="Figure DPI.")
    parser.add_argument("--font-path", type=str, default="", help="Optional path to a Chinese font file.")
    return parser.parse_args()


def setup_matplotlib(font_path: str) -> str:
    selected_font = ""

    if font_path:
        custom_font_path = Path(font_path).expanduser().resolve()
        if not custom_font_path.exists():
            raise FileNotFoundError(f"指定的字体文件不存在: {custom_font_path}")
        font_manager.fontManager.addfont(str(custom_font_path))
        selected_font = font_manager.FontProperties(fname=str(custom_font_path)).get_name()
    else:
        available_fonts = {font.name for font in font_manager.fontManager.ttflist}
        for font_name in CHINESE_FONT_CANDIDATES:
            if font_name in available_fonts:
                selected_font = font_name
                break

    if selected_font:
        plt.rcParams["font.sans-serif"] = [selected_font, "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    return selected_font


def localize_class_name(name: str) -> str:
    return CLASS_NAME_TRANSLATIONS.get(name, name.replace("_", " "))


def localize_class_names(names: Sequence[str]) -> List[str]:
    return [localize_class_name(name) for name in names]


def load_history(csv_path: Path) -> List[Dict[str, float]]:
    if not csv_path.exists():
        return []

    rows = []
    with csv_path.open("r", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            rows.append(
                {
                    "epoch": int(row["epoch"]),
                    "train_loss": float(row["train_loss"]),
                    "train_acc": float(row["train_acc"]),
                    "val_loss": float(row["val_loss"]),
                    "val_acc": float(row["val_acc"]),
                    "lr": float(row["lr"]),
                }
            )
    return rows


def load_report(report_path: Path) -> Dict[str, object]:
    if not report_path.exists():
        return {}
    with report_path.open("r", encoding="utf-8") as json_file:
        return json.load(json_file)


def load_predictions(csv_path: Path) -> List[Dict[str, object]]:
    if not csv_path.exists():
        return []

    rows = []
    with csv_path.open("r", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            rows.append(
                {
                    "image_path": row["image_path"],
                    "true_label": row["true_label"],
                    "pred_label": row["pred_label"],
                    "confidence": float(row["confidence"]),
                    "correct": bool(int(row["correct"])),
                }
            )
    return rows


def load_confusion_matrix(csv_path: Path) -> Tuple[List[str], np.ndarray]:
    if not csv_path.exists():
        return [], np.zeros((0, 0), dtype=np.int64)

    with csv_path.open("r", encoding="utf-8") as csv_file:
        reader = list(csv.reader(csv_file))

    class_names = reader[0][1:]
    matrix = np.array([[int(value) for value in row[1:]] for row in reader[1:]], dtype=np.int64)
    return class_names, matrix


def load_split_meta(json_path: Path) -> Dict[str, object]:
    if not json_path.exists():
        return {}
    with json_path.open("r", encoding="utf-8") as json_file:
        return json.load(json_file)


def save_training_curves(history: Sequence[Dict[str, float]], save_path: Path, dpi: int) -> bool:
    if not history:
        return False

    epochs = [item["epoch"] for item in history]
    train_loss = [item["train_loss"] for item in history]
    val_loss = [item["val_loss"] for item in history]
    train_acc = [item["train_acc"] for item in history]
    val_acc = [item["val_acc"] for item in history]
    learning_rate = [item["lr"] for item in history]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].plot(epochs, train_loss, marker="o", linewidth=2.2, color="#d1495b", label="训练损失")
    axes[0].plot(epochs, val_loss, marker="s", linewidth=2.2, color="#2e4057", label="验证损失")
    axes[0].set_title("损失曲线")
    axes[0].set_xlabel("训练轮次")
    axes[0].set_ylabel("损失值")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    axes[1].plot(epochs, train_acc, marker="o", linewidth=2.2, color="#00798c", label="训练准确率")
    axes[1].plot(epochs, val_acc, marker="s", linewidth=2.2, color="#edae49", label="验证准确率")
    axes[1].set_title("准确率曲线")
    axes[1].set_xlabel("训练轮次")
    axes[1].set_ylabel("准确率")
    axes[1].set_ylim(0.0, 1.02)
    axes[1].grid(alpha=0.25)
    axes[1].legend()

    axes[2].plot(epochs, learning_rate, marker="o", linewidth=2.2, color="#4f772d")
    axes[2].set_title("学习率变化")
    axes[2].set_xlabel("训练轮次")
    axes[2].set_ylabel("学习率")
    axes[2].grid(alpha=0.25)

    fig.suptitle("模型训练过程总览", fontsize=16, y=1.02)
    fig.tight_layout()
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return True


def save_confusion_heatmap(class_names: Sequence[str], confusion: np.ndarray, save_path: Path, dpi: int) -> bool:
    if confusion.size == 0:
        return False

    row_sums = confusion.sum(axis=1, keepdims=True)
    normalized = np.divide(confusion, row_sums, out=np.zeros_like(confusion, dtype=float), where=row_sums != 0)

    display_names = localize_class_names(class_names)

    fig, ax = plt.subplots(figsize=(8.5, 7))
    image = ax.imshow(normalized, cmap="YlOrRd", vmin=0.0, vmax=1.0)
    plt.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="归一化比例")

    ax.set_xticks(np.arange(len(display_names)))
    ax.set_yticks(np.arange(len(display_names)))
    ax.set_xticklabels(display_names, rotation=25, ha="right")
    ax.set_yticklabels(display_names)
    ax.set_xlabel("预测类别")
    ax.set_ylabel("真实类别")
    ax.set_title("混淆矩阵")

    threshold = normalized.max() * 0.6 if normalized.size else 0.0
    for row in range(confusion.shape[0]):
        for col in range(confusion.shape[1]):
            text_color = "white" if normalized[row, col] > threshold else "#222222"
            ax.text(
                col,
                row,
                f"{confusion[row, col]}\n{normalized[row, col] * 100:.1f}%",
                ha="center",
                va="center",
                color=text_color,
                fontsize=10,
            )

    fig.tight_layout()
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return True


def save_class_accuracy_chart(report: Dict[str, object], save_path: Path, dpi: int) -> bool:
    summary = report.get("summary", {})
    per_class = summary.get("per_class", []) if isinstance(summary, dict) else []
    if not per_class:
        return False

    class_names = localize_class_names([item["class_name"] for item in per_class])
    accuracies = [float(item["accuracy"]) for item in per_class]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    colors = plt.cm.Spectral(np.linspace(0.12, 0.88, len(class_names)))
    bars = ax.bar(class_names, accuracies, color=colors)
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("准确率")
    ax.set_title("各类别准确率")
    ax.grid(axis="y", alpha=0.25)

    for bar, value in zip(bars, accuracies):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.02,
            f"{value * 100:.2f}%",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    fig.tight_layout()
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return True


def save_split_distribution(split_meta: Dict[str, object], save_path: Path, dpi: int) -> bool:
    classes = split_meta.get("classes", {}) if isinstance(split_meta, dict) else {}
    if not classes:
        return False

    raw_class_names = list(classes.keys())
    class_names = localize_class_names(raw_class_names)
    train_counts = [int(classes[name]["train"]) for name in raw_class_names]
    val_counts = [int(classes[name]["val"]) for name in raw_class_names]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.bar(class_names, train_counts, color="#386641", label="训练集")
    ax.bar(class_names, val_counts, bottom=train_counts, color="#bc4749", label="验证集")
    ax.set_title("数据集划分分布")
    ax.set_ylabel("图像数量")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()

    for index, (train_count, val_count) in enumerate(zip(train_counts, val_counts)):
        total = train_count + val_count
        ax.text(index, total + max(total * 0.01, 5), str(total), ha="center", va="bottom", fontsize=10)

    fig.tight_layout()
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return True


def select_gallery_rows(rows: Sequence[Dict[str, object]], max_gallery: int, errors_only: bool) -> List[Dict[str, object]]:
    filtered = [row for row in rows if row["correct"] == (not errors_only)]
    if not filtered:
        return []

    filtered.sort(key=lambda item: float(item["confidence"]), reverse=True)
    grouped: Dict[str, List[Dict[str, object]]] = {}
    for row in filtered:
        grouped.setdefault(str(row["true_label"]), []).append(row)

    selected: List[Dict[str, object]] = []
    class_names = sorted(grouped.keys())
    round_index = 0
    while len(selected) < max_gallery:
        added = False
        for class_name in class_names:
            samples = grouped[class_name]
            if round_index < len(samples):
                selected.append(samples[round_index])
                added = True
                if len(selected) >= max_gallery:
                    break
        if not added:
            break
        round_index += 1

    return selected


def save_prediction_gallery(
    rows: Sequence[Dict[str, object]],
    save_path: Path,
    dpi: int,
    max_gallery: int,
    errors_only: bool,
) -> bool:
    selected_rows = select_gallery_rows(rows, max_gallery=max_gallery, errors_only=errors_only)
    if not selected_rows:
        return False

    cols = 4 if len(selected_rows) >= 4 else len(selected_rows)
    rows_count = int(math.ceil(len(selected_rows) / cols))
    fig, axes = plt.subplots(rows_count, cols, figsize=(4.2 * cols, 4.3 * rows_count))
    axes_array = np.atleast_1d(axes).reshape(rows_count, cols)

    for axis in axes_array.flat:
        axis.axis("off")

    for axis, item in zip(axes_array.flat, selected_rows):
        image = Image.open(item["image_path"]).convert("RGB")
        axis.imshow(image)
        axis.axis("off")
        title_color = "#1b4332" if item["correct"] else "#9d0208"
        axis.set_title(
            "真实: {}\n预测: {}\n置信度: {:.3f}".format(
                localize_class_name(str(item["true_label"])),
                localize_class_name(str(item["pred_label"])),
                float(item["confidence"]),
            ),
            fontsize=10,
            color=title_color,
        )

    title = "错误预测样例" if errors_only else "正确预测样例"
    fig.suptitle(title, fontsize=16, y=1.02)
    fig.tight_layout()
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return True


def main() -> None:
    args = parse_args()
    work_dir = Path(args.work_dir).expanduser().resolve()
    eval_dir = Path(args.eval_dir).expanduser().resolve()
    save_dir = Path(args.save_dir).expanduser().resolve() if args.save_dir else work_dir / "figures"
    save_dir.mkdir(parents=True, exist_ok=True)
    selected_font = setup_matplotlib(args.font_path)

    history = load_history(work_dir / "history.csv")
    report = load_report(eval_dir / "report.json")
    predictions = load_predictions(eval_dir / "predictions.csv")
    class_names, confusion = load_confusion_matrix(eval_dir / "confusion_matrix.csv")
    split_meta = load_split_meta(work_dir / "prepared_data" / "split_meta.json")

    generated = []
    if save_training_curves(history, save_dir / "training_curves.png", args.dpi):
        generated.append(save_dir / "training_curves.png")
    if save_confusion_heatmap(class_names, confusion, save_dir / "confusion_matrix_heatmap.png", args.dpi):
        generated.append(save_dir / "confusion_matrix_heatmap.png")
    if save_class_accuracy_chart(report, save_dir / "per_class_accuracy.png", args.dpi):
        generated.append(save_dir / "per_class_accuracy.png")
    if save_split_distribution(split_meta, save_dir / "dataset_distribution.png", args.dpi):
        generated.append(save_dir / "dataset_distribution.png")
    if save_prediction_gallery(predictions, save_dir / "prediction_gallery_correct.png", args.dpi, args.max_gallery, errors_only=False):
        generated.append(save_dir / "prediction_gallery_correct.png")
    if save_prediction_gallery(predictions, save_dir / "prediction_gallery_errors.png", args.dpi, args.max_gallery, errors_only=True):
        generated.append(save_dir / "prediction_gallery_errors.png")

    if not generated:
        raise FileNotFoundError(
            "No figures were generated. Make sure history.csv, report.json, confusion_matrix.csv, "
            "and predictions.csv exist in the expected directories."
        )

    if selected_font:
        print(f"已使用中文字体: {selected_font}")
    else:
        print("未检测到可用中文字体，若图中出现方框，请通过 --font-path 指定中文字体文件。")

    print("已生成图片:")
    for path in generated:
        print(f"  {path}")


if __name__ == "__main__":
    main()
