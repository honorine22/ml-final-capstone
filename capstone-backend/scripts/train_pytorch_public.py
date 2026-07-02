import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageFile
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms
import timm


ImageFile.LOAD_TRUNCATED_IMAGES = True
ROOT = Path(__file__).resolve().parents[1]


def choose_device(force_cpu: bool = False):
    if force_cpu or not torch.cuda.is_available():
        return torch.device("cpu"), "cpu"

    try:
        device = torch.device("cuda")
        smoke = nn.Sequential(nn.Conv2d(3, 4, 3, padding=1), nn.BatchNorm2d(4)).to(device)
        sample = torch.randn(2, 3, 32, 32, device=device)
        _ = smoke(sample)
        torch.cuda.synchronize()
        return device, torch.cuda.get_device_name(0)
    except Exception as exc:
        print("CUDA is visible but unusable in this runtime; falling back to CPU.")
        print(f"CUDA check error: {exc}")
        return torch.device("cpu"), "cpu_fallback"


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 1.5, weight=None, label_smoothing: float = 0.03):
        super().__init__()
        self.gamma = gamma
        self.ce = nn.CrossEntropyLoss(weight=weight, label_smoothing=label_smoothing, reduction="none")

    def forward(self, logits, targets):
        ce_loss = self.ce(logits, targets)
        pt = torch.exp(-ce_loss)
        return ((1 - pt) ** self.gamma * ce_loss).mean()


class MaizeImageDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, class_to_idx: dict[str, int], transform=None):
        self.frame = frame.reset_index(drop=True)
        self.class_to_idx = class_to_idx
        self.transform = transform

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        path = Path(row.get("prepared_path", row["path"]))
        label = self.class_to_idx[row["label"]]
        try:
            image = Image.open(path).convert("RGB")
        except Exception:
            image = Image.new("RGB", (224, 224), color=(230, 210, 170))
        if self.transform:
            image = self.transform(image)
        return image, label


def parse_args():
    parser = argparse.ArgumentParser(description="Train and compare PyTorch public-dataset maize models.")
    parser.add_argument("--manifest-dir", default="reports/public_training")
    parser.add_argument("--out-dir", default="reports/public_training")
    parser.add_argument("--models", nargs="+", default=["mobilenetv3_large_100", "tf_efficientnetv2_b0", "convnext_tiny"])
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--head-epochs", type=int, default=3)
    parser.add_argument("--finetune-epochs", type=int, default=8)
    parser.add_argument("--lr-head", type=float, default=1e-3)
    parser.add_argument("--lr-finetune", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.03)
    parser.add_argument("--focal-gamma", type=float, default=1.5)
    parser.add_argument("--class-weight-cap", type=float, default=2.0)
    parser.add_argument("--balanced-sampler", action="store_true", help="Use only when class imbalance is severe.")
    parser.add_argument("--early-stop-patience", type=int, default=3)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--force-cpu", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-pretrained", action="store_true")
    return parser.parse_args()


def build_transforms(image_size: int):
    train_tfms = transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size, scale=(0.65, 1.0), ratio=(0.85, 1.15)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.2),
            transforms.RandomRotation(degrees=25),
            transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.2, hue=0.03),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.15),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )
    valid_tfms = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )
    return train_tfms, valid_tfms


def create_model(model_name: str, num_classes: int, pretrained: bool):
    return timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)


def set_backbone_trainable(model, trainable: bool):
    for _, parameter in model.named_parameters():
        parameter.requires_grad = trainable
    for name, parameter in model.named_parameters():
        if any(key in name.lower() for key in ["classifier", "head", "fc"]):
            parameter.requires_grad = True


def evaluate(model, loader, criterion, device, class_names, save_rows: bool = False):
    model.eval()
    losses, y_true, y_pred, y_prob = [], [], [], []
    rows = []
    with torch.no_grad():
        for batch_index, (images, labels) in enumerate(loader):
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            probs = torch.softmax(logits, dim=1)
            preds = probs.argmax(dim=1)
            losses.append(criterion(logits, labels).item())
            y_true.extend(labels.cpu().numpy().tolist())
            y_pred.extend(preds.cpu().numpy().tolist())
            y_prob.extend(probs.cpu().numpy().tolist())
            if save_rows:
                for item_index, (true_idx, pred_idx, prob_values) in enumerate(
                    zip(labels.cpu().numpy().tolist(), preds.cpu().numpy().tolist(), probs.cpu().numpy().tolist())
                ):
                    rows.append(
                        {
                            "batch": batch_index,
                            "item": item_index,
                            "true": class_names[true_idx],
                            "pred": class_names[pred_idx],
                            "confidence": float(max(prob_values)),
                            "is_error": true_idx != pred_idx,
                            **{f"prob_{class_names[i]}": float(prob_values[i]) for i in range(len(class_names))},
                        }
                    )

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )
    return {
        "loss": float(np.mean(losses)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision),
        "recall_macro": float(recall),
        "f1_macro": float(f1),
        "classification_report": classification_report(
            y_true,
            y_pred,
            target_names=class_names,
            zero_division=0,
            output_dict=True,
        ),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "prediction_rows": rows,
    }


def train_phase(model, train_loader, val_loader, criterion, optimizer, epochs, phase, model_name, out_dir, device, class_names, patience, grad_clip):
    best_f1 = -1.0
    stale_epochs = 0
    best_path = out_dir / f"{model_name}_{phase}_best.pt"
    rows = []

    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        start = time.perf_counter()

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            losses.append(loss.item())

        val_metrics = evaluate(model, val_loader, criterion, device, class_names)
        row = {
            "model": model_name,
            "phase": phase,
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_f1_macro": val_metrics["f1_macro"],
            "seconds": round(time.perf_counter() - start, 2),
        }
        rows.append(row)
        print(row)

        if val_metrics["f1_macro"] > best_f1:
            best_f1 = val_metrics["f1_macro"]
            stale_epochs = 0
            torch.save(model.state_dict(), best_path)
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                print(f"Early stopping {model_name}/{phase} after {epoch} epochs.")
                break

    return pd.DataFrame(rows), best_path, best_f1


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    manifest_dir = ROOT / args.manifest_dir
    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(manifest_dir / "train_manifest.csv")
    val_df = pd.read_csv(manifest_dir / "val_manifest.csv")
    test_df = pd.read_csv(manifest_dir / "test_manifest.csv")
    class_names = json.loads((manifest_dir / "class_names.json").read_text(encoding="utf-8"))
    class_to_idx = {name: index for index, name in enumerate(class_names)}

    device, device_name = choose_device(args.force_cpu)
    print("Device:", device)
    print("Device name:", device_name)
    print("Classes:", class_names)

    train_tfms, valid_tfms = build_transforms(args.image_size)
    train_ds = MaizeImageDataset(train_df, class_to_idx, transform=train_tfms)
    val_ds = MaizeImageDataset(val_df, class_to_idx, transform=valid_tfms)
    test_ds = MaizeImageDataset(test_df, class_to_idx, transform=valid_tfms)

    class_counts = train_df["label"].value_counts().to_dict()
    sampler = None
    shuffle_train = True
    if args.balanced_sampler:
        sample_weights = train_df["label"].map(lambda label: 1.0 / class_counts[label]).values
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
        shuffle_train = False

    pin_memory = device.type == "cuda"
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler, shuffle=shuffle_train, num_workers=2, pin_memory=pin_memory)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=pin_memory)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=pin_memory)

    counts = train_df["label"].value_counts().reindex(class_names).fillna(1).values.astype(float)
    weights = counts.sum() / (len(counts) * counts)
    weights = np.minimum(weights, args.class_weight_cap)
    class_weights = torch.tensor(weights, dtype=torch.float32).to(device)
    criterion = FocalLoss(gamma=args.focal_gamma, weight=class_weights, label_smoothing=args.label_smoothing)

    histories, summaries = [], []
    for model_name in args.models:
        print(f"\nTraining {model_name}")
        model = create_model(model_name, len(class_names), pretrained=not args.no_pretrained).to(device)

        set_backbone_trainable(model, trainable=False)
        optimizer = torch.optim.AdamW(
            filter(lambda parameter: parameter.requires_grad, model.parameters()),
            lr=args.lr_head,
            weight_decay=args.weight_decay,
        )
        head_history, head_path, _ = train_phase(
            model, train_loader, val_loader, criterion, optimizer, args.head_epochs, "head", model_name, out_dir, device, class_names, args.early_stop_patience, args.grad_clip
        )

        if head_path.exists():
            model.load_state_dict(torch.load(head_path, map_location=device))
        set_backbone_trainable(model, trainable=True)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr_finetune, weight_decay=args.weight_decay)
        finetune_history, best_path, best_val_f1 = train_phase(
            model, train_loader, val_loader, criterion, optimizer, args.finetune_epochs, "finetune", model_name, out_dir, device, class_names, args.early_stop_patience, args.grad_clip
        )

        model.load_state_dict(torch.load(best_path, map_location=device))
        test_metrics = evaluate(model, test_loader, criterion, device, class_names, save_rows=True)
        size_mb = best_path.stat().st_size / (1024 * 1024)

        summary = {
            "model": model_name,
            "best_val_f1_macro": best_val_f1,
            "test_accuracy": test_metrics["accuracy"],
            "test_precision_macro": test_metrics["precision_macro"],
            "test_recall_macro": test_metrics["recall_macro"],
            "test_f1_macro": test_metrics["f1_macro"],
            "model_size_mb": round(size_mb, 2),
            "best_model_path": str(best_path),
            "device_used": str(device),
            "device_name": device_name,
            "classification_report": test_metrics["classification_report"],
            "confusion_matrix": test_metrics["confusion_matrix"],
            "prediction_rows": test_metrics["prediction_rows"],
        }
        summaries.append(summary)
        histories.extend([head_history, finetune_history])
        print(json.dumps({k: v for k, v in summary.items() if k not in {"classification_report", "confusion_matrix", "prediction_rows"}}, indent=2))

    history_df = pd.concat(histories, ignore_index=True)
    summary_df = pd.DataFrame([{k: v for k, v in row.items() if k not in {"classification_report", "confusion_matrix", "prediction_rows"}} for row in summaries])
    summary_df = summary_df.sort_values("best_val_f1_macro", ascending=False).reset_index(drop=True)
    best = next(row for row in summaries if row["model"] == summary_df.iloc[0]["model"])

    final_model_path = out_dir / "maizeguard_public_best_model.pt"
    shutil.copy(Path(best["best_model_path"]), final_model_path)
    pd.DataFrame(best["prediction_rows"] if "prediction_rows" in best else []).to_csv(out_dir / "test_predictions_and_errors.csv", index=False)

    metadata = {
        "model_name": best["model"],
        "class_names": class_names,
        "image_size": args.image_size,
        "normalization_mean": [0.485, 0.456, 0.406],
        "normalization_std": [0.229, 0.224, 0.225],
        "selected_by": "highest validation macro F1 among compared PyTorch/timm models",
        "device_used": best.get("device_used", str(device)),
        "device_name": best.get("device_name", device_name),
        "confidence_threshold": 0.60,
        "risk_priority_threshold": 0.55,
    }

    history_df.to_csv(out_dir / "training_history.csv", index=False)
    summary_df.to_csv(out_dir / "model_comparison_summary.csv", index=False)
    json_summaries = [
        {key: value for key, value in row.items() if key != "prediction_rows"}
        for row in summaries
    ]
    (out_dir / "model_comparison_summary.json").write_text(json.dumps({"ranking": json_summaries, "recommended_model": best["model"]}, indent=2), encoding="utf-8")
    (out_dir / "maizeguard_model_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Recommended model: {best['model']}")
    print(f"Final model: {final_model_path}")


if __name__ == "__main__":
    main()
