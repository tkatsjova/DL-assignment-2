"""Shared helpers used by both intra and cross evaluators."""

from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from src.data.data_loading import ID_TO_LABEL
from src.data.dataset import MEGWindowDataset
from src.models.train import get_device, get_model

CLASS_NAMES = [ID_TO_LABEL[i] for i in range(4)]
BATCH_SIZE  = 32


def load_model(name: str, checkpoint_path: Path, device: torch.device) -> tuple[nn.Module, float, int]:
    model = get_model(name, device)
    ckpt  = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt.get("val_accuracy", 0.0), ckpt.get("epoch", 0)


def find_checkpoint(output_dir: Path, prefix: str, name: str) -> Path | None:
    matches = list(output_dir.glob(f"best_{prefix}_{name}_lr*.pt"))
    if not matches:
        return None
    if len(matches) == 1:
        return matches[0]

    best_path, best_acc = None, -1.0
    for path in matches:
        try:
            ckpt = torch.load(path, map_location="cpu", weights_only=False)
            acc  = float(ckpt.get("val_accuracy", 0.0))
            if acc > best_acc:
                best_acc = acc
                best_path = path
        except Exception as exc:
            print(f"  [warn] could not read {path.name}: {exc}")
    print(f"  [checkpoint] selected {best_path.name}  (val_acc={best_acc:.4f})")
    return best_path


def predict(model: nn.Module, dataset: MEGWindowDataset, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for start in range(0, len(dataset), BATCH_SIZE):
            end    = min(start + BATCH_SIZE, len(dataset))
            x      = torch.from_numpy(dataset.X[start:end]).to(device)
            logits = model(x)
            preds.extend(logits.argmax(dim=1).cpu().numpy())
            labels.extend(dataset.y[start:end])
    return np.array(preds), np.array(labels)


def majority_vote(preds: np.ndarray, labels: np.ndarray, file_names: list[str]) -> tuple[np.ndarray, np.ndarray]:
    file_preds: dict[str, list] = {}
    file_labels: dict[str, int] = {}
    for pred, label, fname in zip(preds, labels, file_names):
        file_preds.setdefault(fname, []).append(pred)
        file_labels[fname] = label
    final_preds, final_labels = [], []
    for fname in file_preds:
        final_preds.append(Counter(file_preds[fname]).most_common(1)[0][0])
        final_labels.append(file_labels[fname])
    return np.array(final_preds), np.array(final_labels)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, label: str = "") -> dict:
    report = classification_report(
        y_true, y_pred, target_names=CLASS_NAMES, zero_division=0, output_dict=True
    )
    result = {
        "accuracy":         round(float(accuracy_score(y_true, y_pred)), 4),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "per_class": {
            cls: {
                "precision": round(report[cls]["precision"], 4),
                "recall":    round(report[cls]["recall"],    4),
                "f1":        round(report[cls]["f1-score"],  4),
            }
            for cls in CLASS_NAMES
        },
    }

    print(f"\n{'='*60}")
    if label:
        print(f"  {label}")
    print(f"{'='*60}")
    print(f"  Window-level accuracy: {result['accuracy']:.4f}")
    print(f"  Per-class F1:")
    for cls in CLASS_NAMES:
        print(f"    {cls:<20} {result['per_class'][cls]['f1']:.4f}")
    cm = np.array(result["confusion_matrix"])
    print(f"  Confusion matrix (rows=true, cols=pred):")
    print(f"  Classes: {CLASS_NAMES}")
    print(f"  {cm}")

    return result
