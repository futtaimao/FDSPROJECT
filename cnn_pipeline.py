"""
cnn_pipeline.py
===============
Deep-learning pipeline for the Malimg dataset.

    Step 12 - Load and transform the images (128x128, normalised).
    Step 13 - Split data and build DataLoaders.
    Step 14 - Build the CNN (three conv blocks + FC classifier).
    Step 15 - Train for 5 epochs.
    Step 16 - Evaluate on the held-out test set.
"""

import os
import sys
import time
import numpy as np
import torch
import torch.nn     as nn
import torch.optim  as optim

from torch.utils.data   import DataLoader, random_split
from torchvision        import datasets, transforms

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from fatima_evaluation import evaluate_model


HERE         = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
MALIMG_DIR   = os.path.join(
    PROJECT_ROOT,
    "databases",
    "Malimg (Original)",
    "malimg_paper_dataset_imgs",
)
RESULTS_DIR  = os.path.abspath(os.path.join(HERE, "..", "results"))
CM_DIR       = os.path.join(RESULTS_DIR, "confusion_matrices")
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE   = 32
IMG_SIZE     = 128
EPOCHS       = 5
LR           = 1e-3
RANDOM_STATE = 42

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(CM_DIR,      exist_ok=True)


# ---------------------------------------------------------------------------
# Step 12 - Data loading and transforms
# ---------------------------------------------------------------------------
def get_transforms():
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])


# ---------------------------------------------------------------------------
# Step 13 - Split and DataLoaders
# ---------------------------------------------------------------------------
def build_loaders():
    tfm = get_transforms()
    dataset = datasets.ImageFolder(MALIMG_DIR, transform=tfm)
    n       = len(dataset)
    ntr     = int(0.8 * n)
    nte     = n - ntr
    g       = torch.Generator().manual_seed(RANDOM_STATE)
    train_ds, test_ds = random_split(dataset, [ntr, nte], generator=g)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=0)

    print(f"[12-13] Loaded {n} images, {len(dataset.classes)} classes "
          f"-> train={ntr}, test={nte}")
    return train_loader, test_loader, dataset.classes


# ---------------------------------------------------------------------------
# Step 14 - Model
# ---------------------------------------------------------------------------
class MalwareCNN(nn.Module):
    """Three conv blocks followed by a fully-connected classifier."""

    def __init__(self, num_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1: 128 -> 64
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            # Block 2: 64 -> 32
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            # Block 3: 32 -> 16
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 16 * 16, 256), nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


# ---------------------------------------------------------------------------
# Step 15 - Training
# ---------------------------------------------------------------------------
def train_cnn(model, loader, epochs=EPOCHS):
    crit  = nn.CrossEntropyLoss()
    opt   = optim.Adam(model.parameters(), lr=LR)
    model.to(DEVICE)
    history = []

    for e in range(1, epochs + 1):
        model.train()
        t0, run_loss, correct, total = time.time(), 0.0, 0, 0
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            out   = model(xb)
            loss  = crit(out, yb)
            loss.backward()
            opt.step()

            run_loss += loss.item() * xb.size(0)
            correct  += (out.argmax(1) == yb).sum().item()
            total    += xb.size(0)

        mean_loss = run_loss / total
        acc       = correct / total
        history.append({"epoch": e, "loss": mean_loss, "acc": acc})
        print(f"[15] Epoch {e}/{epochs}  "
              f"loss={mean_loss:.4f}  acc={acc:.4f}  "
              f"({time.time() - t0:.1f}s)")

    return history


# ---------------------------------------------------------------------------
# Step 16 - Evaluation
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate_cnn(model, loader, class_names, save_dir):
    model.eval()
    model.to(DEVICE)
    y_true, y_pred = [], []
    for xb, yb in loader:
        xb = xb.to(DEVICE)
        out = model(xb)
        y_pred.extend(out.argmax(1).cpu().numpy())
        y_true.extend(yb.numpy())
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return evaluate_model(y_true, y_pred, class_names, "CNN", save_dir,
                          print_report=False), y_true, y_pred


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print(f"Device: {DEVICE}")
    torch.manual_seed(RANDOM_STATE)

    train_loader, test_loader, class_names = build_loaders()
    model = MalwareCNN(num_classes=len(class_names))
    print("[14] Model built:")
    print(model)

    history = train_cnn(model, train_loader, epochs=EPOCHS)

    torch.save(model.state_dict(),
               os.path.join(RESULTS_DIR, "cnn_model.pt"))
    print(f"[15] CNN weights saved -> cnn_model.pt")

    result, y_true, y_pred = evaluate_cnn(
        model, test_loader, class_names, CM_DIR)
    print("\n[16] CNN evaluation")
    for k in ("accuracy", "precision", "recall", "f1"):
        print(f"    {k:<9}: {result[k]:.4f}")

    import pandas as pd
    pd.DataFrame([{
        "model":     "CNN",
        "accuracy":  result["accuracy"],
        "precision": result["precision"],
        "recall":    result["recall"],
        "f1":        result["f1"],
        "cm_path":   result["cm_path"],
    }]).to_csv(os.path.join(RESULTS_DIR, "cnn_results.csv"), index=False)

    pd.DataFrame(history).to_csv(
        os.path.join(RESULTS_DIR, "cnn_training_history.csv"), index=False)

    mis = result["misclassifications"]
    if not mis.empty:
        mis.to_csv(os.path.join(RESULTS_DIR, "cnn_misclassifications.csv"),
                   index=False)
        print("\nTop misclassifications:")
        print(mis.to_string(index=False))


if __name__ == "__main__":
    main()
