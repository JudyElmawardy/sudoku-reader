"""
Train digit classifier (1-9) and a logistic-regression baseline.
Also perform temperature scaling for calibration.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from nn.model import DigitCNN, set_temperature

try:
    from sklearn.linear_model import LogisticRegression
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


def load_mnist_1_9(data_dir: Path, seed: int = 42):
    """Load preprocessed tensors created by make_data.py."""
    data = np.load(data_dir / "mnist_1_9.npz")
    X_train = data["X_train"]
    y_train = data["y_train"]
    X_val = data["X_val"]
    y_val = data["y_val"]
    X_test = data["X_test"]
    y_test = data["y_test"]
    return X_train, y_train, X_val, y_val, X_test, y_test


def train_cnn(
    X_train, y_train, X_val, y_val,
    epochs: int = 8,
    batch_size: int = 128,
    lr: float = 1e-3,
    device: str = "cpu",
    seed: int = 42,
):
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = DigitCNN().to(device)
    opt = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    train_ds = TensorDataset(
        torch.from_numpy(X_train).unsqueeze(1).float(),
        torch.from_numpy(y_train).long(),
    )
    val_ds = TensorDataset(
        torch.from_numpy(X_val).unsqueeze(1).float(),
        torch.from_numpy(y_val).long(),
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=256)

    best_val = 0.0
    best_state = None

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            opt.step()
            total_loss += loss.item() * len(xb)

        # validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb).argmax(1)
                correct += (pred == yb).sum().item()
                total += len(yb)
        val_acc = correct / total
        print(f"Epoch {epoch+1}/{epochs}  loss={total_loss/len(train_ds):.4f}  val_acc={val_acc:.4f}")
        if val_acc > best_val:
            best_val = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    return model, best_val


def temperature_scale(model: DigitCNN, X_val, y_val, device: str = "cpu"):
    """Simple grid search for temperature that minimises NLL on validation set."""
    model.eval()
    xb = torch.from_numpy(X_val).unsqueeze(1).float().to(device)
    yb = torch.from_numpy(y_val).long().to(device)
    with torch.no_grad():
        logits = model.conv1  # just to get device
        # recompute raw logits (temperature=1)
        set_temperature(model, 1.0)
        raw_logits = model(xb)

    best_t, best_nll = 1.0, float("inf")
    for t in np.linspace(0.5, 5.0, 46):
        scaled = raw_logits / t
        nll = nn.functional.cross_entropy(scaled, yb).item()
        if nll < best_nll:
            best_nll = nll
            best_t = t
    set_temperature(model, best_t)
    print(f"Best temperature: {best_t:.3f}  (val NLL={best_nll:.4f})")
    return best_t


def evaluate(model, X, y, device="cpu"):
    model.eval()
    xb = torch.from_numpy(X).unsqueeze(1).float().to(device)
    with torch.no_grad():
        probs = model.predict_proba(xb).cpu().numpy()
    preds = probs.argmax(1)
    acc = float((preds == y).mean())
    # confusion matrix
    cm = np.zeros((9, 9), dtype=int)
    for t, p in zip(y, preds):
        cm[t, p] += 1
    ece = expected_calibration_error(probs, y)
    return acc, cm, ece, probs


def expected_calibration_error(probs, y, n_bins=15):
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    accs = (predictions == y).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (confidences > bins[i]) & (confidences <= bins[i + 1])
        if mask.sum() > 0:
            ece += mask.sum() * abs(accs[mask].mean() - confidences[mask].mean())
    return ece / len(y)


def train_logistic(X_train, y_train, X_test, y_test):
    if not HAS_SKLEARN:
        # fallback: nearest class-mean prototype
        means = np.stack([X_train[y_train == d].mean(0) for d in range(9)])
        Xte = X_test.reshape(len(X_test), -1)
        mf = means.reshape(9, -1)
        preds = ((Xte[:, None, :] - mf[None, :, :]) ** 2).sum(-1).argmin(1)
        acc = float((preds == y_test).mean())
        return {"type": "prototype", "means": means}, acc
    Xtr = X_train.reshape(len(X_train), -1)
    Xte = X_test.reshape(len(X_test), -1)
    clf = LogisticRegression(max_iter=200, multi_class="multinomial", solver="lbfgs", C=1.0)
    clf.fit(Xtr, y_train)
    preds = clf.predict(Xte)
    acc = float((preds == y_test).mean())
    return clf, acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-dir", type=str, default="artifacts")
    parser.add_argument("--model-dir", type=str, default="models")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    X_train, y_train, X_val, y_val, X_test, y_test = load_mnist_1_9(data_dir, args.seed)
    # labels are already 0-8 for digits 1-9
    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    print("\n=== Training CNN ===")
    model, val_acc = train_cnn(
        X_train, y_train, X_val, y_val,
        epochs=args.epochs, device=args.device, seed=args.seed,
    )
    print(f"Best val accuracy: {val_acc:.4f}")

    print("\n=== Temperature scaling ===")
    temp = temperature_scale(model, X_val, y_val, device=args.device)

    print("\n=== Test evaluation (CNN) ===")
    acc, cm, ece, _ = evaluate(model, X_test, y_test, device=args.device)
    print(f"Test accuracy: {acc:.4f}")
    print(f"ECE: {ece:.4f}")
    print("Confusion matrix (rows=true 1-9, cols=pred 1-9):")
    print(cm)

    print("\n=== Baseline (logistic or prototype) ===")
    clf, lr_acc = train_logistic(X_train, y_train, X_test, y_test)
    print(f"Baseline test accuracy: {lr_acc:.4f}")

    # save
    torch.save({
        "state_dict": model.state_dict(),
        "temperature": temp,
        "test_acc": acc,
        "ece": ece,
        "val_acc": val_acc,
    }, model_dir / "digit_cnn.pt")
    if HAS_SKLEARN:
        import joblib
        joblib.dump(clf, model_dir / "logistic_baseline.joblib")
    elif isinstance(clf, dict) and "means" in clf:
        np.savez(model_dir / "prototype_baseline.npz", means=clf["means"])

    results = {
        "cnn_test_acc": float(acc),
        "cnn_ece": float(ece),
        "cnn_val_acc": float(val_acc),
        "baseline_test_acc": float(lr_acc),
        "temperature": float(temp),
    }
    with open(model_dir / "metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved models and metrics to", model_dir)


if __name__ == "__main__":
    main()