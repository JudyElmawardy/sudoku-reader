"""
Calibrated classifier analysis without sklearn dependency.
CNN vs multinomial logistic regression (PyTorch), confusion matrix, ECE.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from nn.model import DigitCNN, set_temperature


def expected_calibration_error(probs: np.ndarray, y: np.ndarray, n_bins: int = 15) -> float:
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    accs = (predictions == y).astype(float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (confidences > bins[i]) & (confidences <= bins[i + 1])
        if mask.sum() == 0:
            continue
        ece += (mask.sum() / len(y)) * abs(accs[mask].mean() - confidences[mask].mean())
    return float(ece)


def confusion_matrix_np(y_true, y_pred, n_classes=9):
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    return cm


def load_cnn(model_path: Path, device: str = "cpu"):
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    model = DigitCNN()
    model.load_state_dict(ckpt["state_dict"])
    set_temperature(model, float(ckpt.get("temperature", 1.0)))
    model.to(device).eval()
    return model, ckpt


def cnn_predict(model, X, device="cpu", batch=512):
    probs_list = []
    with torch.no_grad():
        for i in range(0, len(X), batch):
            xb = torch.from_numpy(X[i:i + batch]).unsqueeze(1).float().to(device)
            probs_list.append(model.predict_proba(xb).cpu().numpy())
    return np.concatenate(probs_list, axis=0)


class SoftmaxRegression(nn.Module):
    def __init__(self, n_features=784, n_classes=9):
        super().__init__()
        self.linear = nn.Linear(n_features, n_classes)

    def forward(self, x):
        return self.linear(x)


def train_logistic(X_train, y_train, X_val, y_val, epochs=30, lr=0.1, device="cpu"):
    model = SoftmaxRegression().to(device)
    opt = optim.SGD(model.parameters(), lr=lr, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss()
    Xtr = torch.from_numpy(X_train.reshape(len(X_train), -1)).float().to(device)
    ytr = torch.from_numpy(y_train).long().to(device)
    Xva = torch.from_numpy(X_val.reshape(len(X_val), -1)).float().to(device)
    yva = torch.from_numpy(y_val).long().to(device)

    best_state, best_acc = None, 0.0
    for ep in range(epochs):
        model.train()
        # mini-batches
        perm = torch.randperm(len(Xtr))
        for i in range(0, len(Xtr), 256):
            idx = perm[i:i + 256]
            opt.zero_grad()
            loss = crit(model(Xtr[idx]), ytr[idx])
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            pred = model(Xva).argmax(1)
            acc = (pred == yva).float().mean().item()
        if acc > best_acc:
            best_acc = acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model


def logistic_predict(model, X, device="cpu"):
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(X.reshape(len(X), -1)).float().to(device))
        return torch.softmax(logits, dim=1).cpu().numpy()


def main():
    data_dir = Path("artifacts")
    model_dir = Path("models")
    fig_dir = Path("figures")
    fig_dir.mkdir(exist_ok=True)

    data = np.load(data_dir / "mnist_1_9.npz")
    X_train, y_train = data["X_train"], data["y_train"]
    X_val, y_val = data["X_val"], data["y_val"]
    X_test, y_test = data["X_test"], data["y_test"]

    device = "cpu"
    model, ckpt = load_cnn(model_dir / "digit_cnn.pt", device=device)

    probs_cnn = cnn_predict(model, X_test, device=device)
    pred_cnn = probs_cnn.argmax(1)
    acc_cnn = float((pred_cnn == y_test).mean())
    cm_cnn = confusion_matrix_np(y_test, pred_cnn)
    ece_cnn = expected_calibration_error(probs_cnn, y_test)
    nll_cnn = float(-np.log(np.clip(probs_cnn[np.arange(len(y_test)), y_test], 1e-12, 1)).mean())

    print("=== CNN ===")
    print(f"  Accuracy : {acc_cnn:.4f}")
    print(f"  ECE      : {ece_cnn:.4f}")
    print(f"  NLL      : {nll_cnn:.4f}")
    print(f"  Temperature: {float(ckpt.get('temperature', 1.0)):.3f}")
    print("  Confusion matrix (rows=true 1-9, cols=pred 1-9):")
    print(cm_cnn)

    print("\n=== Logistic Regression (softmax) baseline ===")
    rng = np.random.default_rng(42)
    n_tr = min(12000, len(X_train))
    idx = rng.choice(len(X_train), n_tr, replace=False)
    n_va = min(2000, len(X_val))
    idxv = rng.choice(len(X_val), n_va, replace=False)

    lr_model = train_logistic(X_train[idx], y_train[idx], X_val[idxv], y_val[idxv],
                              epochs=25, device=device)
    probs_lr = logistic_predict(lr_model, X_test, device=device)
    pred_lr = probs_lr.argmax(1)
    acc_lr = float((pred_lr == y_test).mean())
    cm_lr = confusion_matrix_np(y_test, pred_lr)
    ece_lr = expected_calibration_error(probs_lr, y_test)
    nll_lr = float(-np.log(np.clip(probs_lr[np.arange(len(y_test)), y_test], 1e-12, 1)).mean())

    print(f"  Accuracy : {acc_lr:.4f}")
    print(f"  ECE      : {ece_lr:.4f}")
    print(f"  NLL      : {nll_lr:.4f}")

    torch.save(lr_model.state_dict(), model_dir / "logistic_baseline.pt")

    metrics = {
        "cnn": {
            "accuracy": acc_cnn,
            "ece": ece_cnn,
            "nll": nll_cnn,
            "temperature": float(ckpt.get("temperature", 1.0)),
            "confusion_matrix": cm_cnn.tolist(),
        },
        "logistic_regression": {
            "accuracy": acc_lr,
            "ece": ece_lr,
            "nll": nll_lr,
            "confusion_matrix": cm_lr.tolist(),
        },
        "improvement_pp": (acc_cnn - acc_lr) * 100,
    }
    with open(model_dir / "classifier_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nSaved {model_dir / 'classifier_metrics.json'}")

    # confusion heatmaps (matplotlib only)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, cm, title in [
        (axes[0], cm_cnn, f"CNN (acc={acc_cnn:.3f}, ECE={ece_cnn:.3f})"),
        (axes[1], cm_lr, f"Logistic (acc={acc_lr:.3f}, ECE={ece_lr:.3f})"),
    ]:
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(9)); ax.set_yticks(range(9))
        ax.set_xticklabels(range(1, 10)); ax.set_yticklabels(range(1, 10))
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        ax.set_title(title)
        for i in range(9):
            for j in range(9):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=7)
        fig.colorbar(im, ax=ax, fraction=0.046)
    plt.tight_layout()
    plt.savefig(fig_dir / "classifier_confusion.png", dpi=130)
    plt.close()
    print(f"Saved {fig_dir / 'classifier_confusion.png'}")

    # reliability diagram
    conf = probs_cnn.max(1)
    correct = (pred_cnn == y_test).astype(float)
    bins = np.linspace(0, 1, 11)
    bin_acc, bin_conf = [], []
    for i in range(10):
        m = (conf > bins[i]) & (conf <= bins[i + 1])
        if m.sum() == 0:
            bin_acc.append(0); bin_conf.append((bins[i] + bins[i + 1]) / 2)
        else:
            bin_acc.append(correct[m].mean()); bin_conf.append(conf[m].mean())
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "k--", label="perfect")
    ax.bar(bin_conf, bin_acc, width=0.08, alpha=0.7, label="CNN")
    ax.set_xlabel("Confidence"); ax.set_ylabel("Accuracy")
    ax.set_title(f"Reliability diagram (ECE={ece_cnn:.4f})")
    ax.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / "classifier_reliability.png", dpi=130)
    plt.close()
    print(f"Saved {fig_dir / 'classifier_reliability.png'}")


if __name__ == "__main__":
    main()
