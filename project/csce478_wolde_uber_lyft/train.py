"""Training loop and checkpointing for PyTorch classifiers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from config import BATCH_SIZE, EARLY_STOPPING_PATIENCE, EPOCHS, LR, WEIGHT_DECAY


def _loader(X: np.ndarray, y: np.ndarray, shuffle: bool) -> DataLoader:
    ds = TensorDataset(
        torch.from_numpy(X),
        torch.from_numpy(y).long(),
    )
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle, drop_last=False)


def train_one_model(
    model: nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    device: torch.device,
    epochs: int = EPOCHS,
    lr: float = LR,
    weight_decay: float = WEIGHT_DECAY,
) -> dict:
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    crit = nn.CrossEntropyLoss()
    train_loader = _loader(X_train, y_train, shuffle=True)
    val_loader = _loader(X_val, y_val, shuffle=False)

    best_val = float("inf")
    best_state = None
    patience = 0

    history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}

    for epoch in range(epochs):
        model.train()
        losses = []
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            logits = model(xb)
            loss = crit(logits, yb)
            loss.backward()
            opt.step()
            losses.append(loss.item())
        train_loss = float(np.mean(losses))

        model.eval()
        vlosses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                vlosses.append(crit(logits, yb).item())
        val_loss = float(np.mean(vlosses))

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= EARLY_STOPPING_PATIENCE:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return {"history": history, "best_val_loss": best_val}


def save_model(
    path: Path,
    model: nn.Module,
    meta: dict,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "meta": meta}, path)


def predict_labels(model: nn.Module, X: np.ndarray, device: torch.device) -> np.ndarray:
    model.eval()
    loader = _loader(X, np.zeros(len(X), dtype=np.int64), shuffle=False)
    out: list[np.ndarray] = []
    with torch.no_grad():
        for xb, _ in loader:
            xb = xb.to(device)
            logits = model(xb)
            pred = logits.argmax(dim=1).cpu().numpy()
            out.append(pred)
    return np.concatenate(out, axis=0)
