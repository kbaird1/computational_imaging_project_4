"""
train.py
================
Training pipeline for image reconstruction models (e.g., U-Net).

Integrates hyperparameters from config.py and supports:
    - Custom differentiable loss functions
    - Full metric evaluation from error_on_validation
    - Resume from checkpoint
    - Best and last model saving
    - Early stopping and scheduler
    - Mixed precision (for speed)
    - Automatic training report summary
"""

import os
import time
import torch
import numpy as np
import json

from tqdm import tqdm
from torch import amp
from src import metrics
from src.metrics import compute_metrics
from src.utils import get_device, ensure_dir_exists, log_message


# ============================================================
#  Utility: Checkpoints
# ============================================================
def save_checkpoint(model, optimizer, epoch, path, extra=None):
    """Save checkpoint containing model/optimizer states and optional metadata."""
    ensure_dir_exists(os.path.dirname(path))
    state = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    if isinstance(extra, dict):
        state.update(extra)
    torch.save(state, path)


def load_checkpoint(model, optimizer, path, map_location=None, log_file=None):
    """Load a checkpoint and restore model and optimizer state."""
    ckpt = torch.load(path, map_location=map_location or "cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    log_message(f"[train] Loaded checkpoint from {path} (epoch {ckpt.get('epoch', '?')})", log_file, console=True)
    return ckpt


def get_loss_function(name):
    """
    Retrieve differentiable loss function by name.

    All losses are sourced from src.metrics to ensure consistent computation
    between training, validation, and testing. Non-differentiable metrics are
    wrapped into torch tensors for safe backward compatibility.
    """

    name = name.lower()

    if name == "mse":
        return lambda y_pred, y_true: torch.tensor(
            metrics.mean_squared_error(y_true, y_pred), device=y_pred.device, requires_grad=True
        )

    elif name in ["l1", "mae"]:
        return lambda y_pred, y_true: torch.tensor(
            metrics.mean_absolute_error(y_true, y_pred), device=y_pred.device, requires_grad=True
        )

    elif name == "dssim":
        return lambda y_pred, y_true: torch.tensor(
            metrics.dssim(y_true, y_pred), device=y_pred.device, requires_grad=True
        )

    elif name == "gdl":
        return lambda y_pred, y_true: torch.tensor(
            metrics.gradient_difference_loss(y_true, y_pred), device=y_pred.device, requires_grad=True
        )

    elif name == "tv":
        return lambda y_pred, y_true: torch.tensor(
            metrics.total_variation(y_pred), device=y_pred.device, requires_grad=True
        )

    elif name == "lpips":
        return lambda y_pred, y_true: torch.tensor(
            metrics.lpips_score(y_true, y_pred), device=y_pred.device, requires_grad=True
        )

    elif name == "ms_ssim":
        return lambda y_pred, y_true: torch.tensor(
            metrics.multi_scale_ssim(y_true, y_pred), device=y_pred.device, requires_grad=True
        )

    else:
        raise ValueError(f"Unsupported loss function: {name}")


# ============================================================
#  One Epoch of Training / Validation
# ============================================================
def train_one_epoch(model, loader, optimizer, loss_fn, device, scaler):
    """Train model for one epoch with mixed precision."""
    model.train()
    running_loss = 0.0
    use_amp = device.type == "cuda"

    for X_batch, Y_batch in loader:
        X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
        optimizer.zero_grad(set_to_none=True)

        if use_amp:
            with amp.autocast(device_type=device.type, dtype=torch.float16):
                preds = model(X_batch)
                loss = loss_fn(preds, Y_batch)
        else:
            preds = model(X_batch)
            loss = loss_fn(preds, Y_batch)

        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        running_loss += loss.item() * X_batch.size(0)

    return running_loss / len(loader.dataset)


def validate_one_epoch(model, loader, loss_fn, device, epoch, metrics_list):
    """Validate model and compute all selected metrics individually (not averaged together)."""
    model.eval()
    val_loss = 0.0
    metric_sums = {m: 0.0 for m in metrics_list}
    total_samples = 0
    use_amp = device.type == "cuda"

    with torch.no_grad():
        for X_batch, Y_batch in tqdm(loader, desc=f"[val] Epoch {epoch}", leave=False):
            X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
            batch_size = X_batch.size(0)
            total_samples += batch_size

            if use_amp:
                with amp.autocast(device_type=device.type, dtype=torch.float16):
                    preds = model(X_batch)
                    loss = loss_fn(preds, Y_batch)
            else:
                preds = model(X_batch)
                loss = loss_fn(preds, Y_batch)

            val_loss += loss.item() * batch_size

            # Compute selected validation metrics for this batch
            batch_metrics = compute_metrics(preds, Y_batch, metrics_list)

            # Safely accumulate each metric individually
            for k, val in batch_metrics.items():
                if val is None or isinstance(val, str):
                    print(f"[validate] Skipping metric {k} with non-numeric value: {val}")
                    continue
                try:
                    metric_sums[k] += float(val) * batch_size
                except (TypeError, ValueError):
                    print(f"[validate] Warning: could not convert metric {k} value: {val}")
                    continue

    # Average loss and each metric across the entire validation set
    avg_loss = val_loss / total_samples
    avg_metrics = {k: metric_sums[k] / total_samples for k in metric_sums}

    return avg_loss, avg_metrics


# ============================================================
#  Main Training Function
# ============================================================
def train_model(model, train_loader, val_loader, config,
                save_path="results/checkpoints/best_model.pt",
                save_last_path="results/checkpoints/last_model.pt",
                resume_path=None,
                log_file=None):
    """
    Full training routine integrating metrics, loss functions, and early stopping.

    Returns
    -------
    model : torch.nn.Module (best model)
    history : dict with keys:
        - "train_loss", "val_loss"
        - "metrics": {metric_name: [values over epochs]}
        - "epoch_times": list of per-epoch durations
        - "total_time": total training duration (s)
    """
    device = get_device()
    model = model.to(device)
    scaler = amp.GradScaler(enabled=(device.type != "cpu"))

    # --- Parse config ---
    train_cfg = config.get("training", {})
    lr = train_cfg.get("lr", 1e-3)
    weight_decay = train_cfg.get("weight_decay", 0.0)
    optimizer_name = train_cfg.get("optimizer", "Adam").lower()
    loss_name = train_cfg.get("loss_fn", "mse")
    scheduler_name = train_cfg.get("scheduler", "None")
    num_epochs = train_cfg.get("epochs", 20)
    patience = train_cfg.get("patience", 5)
    metrics_list = train_cfg.get("error_on_validation", ["mse", "psnr", "ssim"])
    seed = train_cfg.get("seed", 42)
    torch.manual_seed(seed)

    # --- Initialize loss, optimizer, scheduler ---
    loss_fn = get_loss_function(loss_name)
    if optimizer_name == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == "rmsprop":
        optimizer = torch.optim.RMSprop(model.parameters(), lr=lr)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    scheduler = None
    if scheduler_name == "ReduceLROnPlateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=3)
    elif scheduler_name == "CosineAnnealingLR":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

    # --- Resume ---
    start_epoch = 1
    best_val_loss = float("inf")
    if resume_path and os.path.exists(resume_path):
        ckpt = load_checkpoint(model, optimizer, resume_path, map_location=device, log_file=log_file)
        start_epoch = ckpt.get("epoch", 1) + 1
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        log_message(f"[train] Resuming from epoch {start_epoch}", log_file, console=True)

    # --- Containers for history ---
    history = {
        "train_loss": [],
        "val_loss": [],
        "metrics": {m: [] for m in metrics_list},
        "epoch_times": [],
        "total_time": 0.0
    }

    ensure_dir_exists(os.path.dirname(save_path))
    log_message(f"[train] Training for {num_epochs} epochs on {device} (loss={loss_name}).", log_file, console=True)

    # --- Training loop ---
    start_time = time.time()
    epochs_no_improve = 0

    for epoch in range(start_epoch, num_epochs + 1):
        epoch_start = time.time()

        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device, scaler)
        val_loss, metrics = validate_one_epoch(model, val_loader, loss_fn, device, epoch, metrics_list)

        # --- Record ---
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        for m in metrics_list:
            history["metrics"][m].append(metrics[m])

        epoch_time = time.time() - epoch_start
        history["epoch_times"].append(epoch_time)

        # --- Log summary ---
        summary = f"[Epoch {epoch}] Train={train_loss:.5f}, Val={val_loss:.5f}, " + \
                  ", ".join([f"{m.upper()}={metrics[m]:.4f}" for m in metrics_list])
        log_message(summary, log_file, console=True)

        # --- Checkpoint saving ---
        save_checkpoint(model, optimizer, epoch, save_last_path,
                        extra={"config": config, "history": history, "best_val_loss": best_val_loss})
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            save_checkpoint(model, optimizer, epoch, save_path,
                            extra={"config": config, "history": history, "best_val_loss": best_val_loss})
            log_message(f"[train] New best model at epoch {epoch} saved → {save_path}", log_file, console=True)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                log_message(f"[train] Early stopping triggered at epoch {epoch}.", log_file, console=True)
                break

        # --- Scheduler step ---
        if scheduler:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()

    total_time = time.time() - start_time
    history["total_time"] = total_time

    log_message(f"[train] Training complete. Best Val Loss: {best_val_loss:.5f}. Total time: {total_time:.2f}s.", log_file, console=True)
    return model, history


# ============================================================
#  Evaluation & Report
# ============================================================
def evaluate_model(model, test_loader, config, log_file=None):
    """Evaluate final model on test set using same validation metrics."""
    device = get_device()
    model = model.to(device).eval()
    loss_fn = get_loss_function(config["training"].get("loss_fn", "mse"))
    metrics_list = config["training"].get("error_on_validation", ["mse", "psnr", "ssim"])

    total_loss = 0.0
    metric_sums = {m: 0.0 for m in metrics_list}
    use_amp = device.type == "cuda"

    with torch.no_grad():
        for X, Y in tqdm(test_loader, desc="[test]", leave=False):
            X, Y = X.to(device), Y.to(device)
            if use_amp:
                with amp.autocast(device_type=device.type, dtype=torch.float16):
                    preds = model(X)
                    loss = loss_fn(preds, Y)
            else:
                preds = model(X)
                loss = loss_fn(preds, Y)

            total_loss += loss.item() * X.size(0)
            metrics = compute_metrics(preds, Y, metrics_list)
            for k in metrics_list:
                metric_sums[k] += metrics[k] * X.size(0)

    n = len(test_loader.dataset)
    avg_metrics = {k: metric_sums[k] / n for k in metrics_list}
    avg_metrics["loss"] = total_loss / n

    log_message("[test] Final Test Metrics:", log_file, console=True)
    for k, v in avg_metrics.items():
        log_message(f"   {k.upper()}: {v:.4f}", log_file, console=True)
    return avg_metrics


# ============================================================
#  Training Report Writer
# ============================================================
def generate_training_report(output_path, config, history, test_metrics, log_file=None):
    """
    Write a summary report containing:
        - Model & training configuration
        - Average and best metrics
        - Timing stats (per-epoch & total)
        - Final test results
    """
    ensure_dir_exists(os.path.dirname(output_path))
    avg_epoch_time = np.mean(history["epoch_times"])
    best_epoch = int(np.argmin(history["val_loss"])) + 1

    report = {
        "model_name": config["model"]["model_name"],
        "config": config,
        "timing": {
            "avg_epoch_time_sec": float(avg_epoch_time),
            "total_training_time_sec": float(history["total_time"]),
            "num_epochs": len(history["train_loss"]),
            "best_epoch": best_epoch,
        },
        "training_summary": {
            "best_val_loss": float(np.min(history["val_loss"])),
            "final_val_loss": float(history["val_loss"][-1]),
            "final_train_loss": float(history["train_loss"][-1]),
            "final_validation_metrics": {
                k: float(v[-1]) for k, v in history["metrics"].items()
            },
        },
        "test_summary": test_metrics,
    }

    # Write to JSON
    with open(output_path, "w") as f:
        json.dump(report, f, indent=4)

    log_message(f"[report] Training report saved → {output_path}", log_file, console=True)
    return report
