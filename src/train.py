"""
train.py
================
Training pipeline for image reconstruction models (e.g., U-Net).

Integrates hyperparameters from config.py and supports:
    - Resume from checkpoint
    - Best and last model saving
    - Early stopping and scheduler
    - Mixed precision (for speed)
"""

import os
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
from src.metrics import compute_metrics
from src.utils import get_device, ensure_dir_exists, log_message, timer


# ============================================================
#  Checkpoint Utilities
# ============================================================
def save_checkpoint(model, optimizer, epoch, path, extra=None):
    """Save a training checkpoint containing model, optimizer, and optional extras."""
    ensure_dir_exists(path)
    state = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    if isinstance(extra, dict):
        state.update(extra)
    torch.save(state, path)


def load_checkpoint(model, optimizer, path, map_location=None):
    """Load a checkpoint and restore model and optimizer state."""
    ckpt = torch.load(path, map_location=map_location or "cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    print(f"[train] Loaded checkpoint from {path} (epoch {ckpt.get('epoch', '?')})")
    return ckpt


# ============================================================
#  Training & Validation Epochs
# ============================================================
def train_one_epoch(model, loader, optimizer, loss_fn, device, scaler):
    """Run one epoch of training with backend-aware mixed precision."""
    model.train()
    running_loss = 0.0

    # Mixed precision only for CUDA
    use_amp = device.type == "cuda"

    for X_batch, Y_batch in loader:
        X_batch, Y_batch = X_batch.to(device, non_blocking=True), Y_batch.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        if use_amp:
            with torch.cuda.amp.autocast(dtype=torch.float16):
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


def validate_one_epoch(model, loader, loss_fn, device, epoch, log_file=None):
    """Validate model and compute reconstruction metrics (with AMP on CUDA)."""
    model.eval()
    val_loss = 0.0
    all_metrics = {"mse": 0, "mae": 0, "psnr": 0, "ssim": 0}

    use_amp = device.type == "cuda"

    with torch.no_grad():
        iterator = tqdm(loader, desc=f"[val] Epoch {epoch}", leave=False)
        for X_batch, Y_batch in iterator:
            X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)

            # Use AMP only on CUDA
            if use_amp:
                with torch.cuda.amp.autocast(dtype=torch.float16):
                    preds = model(X_batch)
                    loss = loss_fn(preds, Y_batch)
            else:
                preds = model(X_batch)
                loss = loss_fn(preds, Y_batch)

            val_loss += loss.item() * X_batch.size(0)

            metrics = compute_metrics(preds, Y_batch)
            for k in all_metrics:
                all_metrics[k] += metrics[k] * X_batch.size(0)

    total = len(loader.dataset)
    for k in all_metrics:
        all_metrics[k] /= total

    avg_loss = val_loss / total

    return avg_loss, all_metrics


# ============================================================
#  Main Training Function
# ============================================================
def train_model(
    model,
    train_loader,
    val_loader,
    config,
    save_path="results/checkpoints/best_model.pt",
    save_last_path="results/checkpoints/last_model.pt",
    resume_path=None,
):
    """
    Trains a model using parameters from config, with optional resume capability.
    """
    device = get_device()
    model = model.to(device)
    scaler = GradScaler(enabled=(device.type != "cpu"))

    # --- Parse config ---
    train_cfg = config.get("training", {})
    lr = train_cfg.get("lr", 1e-3)
    weight_decay = train_cfg.get("weight_decay", 0.0)
    optimizer_name = train_cfg.get("optimizer", "Adam").lower()
    loss_name = train_cfg.get("loss_fn", "MSE").upper()
    scheduler_name = train_cfg.get("scheduler", None)
    num_epochs = train_cfg.get("epochs", 20)
    patience = train_cfg.get("patience", 5)
    log_file = train_cfg.get("log_file", None)

    # --- Loss function ---
    if loss_name == "L1":
        loss_fn = nn.L1Loss()
    elif loss_name == "BCE":
        loss_fn = nn.BCELoss()
    else:
        loss_fn = nn.MSELoss()

    # --- Optimizer ---
    if optimizer_name == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == "rmsprop":
        optimizer = torch.optim.RMSprop(model.parameters(), lr=lr)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # --- Scheduler ---
    scheduler = None
    if scheduler_name == "ReduceLROnPlateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=3)
    elif scheduler_name == "CosineAnnealingLR":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

    # --- Resume from checkpoint ---
    start_epoch = 1
    best_val_loss = float("inf")
    if resume_path and os.path.exists(resume_path):
        ckpt = load_checkpoint(model, optimizer, resume_path, map_location=device)
        start_epoch = ckpt.get("epoch", 1) + 1
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        print(f"[train] Resuming from epoch {start_epoch}")

    # --- Training loop ---
    epochs_no_improve = 0
    history = {"train_loss": [], "val_loss": [], "psnr": [], "ssim": []}

    ensure_dir_exists(save_path)
    ensure_dir_exists(save_last_path)
    print(f"[train] Starting training for {num_epochs} epochs on {device}.")

    for epoch in range(start_epoch, num_epochs + 1):
        with timer(f"Epoch {epoch}"):
            train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device, scaler)
            val_loss, metrics = validate_one_epoch(model, val_loader, loss_fn, device, epoch, log_file)

            # --- Log epoch summary ---
            summary = (f"[train] Epoch {epoch:03d} | Train: {train_loss:.6f} | "
                       f"Val: {val_loss:.6f} | PSNR: {metrics['psnr']:.2f} | SSIM: {metrics['ssim']:.4f}")
            print(summary)
            if log_file:
                log_message(summary, log_file)

            # --- Save last model ---
            save_checkpoint(model, optimizer, epoch, save_last_path,
                            extra={"config": config, "history": history, "best_val_loss": best_val_loss})

            # --- Track best model ---
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
                save_checkpoint(model, optimizer, epoch, save_path,
                                extra={"config": config, "history": history, "best_val_loss": best_val_loss})
                msg = f"[train] New best model saved at epoch {epoch} → {save_path}"
                print(msg)
                if log_file:
                    log_message(msg, log_file)
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    stop_msg = "[train] Early stopping triggered."
                    print(stop_msg)
                    if log_file:
                        log_message(stop_msg, log_file)
                    break

            # --- Scheduler step ---
            if scheduler:
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(val_loss)
                else:
                    scheduler.step()

    final_msg = f"[train] Training complete. Best Val Loss: {best_val_loss:.5f}"
    print(final_msg)
    if log_file:
        log_message(final_msg, log_file)
    return model, history


# ============================================================
#  Test Evaluation Function
# ============================================================
def evaluate_model(model, test_loader, config):
    """Evaluate model on the test set with AMP on CUDA for speed."""
    device = get_device()
    model = model.to(device)
    model.eval()
    loss_fn = nn.MSELoss()

    total_loss, total_metrics = 0, {"mse": 0, "mae": 0, "psnr": 0, "ssim": 0}
    use_amp = device.type == "cuda"

    with torch.no_grad():
        for X, Y in tqdm(test_loader, desc="[test]", leave=False):
            X, Y = X.to(device), Y.to(device)

            # Use AMP only on CUDA
            if use_amp:
                with torch.cuda.amp.autocast(dtype=torch.float16):
                    preds = model(X)
                    loss = loss_fn(preds, Y)
            else:
                preds = model(X)
                loss = loss_fn(preds, Y)

            total_loss += loss.item() * X.size(0)

            metrics = compute_metrics(preds, Y)
            for k in total_metrics:
                total_metrics[k] += metrics[k] * X.size(0)

    n = len(test_loader.dataset)
    for k in total_metrics:
        total_metrics[k] /= n

    msg = (f"[test] Loss: {total_loss/n:.6f} | "
           f"PSNR: {total_metrics['psnr']:.2f} | SSIM: {total_metrics['ssim']:.4f}")
    print(msg)
    log_file = config["training"].get("log_file")
    if log_file:
        log_message(msg, log_file)
    return total_metrics
