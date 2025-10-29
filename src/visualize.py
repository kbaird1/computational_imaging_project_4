"""
visualize.py
================
Visualization utilities for single-pixel / U-Net image reconstruction.

Includes:
    - Plotting training & validation losses
    - Plotting all available validation metrics dynamically
    - Displaying side-by-side reconstructions
    - Visualizing pixel-wise error maps
    - Comparing metrics across multiple model runs
"""

import os
import torch
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


# ============================================================
# Utility: Ensure figure directory exists
# ============================================================
def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


# ============================================================
# 1. Plot Training & Validation History
# ============================================================
def plot_training_history(history, save_dir="results/figures", show=True):
    """
    Plot training/validation loss and all tracked validation metrics.

    Expected structure of history:
    {
        "train_loss": [...],
        "val_loss": [...],
        "metrics": {
            "psnr": [...],
            "ssim": [...],
            "mse": [...],
            ...
        }
    }
    """
    ensure_dir(save_dir)
    epochs = np.arange(1, len(history["train_loss"]) + 1)

    # ---- Loss Curves ----
    plt.figure(figsize=(6, 4))
    plt.plot(epochs, history["train_loss"], label="Train", linewidth=2)
    plt.plot(epochs, history["val_loss"], label="Validation", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training & Validation Loss")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/loss_curve.png", dpi=300)
    if show: plt.show(); plt.close()

    # ---- Validation Metrics ----
    if "metrics" in history and isinstance(history["metrics"], dict):
        for metric_name, values in history["metrics"].items():
            plt.figure(figsize=(6, 4))
            plt.plot(epochs, values, linewidth=2)
            plt.xlabel("Epoch")
            plt.ylabel(metric_name.upper())
            plt.title(f"Validation {metric_name.upper()} Over Time")
            plt.grid(alpha=0.3)
            plt.tight_layout()
            plt.savefig(f"{save_dir}/metric_{metric_name}.png", dpi=300)
            if show: plt.show(); plt.close()


# ============================================================
# 2. Visualize Reconstruction & Error Map
# ============================================================
def visualize_reconstruction(X, Y_true, Y_pred, idx=0, save_dir="results/figures", show=True):
    """
    Visualize input, reconstruction, ground truth, and absolute error map.

    Parameters
    ----------
    X, Y_true, Y_pred : torch.Tensor or np.ndarray
        Shape [N, 1, H, W], values normalized in [0, 1].
    idx : int
        Index of sample to visualize.
    """
    ensure_dir(save_dir)

    def to_np(t):
        return np.squeeze(t.detach().cpu().numpy() if isinstance(t, torch.Tensor) else t)

    x = to_np(X[idx])
    y_true = to_np(Y_true[idx])
    y_pred = to_np(Y_pred[idx])
    diff = np.abs(y_true - y_pred)

    fig, axs = plt.subplots(1, 4, figsize=(12, 3))
    titles = ["Input", "Reconstruction", "Ground Truth", "Abs Error"]
    images = [x, y_pred, y_true, diff]

    for ax, img, title in zip(axs, images, titles):
        im = ax.imshow(img, cmap="gray", vmin=0, vmax=1)
        ax.set_title(title)
        ax.axis("off")
        if title == "Abs Error":
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.suptitle(f"Sample {idx}")
    plt.tight_layout()
    plt.savefig(f"{save_dir}/reconstruction_{idx}.png", dpi=300)
    if show:
        plt.show()
    plt.close()


# ============================================================
# 3. Visualize Multiple Reconstructions
# ============================================================
def visualize_batch_grid(X, Y_true, Y_pred, n=5, save_dir="results/figures", show=True):
    """
    Display n reconstructions side-by-side.

    Inputs: tensors or arrays of shape [N,1,H,W], normalized [0,1].
    """
    ensure_dir(save_dir)
    N = min(n, X.shape[0])
    fig, axes = plt.subplots(N, 3, figsize=(9, 3 * N))

    for i in range(N):
        for j, (img, title) in enumerate(zip(
            [X[i], Y_pred[i], Y_true[i]],
            ["Input", "Reconstruction", "Ground Truth"]
        )):
            ax = axes[i, j] if N > 1 else axes[j]
            img_np = np.squeeze(img.detach().cpu().numpy() if isinstance(img, torch.Tensor) else img)
            ax.imshow(img_np, cmap="gray", vmin=0, vmax=1)
            ax.set_title(title)
            ax.axis("off")

    plt.tight_layout()
    plt.savefig(f"{save_dir}/reconstruction_grid.png", dpi=300)
    if show:
        plt.show()
    plt.close()


# ============================================================
# 4. Visualize Error Distribution
# ============================================================
def plot_error_distribution(Y_true, Y_pred, save_dir="results/figures", show=True):
    """
    Plot histogram of pixel-wise reconstruction error across the test set.
    """
    ensure_dir(save_dir)
    y_true = Y_true.detach().cpu().numpy() if isinstance(Y_true, torch.Tensor) else Y_true
    y_pred = Y_pred.detach().cpu().numpy() if isinstance(Y_pred, torch.Tensor) else Y_pred
    errors = np.abs(y_true - y_pred).flatten()

    plt.figure(figsize=(6, 4))
    sns.histplot(errors, bins=50, kde=True, color="steelblue")
    plt.xlabel("Absolute Error")
    plt.ylabel("Pixel Count")
    plt.title("Distribution of Reconstruction Error")
    plt.tight_layout()
    plt.savefig(f"{save_dir}/error_distribution.png", dpi=300)
    if show: plt.show(); plt.close()


# ============================================================
# 5. Compare Models Across Runs
# ============================================================
def compare_model_metrics(model_results, metric="psnr", save_dir="results/figures", show=True):
    """
    Compare multiple models using a single metric.

    Parameters
    ----------
    model_results : dict
        e.g. {"unet_baseline": 32.1, "unet_large": 34.5, "unet_lpips": 31.7}
    metric : str
        Name of the metric being compared.
    """
    ensure_dir(save_dir)
    models = list(model_results.keys())
    scores = list(model_results.values())

    plt.figure(figsize=(6, 4))
    sns.barplot(x=models, y=scores, palette="crest")
    plt.ylabel(metric.upper())
    plt.title(f"Model Comparison: {metric.upper()}")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/compare_{metric}.png", dpi=300)
    if show: plt.show(); plt.close()
