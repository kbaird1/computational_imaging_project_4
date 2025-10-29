"""
visualize.py
================
Visualization utilities for single-pixel image reconstruction.

Includes:
    - Plotting training/validation losses
    - Plotting PSNR and SSIM metrics over epochs
    - Displaying side-by-side image reconstructions
    - Saving publication-quality figures
"""

import os
import torch
import matplotlib.pyplot as plt
import numpy as np


# ============================================================
# Utility: Ensure figure directory exists
# ============================================================
def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


# ============================================================
# 1. Plot Training History
# ============================================================
def plot_training_history(history, save_dir="results/figures", show=True):
    """
    Plot training and validation loss curves and metric trends.

    Parameters
    ----------
    history : dict
        Dictionary returned from train_model() containing keys:
        ['train_loss', 'val_loss', 'psnr', 'ssim']
    save_dir : str
        Directory to save the plots.
    show : bool
        Whether to display the plots interactively.
    """
    ensure_dir(save_dir)
    epochs = np.arange(1, len(history["train_loss"]) + 1)

    # ---- Plot Loss ----
    plt.figure(figsize=(6, 4))
    plt.plot(epochs, history["train_loss"], label="Train Loss", linewidth=2)
    plt.plot(epochs, history["val_loss"], label="Val Loss", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Loss (MSE)")
    plt.title("Training & Validation Loss")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/loss_curve.png", dpi=300)
    if show: plt.show(); plt.close()

    # ---- Plot Metrics ----
    plt.figure(figsize=(6, 4))
    plt.plot(epochs, history["psnr"], label="PSNR (dB)", linewidth=2)
    plt.plot(epochs, history["ssim"], label="SSIM", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Metric Value")
    plt.title("Validation Metrics Over Time")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/metrics_curve.png", dpi=300)
    if show: plt.show(); plt.close()


# ============================================================
# 2. Visualize Image Reconstruction
# ============================================================
def visualize_reconstruction(X, Y_true, Y_pred, idx=0, save_dir="results/figures", show=True):
    """
    Display side-by-side comparison of input, reconstruction, and ground truth.

    Parameters
    ----------
    X : torch.Tensor or np.ndarray
        Input image batch [N,1,H,W]
    Y_true : torch.Tensor or np.ndarray
        Ground truth batch [N,1,H,W]
    Y_pred : torch.Tensor or np.ndarray
        Model output batch [N,1,H,W]
    idx : int
        Index of image to visualize
    save_dir : str
        Directory to save figure
    show : bool
        Whether to display the figure interactively
    """
    ensure_dir(save_dir)

    # Convert to numpy
    def to_np(t):
        if isinstance(t, torch.Tensor):
            t = t.detach().cpu().numpy()
        return np.squeeze(t)

    x = to_np(X[idx])
    y_true = to_np(Y_true[idx])
    y_pred = to_np(Y_pred[idx])

    plt.figure(figsize=(9, 3))
    titles = ["Low-Quality Input", "Reconstruction (Pred)", "Ground Truth"]
    images = [x, y_pred, y_true]

    for i in range(3):
        plt.subplot(1, 3, i + 1)
        plt.imshow(images[i], cmap="gray", vmin=0, vmax=1)
        plt.title(titles[i], fontsize=11)
        plt.axis("off")

    plt.suptitle(f"Sample {idx}", fontsize=12)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/reconstruction_{idx}.png", dpi=300)
    if show:
        plt.show()
    plt.close()


# ============================================================
# 3. Visualize Batch Grid
# ============================================================
def visualize_batch_grid(X, Y_true, Y_pred, n=5, save_dir="results/figures", show=True):
    """
    Display multiple reconstructions in a grid.

    Parameters
    ----------
    X, Y_true, Y_pred : tensors or arrays of shape [N,1,H,W]
    n : int
        Number of samples to show
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
            img_np = img.detach().cpu().numpy().squeeze() if isinstance(img, torch.Tensor) else np.squeeze(img)
            ax.imshow(img_np, cmap="gray", vmin=0, vmax=1)
            ax.set_title(title)
            ax.axis("off")

    plt.tight_layout()
    plt.savefig(f"{save_dir}/reconstruction_grid.png", dpi=300)
    if show:
        plt.show()
    plt.close()


# # ============================================================
# # Sanity Test
# # ============================================================
# if __name__ == "__main__":
#     print("Testing visualization module...")
#     # Simulate data
#     X = torch.rand((5, 1, 64, 64))
#     Y_true = torch.rand((5, 1, 64, 64))
#     Y_pred = torch.rand((5, 1, 64, 64))

#     history = {
#         "train_loss": np.random.rand(10).tolist(),
#         "val_loss": np.random.rand(10).tolist(),
#         "psnr": np.random.rand(10).tolist(),
#         "ssim": np.random.rand(10).tolist(),
#     }

#     plot_training_history(history, show=False)
#     visualize_reconstruction(X, Y_true, Y_pred, idx=0, show=False)
#     visualize_batch_grid(X, Y_true, Y_pred, n=3, show=False)
#     print("Visualization tests passed")
