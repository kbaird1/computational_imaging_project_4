"""
metrics.py
================
Evaluation metrics for single-pixel image reconstruction.

Implements:
    - Mean Squared Error (MSE)
    - Mean Absolute Error (MAE)
    - Peak Signal-to-Noise Ratio (PSNR)
    - Structural Similarity Index (SSIM)

All functions can take either torch.Tensor or np.ndarray inputs.
"""

import torch
import numpy as np
from skimage.metrics import structural_similarity as ssim_fn


# ============================================================
# Utility: convert to numpy array for uniform processing
# ============================================================
def _to_numpy(x):
    """Convert torch.Tensor or array-like to np.ndarray for consistent metric computation."""
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    return np.squeeze(x)  # remove batch/channel dims if present


# ============================================================
# 1. Mean Squared Error (MSE)
# ============================================================
def mean_squared_error(y_true, y_pred):
    """Compute mean squared error between two images or batches."""
    y_true, y_pred = _to_numpy(y_true), _to_numpy(y_pred)
    return float(np.mean((y_true - y_pred) ** 2))


# ============================================================
# 2. Mean Absolute Error (MAE)
# ============================================================
def mean_absolute_error(y_true, y_pred):
    """Compute mean absolute error between two images or batches."""
    y_true, y_pred = _to_numpy(y_true), _to_numpy(y_pred)
    return float(np.mean(np.abs(y_true - y_pred)))


# ============================================================
# 3. Peak Signal-to-Noise Ratio (PSNR)
# ============================================================
def peak_signal_to_noise_ratio(y_true, y_pred, data_range=1.0):
    """Compute PSNR (in decibels) between two images."""
    mse = mean_squared_error(y_true, y_pred)
    if mse <= 1e-10:
        return float("inf")
    return 20 * np.log10(data_range) - 10 * np.log10(mse)


# ============================================================
# 4. Structural Similarity Index (SSIM)
# ============================================================
def structural_similarity_index(y_true, y_pred, data_range=1.0):
    """
    Compute mean SSIM between two images.
    Works for single or batched images (N×1×H×W).
    """
    y_true, y_pred = _to_numpy(y_true), _to_numpy(y_pred)

    if y_true.ndim == 4:  # batched input
        ssim_vals = []
        for t, p in zip(y_true, y_pred):
            try:
                ssim_val = ssim_fn(t[0], p[0], data_range=data_range)
            except ValueError:
                # Handle cases where images are not the same shape
                ssim_val = 0.0
            ssim_vals.append(ssim_val)
        return float(np.mean(ssim_vals))
    else:
        return float(ssim_fn(y_true, y_pred, data_range=data_range))


# ============================================================
# 5. Unified Entry Point
# ============================================================
def compute_metrics(y_pred, y_true):
    """
    Compute and return a dictionary of all implemented metrics.
    Matches expectations of train.py and main.py.

    Returns
    -------
    dict:
        {
            "mse": ...,
            "mae": ...,
            "psnr": ...,
            "ssim": ...
        }
    """
    return {
        "mse": mean_squared_error(y_true, y_pred),
        "mae": mean_absolute_error(y_true, y_pred),
        "psnr": peak_signal_to_noise_ratio(y_true, y_pred),
        "ssim": structural_similarity_index(y_true, y_pred),
    }


# ============================================================
# 6. Sanity Test (run manually)
# ============================================================
if __name__ == "__main__":
    print("Testing metrics module...")
    y_true = torch.rand((1, 1, 64, 64))
    y_pred = y_true * 0.9  # slightly scaled prediction

    results = compute_metrics(y_pred, y_true)
    for k, v in results.items():
        print(f"{k.upper()}: {v:.4f}")
    print("Metrics test passed ")
