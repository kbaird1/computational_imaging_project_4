"""
metrics.py
================
Evaluation and perceptual metrics for single-pixel or U-Net image reconstruction.

Implements:
    - MSE, MAE, PSNR, SSIM
    - DSSIM (1 - SSIM) / 2
    - Gradient Difference Loss (GDL)
    - Total Variation (TV)
    - Optional LPIPS (requires `pip install lpips`)
    - Optional MS-SSIM (requires `pip install piq`)

All metrics accept torch.Tensor or np.ndarray inputs.
"""

import torch
import numpy as np
from skimage.metrics import structural_similarity as ssim_fn

# Optional dependencies
try:
    import lpips
    _HAS_LPIPS = True
    _lpips_model = lpips.LPIPS(net="alex").eval()
except ImportError:
    _HAS_LPIPS = False

try:
    from piq import multi_scale_ssim as ms_ssim_fn
    _HAS_MS_SSIM = True
except ImportError:
    _HAS_MS_SSIM = False


# ============================================================
# Utility: convert to numpy
# ============================================================
def _to_numpy(x):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    return np.squeeze(x)


# ============================================================
# Core Metrics
# ============================================================
def mean_squared_error(y_true, y_pred):
    y_true, y_pred = _to_numpy(y_true), _to_numpy(y_pred)
    return float(np.mean((y_true - y_pred) ** 2))


def mean_absolute_error(y_true, y_pred):
    y_true, y_pred = _to_numpy(y_true), _to_numpy(y_pred)
    return float(np.mean(np.abs(y_true - y_pred)))


def peak_signal_to_noise_ratio(y_true, y_pred, data_range=1.0):
    mse = mean_squared_error(y_true, y_pred)
    if mse <= 1e-10:
        return float("inf")
    return 20 * np.log10(data_range) - 10 * np.log10(mse)


def structural_similarity_index(y_true, y_pred, data_range=1.0):
    y_true, y_pred = _to_numpy(y_true), _to_numpy(y_pred)
    if y_true.ndim == 4:
        vals = [ssim_fn(t[0], p[0], data_range=data_range) for t, p in zip(y_true, y_pred)]
        return float(np.mean(vals))
    return float(ssim_fn(y_true, y_pred, data_range=data_range))


# ============================================================
# Advanced Metrics
# ============================================================
def dssim(y_true, y_pred, data_range=1.0):
    """Dissimilarity form of SSIM, i.e. (1 - SSIM) / 2."""
    return 0.5 * (1 - structural_similarity_index(y_true, y_pred, data_range=data_range))


def gradient_difference_loss(y_true, y_pred):
    """Gradient Difference Loss (edge preservation)."""
    if not isinstance(y_true, torch.Tensor):
        y_true = torch.tensor(y_true)
    if not isinstance(y_pred, torch.Tensor):
        y_pred = torch.tensor(y_pred)

    def gradient(x):
        dx = torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1])
        dy = torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :])
        return dx, dy

    dx_true, dy_true = gradient(y_true)
    dx_pred, dy_pred = gradient(y_pred)

    loss = torch.mean(torch.abs(dx_true - dx_pred)) + torch.mean(torch.abs(dy_true - dy_pred))
    return float(loss.item())


def total_variation(y_pred):
    """Total variation regularization."""
    if not isinstance(y_pred, torch.Tensor):
        y_pred = torch.tensor(y_pred)

    dx = torch.abs(y_pred[:, :, :, 1:] - y_pred[:, :, :, :-1])
    dy = torch.abs(y_pred[:, :, 1:, :] - y_pred[:, :, :-1, :])
    tv = torch.mean(dx) + torch.mean(dy)
    return float(tv.item())


def lpips_score(y_true, y_pred):
    """Learned Perceptual Image Patch Similarity (lower = better)."""
    if not _HAS_LPIPS:
        raise ImportError("LPIPS not installed. Run `pip install lpips`.")
    if not isinstance(y_true, torch.Tensor):
        y_true = torch.tensor(y_true)
    if not isinstance(y_pred, torch.Tensor):
        y_pred = torch.tensor(y_pred)
    if y_true.ndim == 2:
        y_true = y_true.unsqueeze(0).unsqueeze(0)
        y_pred = y_pred.unsqueeze(0).unsqueeze(0)
    if y_true.shape[1] == 1:  # grayscale to 3ch
        y_true = y_true.repeat(1, 3, 1, 1)
        y_pred = y_pred.repeat(1, 3, 1, 1)
    return float(_lpips_model(y_true, y_pred).mean().item())


def multi_scale_ssim(y_true, y_pred):
    """Multi-scale SSIM (requires piq)."""
    if not _HAS_MS_SSIM:
        raise ImportError("piq not installed. Run `pip install piq`.")
    if not isinstance(y_true, torch.Tensor):
        y_true = torch.tensor(y_true)
    if not isinstance(y_pred, torch.Tensor):
        y_pred = torch.tensor(y_pred)
    return float(ms_ssim_fn(y_pred, y_true).item())


# ============================================================
# Unified Entry Point
# ============================================================
def compute_metrics(y_pred, y_true, metrics_list=None):
    """
    Compute selected metrics and return dict.

    Example
    -------
    compute_metrics(y_pred, y_true, metrics_list=["mse", "ssim", "lpips"])
    """
    available = {
        "mse": mean_squared_error,
        "mae": mean_absolute_error,
        "psnr": peak_signal_to_noise_ratio,
        "ssim": structural_similarity_index,
        "dssim": dssim,
        "gdl": gradient_difference_loss,
        "tv": total_variation,
        "lpips": lpips_score if _HAS_LPIPS else None,
        "ms_ssim": multi_scale_ssim if _HAS_MS_SSIM else None,
    }

    if metrics_list is None:
        metrics_list = list(available.keys())

    results = {}
    for m in metrics_list:
        fn = available.get(m)
        if fn is None:
            results[m] = None
        else:
            try:
                results[m] = fn(y_true, y_pred)
            except Exception as e:
                results[m] = f"error: {e}"
    return results


# ============================================================
# Sanity Test
# ============================================================
if __name__ == "__main__":
    print("Testing metrics module...")
    y_true = torch.rand((1, 1, 64, 64))
    y_pred = y_true * 0.9 + 0.05 * torch.randn_like(y_true)

    results = compute_metrics(y_pred, y_true, metrics_list=["mse", "ssim", "dssim", "gdl", "tv"])
    for k, v in results.items():
        print(f"{k.upper()}: {v}")
    print("Metrics test complete")
