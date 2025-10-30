"""
metrics.py
================
Evaluation and perceptual metrics for single-pixel or U-Net image reconstruction.

Now fully differentiable — all functions can be used both as metrics and as loss functions.

Implements:
    - MSE, MAE, PSNR, SSIM
    - DSSIM (1 - SSIM) / 2
    - Gradient Difference Loss (GDL)
    - Total Variation (TV)
    - LPIPS (requires `pip install lpips`)
    - MS-SSIM (requires `pip install piq`)
"""

import torch
import numpy as np
import lpips
from piq import ssim as differentiable_ssim

# Preload LPIPS model (for speed)
_lpips_model = lpips.LPIPS(net="alex").eval()


# ============================================================
# Utility helpers
# ============================================================
def _to_tensor(x, device=None):
    """Ensure input is a float32 torch.Tensor."""
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x, dtype=torch.float32)
    if device is not None:
        x = x.to(device)
    return x


def _to_numpy(x):
    """Convert tensor to numpy for evaluation-only contexts."""
    if isinstance(x, torch.Tensor):
        if x.requires_grad:
            return x  # Keep as tensor if still part of autograd
        return x.detach().cpu().numpy()
    return np.array(x)


# ============================================================
# Core Metrics (Differentiable)
# ============================================================
def mean_squared_error(y_true, y_pred):
    """Differentiable Mean Squared Error."""
    y_true = _to_tensor(y_true)
    y_pred = _to_tensor(y_pred)
    return torch.mean((y_true - y_pred) ** 2)


def mean_absolute_error(y_true, y_pred):
    """Differentiable Mean Absolute Error."""
    y_true = _to_tensor(y_true)
    y_pred = _to_tensor(y_pred)
    return torch.mean(torch.abs(y_true - y_pred))


def peak_signal_to_noise_ratio(y_true, y_pred, data_range=1.0):
    """
    PSNR — non-differentiable but useful for evaluation.
    Kept as a numeric metric (not intended as a training loss).
    """
    y_true_np = _to_numpy(y_true)
    y_pred_np = _to_numpy(y_pred)
    if isinstance(y_true_np, torch.Tensor):
        # fallback in case it's still tensor
        mse = mean_squared_error(y_true_np, y_pred_np)
        mse_val = float(mse.detach().cpu().numpy())
    else:
        mse_val = np.mean((y_true_np - y_pred_np) ** 2)
    if mse_val <= 1e-10:
        return float("inf")
    return float(20 * np.log10(data_range) - 10 * np.log10(mse_val))

def structural_similarity_index(y_true, y_pred, data_range=1.0):
    """
    Differentiable SSIM using PIQ (AMP-safe).
    Forces float32 precision to avoid half<->float mismatches under autocast.
    """
    if not isinstance(y_true, torch.Tensor):
        y_true = torch.tensor(y_true)
    if not isinstance(y_pred, torch.Tensor):
        y_pred = torch.tensor(y_pred)

    device = y_pred.device
    # always cast to float32 so mixed-precision AMP won't break
    y_true = y_true.to(device=device, dtype=torch.float32)
    y_pred = y_pred.to(device=device, dtype=torch.float32)

    val = differentiable_ssim(y_pred, y_true, data_range=data_range)
    return val.mean()  # keep as differentiable scalar


# ============================================================
# Advanced Metrics (Differentiable)
# ============================================================
def dssim(y_true, y_pred, data_range=1.0):
    """
    Differentiable DSSIM = (1 - SSIM) / 2
    """
    ssim_val = structural_similarity_index(y_true, y_pred, data_range=data_range)
    return (1.0 - ssim_val) / 2.0

def gradient_difference_loss(y_true, y_pred):
    """Gradient Difference Loss (edge preservation)."""
    y_true, y_pred = _to_tensor(y_true), _to_tensor(y_pred)

    def gradient(x):
        dx = torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1])
        dy = torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :])
        return dx, dy

    dx_true, dy_true = gradient(y_true)
    dx_pred, dy_pred = gradient(y_pred)
    return torch.mean(torch.abs(dx_true - dx_pred)) + torch.mean(torch.abs(dy_true - dy_pred))


def total_variation(y_pred, y_true=None):
    """Total variation regularization (independent of y_true)."""
    y_pred = _to_tensor(y_pred)
    dx = torch.abs(y_pred[:, :, :, 1:] - y_pred[:, :, :, :-1])
    dy = torch.abs(y_pred[:, :, 1:, :] - y_pred[:, :, :-1, :])
    return torch.mean(dx) + torch.mean(dy)


def lpips_score(y_true, y_pred, device=None):
    """Learned Perceptual Image Patch Similarity (fully differentiable)."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    y_true = _to_tensor(y_true, device)
    y_pred = _to_tensor(y_pred, device)

    if y_true.ndim == 2:
        y_true = y_true.unsqueeze(0).unsqueeze(0)
        y_pred = y_pred.unsqueeze(0).unsqueeze(0)
    if y_true.shape[1] == 1:  # grayscale → 3-channel
        y_true = y_true.repeat(1, 3, 1, 1)
        y_pred = y_pred.repeat(1, 3, 1, 1)

    _lpips_model.to(device)
    return _lpips_model(y_true, y_pred).mean()


# ============================================================
# Unified Metric Interface
# ============================================================
def compute_metrics(y_pred, y_true, metrics_list=None, as_float=False):
    """
    Compute selected metrics and return dict.
    If as_float=True, detaches and converts to float.
    """
    available = {
        "mse": mean_squared_error,
        "mae": mean_absolute_error,
        "psnr": peak_signal_to_noise_ratio,
        "ssim": structural_similarity_index,
        "dssim": dssim,
        "gdl": gradient_difference_loss,
        "tv": total_variation,
        "lpips": lpips_score,
    }

    if metrics_list is None:
        metrics_list = list(available.keys())

    results = {}
    for m in metrics_list:
        fn = available.get(m)
        try:
            val = fn(y_true, y_pred)
            if as_float and isinstance(val, torch.Tensor):
                val = float(val.detach().cpu().numpy())
            results[m] = val
        except Exception as e:
            results[m] = f"error: {e}"
    return results


# ============================================================
# Sanity Test
# ============================================================
if __name__ == "__main__":
    print("\n========== [METRICS MODULE TEST] ==========")
    torch.manual_seed(42)
    y_true = torch.rand((2, 1, 64, 64))
    y_pred_close = y_true * 0.9 + 0.05 * torch.randn_like(y_true)
    y_pred_far = torch.zeros_like(y_true)

    test_cases = {
        "IDENTICAL": (y_true, y_true),
        "NOISY": (y_true, y_pred_close),
        "DIFFERENT": (y_true, y_pred_far),
    }

    for name, (yt, yp) in test_cases.items():
        print(f"\n--- Testing case: {name} ---")
        results = compute_metrics(yp, yt, as_float=True)
        for k, v in results.items():
            print(f"  {k.upper():<8} = {v}")

    # Differentiability tests
    print("\n--- Differentiability Tests (Backward Pass) ---")
    for name, fn in [
        ("MSE", mean_squared_error),
        ("MAE", mean_absolute_error),
        ("SSIM", structural_similarity_index),
        ("DSSIM", dssim),
        ("GDL", gradient_difference_loss),
        ("TV", total_variation),
    ]:
        y_pred = y_true.clone().requires_grad_(True)
        try:
            loss = fn(y_true, y_pred)
            loss.backward()
            grad_norm = y_pred.grad.norm().item()
            print(f"Testing differentiability for {name}... ✅ success (grad norm={grad_norm:.4f})")
        except Exception as e:
            print(f"Testing differentiability for {name}... ❌ failed ({e})")

    print("\n✅ Unified differentiable metrics ready.")
