"""
metrics.py
================
Evaluation and perceptual metrics for single-pixel or U-Net image reconstruction.

Implements:
    - MSE, MAE, PSNR, SSIM
    - DSSIM (1 - SSIM) / 2
    - Gradient Difference Loss (GDL)
    - Total Variation (TV)
    - LPIPS (requires `pip install lpips`)
    - MS-SSIM (requires `pip install piq`)

All metrics accept torch.Tensor or np.ndarray inputs.
"""

import torch
import numpy as np
from skimage.metrics import structural_similarity as ssim_fn
import lpips
from piq import multi_scale_ssim as ms_ssim_fn
from piq import ssim as single_ssim_fn


# Preload LPIPS model (for speed)
_lpips_model = lpips.LPIPS(net="alex").eval()

# ============================================================
# Utility: convert to numpy or tensor
# ============================================================
def _to_numpy(x):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    return np.squeeze(x)


def _to_tensor(x, device=None):
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x, dtype=torch.float32)
    if device is not None:
        x = x.to(device)
    return x


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
    return float(20 * np.log10(data_range) - 10 * np.log10(mse))


def structural_similarity_index(y_true, y_pred, data_range=1.0):
    """Compute SSIM robustly for any image size and batch shape."""
    y_true, y_pred = _to_numpy(y_true), _to_numpy(y_pred)

    def _ssim_safe(a, b):
        # Remove leading batch or channel dimensions > 2D
        while a.ndim > 2:
            a = a[0]
            b = b[0]
        h, w = a.shape[-2:]
        win_size = min(7, h, w)
        if win_size % 2 == 0:
            win_size -= 1
        win_size = max(3, win_size)
        return ssim_fn(a, b, data_range=data_range, win_size=win_size, channel_axis=None)

    if y_true.ndim == 4:
        vals = [_ssim_safe(t, p) for t, p in zip(y_true, y_pred)]
        return float(np.mean(vals))
    else:
        return float(_ssim_safe(y_true, y_pred))




# ============================================================
# Advanced Metrics
# ============================================================
def dssim(y_true, y_pred, data_range=1.0):
    """Dissimilarity form of SSIM."""
    return 0.5 * (1 - structural_similarity_index(y_true, y_pred, data_range=data_range))


def gradient_difference_loss(y_true, y_pred):
    """Gradient Difference Loss (edge preservation)."""
    y_true, y_pred = _to_tensor(y_true), _to_tensor(y_pred)

    def gradient(x):
        dx = torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1])
        dy = torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :])
        return dx, dy

    dx_true, dy_true = gradient(y_true)
    dx_pred, dy_pred = gradient(y_pred)
    loss = torch.mean(torch.abs(dx_true - dx_pred)) + torch.mean(torch.abs(dy_true - dy_pred))
    return float(loss.item())


def total_variation(y_pred, y_true=None):
    """Total variation regularization (independent of y_true)."""
    y_pred = _to_tensor(y_pred)
    dx = torch.abs(y_pred[:, :, :, 1:] - y_pred[:, :, :, :-1])
    dy = torch.abs(y_pred[:, :, 1:, :] - y_pred[:, :, :-1, :])
    tv = torch.mean(dx) + torch.mean(dy)
    return float(tv.item())


def lpips_score(y_true, y_pred, device=None):
    """Learned Perceptual Image Patch Similarity (lower = better)."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    y_true = _to_tensor(y_true, device)
    y_pred = _to_tensor(y_pred, device)

    if y_true.ndim == 2:
        y_true = y_true.unsqueeze(0).unsqueeze(0)
        y_pred = y_pred.unsqueeze(0).unsqueeze(0)
    if y_true.shape[1] == 1:  # grayscale to 3-channel
        y_true = y_true.repeat(1, 3, 1, 1)
        y_pred = y_pred.repeat(1, 3, 1, 1)

    _lpips_model.to(device)
    return float(_lpips_model(y_true, y_pred).mean().item())


def multi_scale_ssim(y_true, y_pred):
    """
    Multi-scale SSIM using PIQ with safety fixes.
    - Falls back to single-scale SSIM for small (<161x161) images.
    - Clamps negative values to [0, 1].
    - Always returns a numeric value.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if not isinstance(y_true, torch.Tensor):
        y_true = torch.tensor(y_true, device=device)
    if not isinstance(y_pred, torch.Tensor):
        y_pred = torch.tensor(y_pred, device=device)

    y_true = torch.clamp(y_true.float(), 0, 1)
    y_pred = torch.clamp(y_pred.float(), 0, 1)

    h, w = y_true.shape[-2:]
    too_small = min(h, w) < 161

    try:
        if too_small:
            val = single_ssim_fn(y_pred, y_true, data_range=1.0)
        else:
            val = ms_ssim_fn(y_pred, y_true, data_range=1.0)
        # Clamp negatives, handle NaNs
        val = torch.nan_to_num(val, nan=0.0)
        val = torch.clamp(val, 0.0, 1.0)
        return float(val.item())
    except Exception as e:
        print(f"[metrics] MS-SSIM fallback due to error: {e}")
        val = single_ssim_fn(y_pred, y_true, data_range=1.0)
        val = torch.clamp(val, 0.0, 1.0)
        return float(val.item())



# ============================================================
# Unified Entry Point
# ============================================================
def compute_metrics(y_pred, y_true, metrics_list=None):
    """
    Compute selected metrics and return dict.
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
        "ms_ssim": multi_scale_ssim,
    }

    if metrics_list is None:
        metrics_list = list(available.keys())

    results = {}
    for m in metrics_list:
        fn = available.get(m)
        try:
            results[m] = fn(y_true, y_pred)
        except Exception as e:
            results[m] = f"error: {e}"
    return results


# ============================================================
# Extended Metrics Test Suite
# ============================================================
def test_metrics():
    """Run full validation suite on all metrics."""
    print("\n========== [METRICS MODULE TEST] ==========")

    torch.manual_seed(42)
    y_true = torch.rand((2, 1, 64, 64))
    y_pred_close = y_true * 0.9 + 0.05 * torch.randn_like(y_true)
    y_pred_far = torch.zeros_like(y_true)

    test_cases = {
        "identical": (y_true, y_true),
        "noisy": (y_true, y_pred_close),
        "completely different": (y_true, y_pred_far),
    }

    all_metrics = ["mse", "mae", "psnr", "ssim", "dssim", "gdl", "tv", "lpips", "ms_ssim"]

    for case_name, (yt, yp) in test_cases.items():
        print(f"\n--- Testing case: {case_name.upper()} ---")
        results = compute_metrics(yp, yt, metrics_list=all_metrics)
        for k, v in results.items():
            if isinstance(v, (float, int)):
                print(f"  {k.upper():<8} = {v:.6f}")
            else:
                print(f"  {k.upper():<8} -> {v}")

    print("\n--- Sanity Checks ---")
    mse_identical = mean_squared_error(y_true, y_true)
    mse_different = mean_squared_error(y_true, torch.zeros_like(y_true))
    assert mse_identical < 1e-8, f"MSE identical images should be near 0, got {mse_identical}"
    assert mse_different > 0.05, f"MSE different images too low, got {mse_different}"
    print("✅ MSE sanity check passed")

    psnr_high = peak_signal_to_noise_ratio(y_true, y_true)
    psnr_low = peak_signal_to_noise_ratio(y_true, torch.zeros_like(y_true))
    assert psnr_high > psnr_low, "PSNR should be higher for identical images"
    print("✅ PSNR sanity check passed")

    print("\n========== [METRICS TEST COMPLETE] ==========\n")


if __name__ == "__main__":
    test_metrics()
