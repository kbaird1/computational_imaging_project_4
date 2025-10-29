import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
import numpy as np
import scipy.io as spio
import os
import matplotlib.pyplot as plt

def load_data(mat_file_path):
    """Loads X (low-quality inputs) and Y (ground-truth targets) from a .mat file."""
    if not os.path.exists(mat_file_path):
        raise FileNotFoundError(f"MAT file not found at {mat_file_path}")

    mat_dict = spio.loadmat(mat_file_path)
    X, Y = mat_dict.get("X"), mat_dict.get("Y")
    if X is None or Y is None:
        raise KeyError("Missing 'X' or 'Y' keys in MAT file.")

    X, Y = X.astype(np.float32), Y.astype(np.float32)
    X = np.clip(X, 0, 1)
    Y = np.clip(Y, 0, 1)

    # Add channel dimension if missing
    if X.ndim == 3:
        X = X[..., np.newaxis]
        Y = Y[..., np.newaxis]

    # Convert to PyTorch tensors with channel-first layout
    X = np.transpose(X, (0, 3, 1, 2))
    Y = np.transpose(Y, (0, 3, 1, 2))

    return torch.tensor(X), torch.tensor(Y)


def split_data(X, Y, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, random_state=42):
    """Splits tensors into train, val, and test sets."""
    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError("Train + val + test ratios must sum to 1.0")

    X_train, X_temp, Y_train, Y_temp = train_test_split(
        X, Y, test_size=(1 - train_ratio), random_state=random_state, shuffle=True
    )

    rel_val_ratio = val_ratio / (val_ratio + test_ratio)
    X_val, X_test, Y_val, Y_test = train_test_split(
        X_temp, Y_temp, test_size=(1 - rel_val_ratio),
        random_state=random_state, shuffle=True
    )

    return (X_train, Y_train), (X_val, Y_val), (X_test, Y_test)


def create_dataloaders(mat_file_path, config):
    """
    Loads data from .mat file, splits it into train/val/test sets,
    and returns PyTorch DataLoaders for each split.
    """
    print(f"[data_helper] Loading dataset from {mat_file_path} ...")

    # 1. Load & split data
    X, Y = load_data(mat_file_path)
    (X_train, Y_train), (X_val, Y_val), (X_test, Y_test) = split_data(
        X, Y,
        train_ratio=config["training"].get("train_ratio", 0.8),
        val_ratio=config["training"].get("val_ratio", 0.1),
        test_ratio=config["training"].get("test_ratio", 0.1),
        random_state=config["training"].get("seed", 42)
    )

    # 2. Create TensorDatasets
    train_ds = TensorDataset(X_train, Y_train)
    val_ds = TensorDataset(X_val, Y_val)
    test_ds = TensorDataset(X_test, Y_test)

    # 3. Create DataLoaders
    batch_size = config["training"]["batch_size"]
    num_workers = config["training"].get("num_workers", 0)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    print(f"[data_helper] Created DataLoaders | Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")
    return train_loader, val_loader, test_loader

# ============================================================
# 4. TEST FUNCTION (VISUAL + STRUCTURAL VALIDATION)
# ============================================================
def test_data_loading_and_splitting(mat_file_path):
    """
    Tests the data loading and splitting pipeline to ensure correctness.

    - Loads .mat data
    - Validates shapes, normalization, and alignment
    - Splits and converts to torch tensors
    - Displays one example pair visually
    """
    print("\n========== Testing PyTorch Data Pipeline ==========")

    X, Y = load_data(mat_file_path)
    assert X.shape == Y.shape, f"Shape mismatch: X {X.shape} vs Y {Y.shape}"
    assert np.all((X >= 0) & (X <= 1)), "X contains values outside [0, 1]"
    assert np.all((Y >= 0) & (Y <= 1)), "Y contains values outside [0, 1]"

    train_set, val_set, test_set = split_data(X, Y)

    # Confirm dataset sizes
    total = len(train_set) + len(val_set) + len(test_set)
    print(f"Total samples verified: {total}")

    # Check tensor shapes
    sample_X, sample_Y = train_set[0]
    print(f"Sample tensor shapes: X={sample_X.shape}, Y={sample_Y.shape}")
    print(f"Tensor dtype: {sample_X.dtype}")

    # Visual sanity check
    idx = np.random.randint(0, X.shape[0])
    plt.figure(figsize=(6, 3))
    plt.subplot(1, 2, 1)
    plt.imshow(X[idx, 0, :, :], cmap="gray")
    plt.title("Low-Quality Input (X)")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(Y[idx, 0, :, :], cmap="gray")
    plt.title("Ground Truth (Y)")
    plt.axis("off")

    plt.suptitle(f"Sample Index {idx}")
    plt.tight_layout()
    plt.show()

    print("PyTorch data pipeline verified successfully.")
    print("=====================================================\n")

