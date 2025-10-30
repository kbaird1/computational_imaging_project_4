"""
config.py
================
Central configuration module for all model, training, and data hyperparameters.

Provides:
    - Typed dataclasses for model and training configs
    - Baseline configurations for experiments
    - Full search spaces for hyperparameter optimization
    - Helper functions to retrieve or randomly sample configs

Usage:
    from src.config import get_config
    cfg = get_config("unet_baseline")
"""

from dataclasses import dataclass, asdict
from typing import Dict, Any, List
import random
import torch


# ============================================================
# 1. Dataclasses for Clarity
# ============================================================

@dataclass
class ModelConfig:
    """Model architecture hyperparameters."""
    model_name: str = "UNet"
    in_channels: int = 1
    out_channels: int = 1
    init_features: int = 64
    depth: int = 5
    activation: str = "ReLU"
    batch_norm: bool = True
    dropout_rate: float = 0.15
    kernel_size: int = 3
    output_activation: str = "Sigmoid"
    padding_mode: str = "zeros"


@dataclass
class TrainingConfig:
    """Training hyperparameters and runtime options."""
    epochs: int = 100
    batch_size: int = 32
    lr: float = 1e-4
    optimizer: str = "Adam"
    weight_decay: float = 1e-5
    loss_fn: str = "mse:0.9+dssim:0.1"
    # Validation metrics computed and logged each epoch
    error_on_validation: List[str] = (
        "mse", "ssim", "dssim"
    )
    scheduler: str = "None"  # "None", "ReduceLROnPlateau", "CosineAnnealingLR"
    patience: int = 1000
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42
    save_path: str = "results/checkpoints/unet_baseline.pt"
    verbose: bool = True          # detailed per-batch logs
    progress_bar: bool = True     # tqdm progress bar (if verbose=False)
    log_interval: int = 1         # print interval for epoch losses


# ============================================================
# 2. Baseline Configurations
# ============================================================

BASELINE_CONFIGS: Dict[str, Dict[str, Any]] = {
    # ---- Core baseline ----
    "unet_baseline": {
        "model": ModelConfig(),
        "training": TrainingConfig(),
    }
}


# ============================================================
# 3. Search Spaces (for Optuna / Random Sweeps)
# ============================================================

SEARCH_SPACES = {
    "model": {
        "init_features": [16, 32, 64, 128],
        "depth": [3, 4, 5],
        "activation": ["ReLU", "LeakyReLU", "ELU"],
        "dropout_rate": [0.0, 0.1, 0.2, 0.3],
        "kernel_size": [3, 5],
        "output_activation": ["Sigmoid", "Tanh"],
        "padding_mode": ["zeros", "reflect"],
    },
    "training": {
        "lr": [1e-4, 5e-4, 1e-3, 2e-3],
        "batch_size": [8, 16, 32, 64],
        "optimizer": ["Adam", "AdamW", "RMSprop"],
        "weight_decay": [0.0, 1e-6, 1e-5, 1e-4],
        # All differentiable losses supported in metrics.py
        "loss_fn": ["mse", "mae"],
        # Validation metrics (computed at epoch end)
        "error_on_validation": [
            "mse", "mae", "psnr", "ssim", 
            "dssim", "gdl", "tv", "lpips",
        ],
        "scheduler": ["None", "ReduceLROnPlateau", "CosineAnnealingLR"],
        "patience": [5, 10],
        "epochs": [25, 50, 100, 200],
        "seed": [21, 42, 77, 99],
    },
}


# ============================================================
# 4. Helper Functions
# ============================================================

def get_config(name: str = "unet_baseline") -> Dict[str, Any]:
    """
    Return a full configuration dictionary for a named experiment.
    Each configuration includes model and training parameters and a top-level seed.
    """
    if name not in BASELINE_CONFIGS:
        raise ValueError(f"Unknown config name: {name}")

    cfg = BASELINE_CONFIGS[name]
    config_dict = {
        "model": asdict(cfg["model"]),
        "training": asdict(cfg["training"]),
        "seed": cfg["training"].seed,
    }

    # Always re-detect device at runtime
    config_dict["training"]["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    return config_dict


def random_config() -> Dict[str, Any]:
    """
    Randomly sample a configuration from SEARCH_SPACES.
    Useful for exploratory runs or basic random hyperparameter search.
    """
    model_cfg = {k: random.choice(v) for k, v in SEARCH_SPACES["model"].items()}
    train_cfg = {k: random.choice(v) for k, v in SEARCH_SPACES["training"].items()}

    train_cfg["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    seed = train_cfg.get("seed", 42)
    return {"model": model_cfg, "training": train_cfg, "seed": seed}


def list_configs() -> List[str]:
    """Return names of all predefined baseline configurations."""
    return list(BASELINE_CONFIGS.keys())


# ============================================================
# 5. Sanity Test
# ============================================================

if __name__ == "__main__":
    print("Available configs:", list_configs())
    cfg = get_config("unet_baseline")
    print("\nBaseline Config:")
    print(cfg)

    print("\nRandom Config Sample:")
    print(random_config())
