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
    depth: int = 4
    activation: str = "LeakyReLU"
    batch_norm: bool = True
    dropout_rate: float = 0.15
    kernel_size: int = 3
    output_activation: str = "Sigmoid"
    padding_mode: str = "zeros"


@dataclass
class TrainingConfig:
    """Training hyperparameters and runtime options."""
    epochs: int = 20
    batch_size: int = 32
    lr: float = 1e-3
    optimizer: str = "Adam"
    weight_decay: float = 1e-5
    loss_fn: str = "mse"
    # Validation metrics computed and logged each epoch
    error_on_validation: List[str] = (
        "mse", "mae", "psnr", "ssim", 
        "dssim", "gdl", "tv", "lpips"
    )
    scheduler: str = "ReduceLROnPlateau"  # "None", "ReduceLROnPlateau", "CosineAnnealingLR"
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
    },

    # ---- Larger capacity model (more depth/features, slower but higher fidelity) ----
    "unet_large": {
        "model": ModelConfig(init_features=128, depth=5, dropout_rate=0.1),
        "training": TrainingConfig(
            lr=5e-4, batch_size=16, epochs=100, seed=42, loss_fn="mse"
        ),
    },

    # ---- Smaller and faster model (lightweight for debugging / small datasets) ----
    "unet_small": {
        "model": ModelConfig(init_features=32, depth=3),
        "training": TrainingConfig(
            lr=2e-3, batch_size=64, epochs=30, seed=42, loss_fn="mae"
        ),
    },

    # ---- Perceptual configuration (for high-quality reconstruction) ----
    "unet_perceptual": {
        "model": ModelConfig(init_features=64, depth=4, dropout_rate=0.05),
        "training": TrainingConfig(
            lr=1e-4,
            batch_size=8,
            epochs=100,
            loss_fn="lpips",
            scheduler="ReduceLROnPlateau",
            seed=77,
        ),
    },

    # ---- Structure-aware configuration (for edge/detail preservation) ----
    "unet_structural": {
        "model": ModelConfig(init_features=64, depth=4, activation="LeakyReLU"),
        "training": TrainingConfig(
            lr=1e-3,
            batch_size=16,
            epochs=100,
            loss_fn="dssim",
            scheduler="CosineAnnealingLR",
            seed=99,
        ),
    },

    # ---- Smooth regularized model (for denoising with total variation loss) ----
    "unet_smooth": {
        "model": ModelConfig(init_features=64, depth=3, dropout_rate=0.1),
        "training": TrainingConfig(
            lr=1e-3,
            batch_size=32,
            epochs=80,
            loss_fn="tv",
            scheduler="ReduceLROnPlateau",
            seed=21,
        ),
    },
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
