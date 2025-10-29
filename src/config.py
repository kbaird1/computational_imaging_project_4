"""
config.py
================
Central configuration module for all model, training, and data hyperparameters.

Supports:
    - Defining default baseline configs
    - Defining full search spaces for hyperparameter optimization
    - Selecting subsets for specific experiments

Use:
    from src.config import get_config
    cfg = get_config("unet_baseline")
"""

from dataclasses import dataclass, asdict
from typing import Dict, Any, List
import random
import torch


# ============================================================
# 1. Dataclasses for clarity (typed configs)
# ============================================================

@dataclass
class ModelConfig:
    model_name: str = "UNet"
    in_channels: int = 1
    out_channels: int = 1
    init_features: int = 64
    depth: int = 4
    activation: str = "ReLU"
    batch_norm: bool = True
    dropout_rate: float = 0.0
    kernel_size: int = 3
    output_activation: str = "Sigmoid"
    padding_mode: str = "zeros"
    
@dataclass
class TrainingConfig:
    epochs: int = 2
    batch_size: int = 32
    lr: float = 1e-3
    optimizer: str = "Adam"
    weight_decay: float = 1e-5
    loss_fn: str = "MSE"
    scheduler: str = "None"  # e.g., "ReduceLROnPlateau"
    patience: int = 5
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42
    save_path: str = "results/checkpoints/unet_baseline.pt"
    verbose: bool = True         # show detailed per-batch training logs
    progress_bar: bool = True    # show tqdm progress bar (only when verbose=False)
    log_interval: int = 1       # how often to print epoch loss if verbose=True


# ============================================================
# 2. Baseline configurations
# ============================================================

BASELINE_CONFIGS: Dict[str, Dict[str, Any]] = {
    "unet_baseline": {
        "model": ModelConfig(),
        "training": TrainingConfig(),
    },
    "unet_large": {
        "model": ModelConfig(init_features=128, dropout_rate=0.1),
        "training": TrainingConfig(lr=5e-4, batch_size=16, epochs=100, seed=42),
    },
    "unet_small": {
        "model": ModelConfig(init_features=32, depth=3),
        "training": TrainingConfig(lr=2e-3, batch_size=64, epochs=30, seed=42),
    },
}


# ============================================================
# 3. Search spaces (for Optuna or manual sweeps)
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
        "loss_fn": ["MSE", "L1"],
        "scheduler": ["None", "ReduceLROnPlateau", "CosineAnnealingLR"],
        "patience": [5, 10],
        "epochs": [25, 50, 100, 200],
        "seed": [21, 42, 77, 99],
    },
}


# ============================================================
# 4. Helper functions
# ============================================================

def get_config(name: str = "unet_baseline") -> Dict[str, Any]:
    """
    Returns a full configuration dictionary for a named experiment.

    Each configuration includes:
        - model: dict of model hyperparameters
        - training: dict of training hyperparameters (including seed & device)
        - seed: global integer seed for reproducibility
    """
    if name not in BASELINE_CONFIGS:
        raise ValueError(f"Unknown config name: {name}")

    cfg = BASELINE_CONFIGS[name]

    # Convert dataclasses to dictionaries
    config_dict = {
        "model": asdict(cfg["model"]),
        "training": asdict(cfg["training"]),
        "seed": cfg["training"].seed,  # ✅ add top-level seed for main.py
    }

    # Make sure device is detected dynamically
    config_dict["training"]["device"] = (
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    return config_dict


def random_config() -> Dict[str, Any]:
    """
    Samples a random configuration from the SEARCH_SPACES.
    Useful for exploratory training runs or grid search.
    """
    model_cfg = {k: random.choice(v) for k, v in SEARCH_SPACES["model"].items()}
    train_cfg = {k: random.choice(v) for k, v in SEARCH_SPACES["training"].items()}

    # Auto-detect device and attach a top-level seed
    train_cfg["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    seed = train_cfg.get("seed", 42)

    return {
        "model": model_cfg,
        "training": train_cfg,
        "seed": seed,
    }


def list_configs() -> List[str]:
    """Lists all predefined configuration names."""
    return list(BASELINE_CONFIGS.keys())


# ============================================================
# 5. Sanity test (manual execution)
# ============================================================

if __name__ == "__main__":
    print("Available configs:", list_configs())
    cfg = get_config("unet_baseline")
    print("\nBaseline Config:")
    print(cfg)

    print("\nRandom Config Sample:")
    print(random_config())
