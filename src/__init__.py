# src/__init__.py
"""
src package initialization for Project 4: Single-Pixel Reconstruction
This package provides modular components for:
    - Data loading (data_helper)
    - Model definition (unet_model)
    - Training (train)
    - Metrics and visualization
    - Utilities (utils)
"""

from .data_helper import load_data, split_data
from .unet_model import build_unet
from .train import train_model
from .metrics import compute_metrics
from .visualize import plot_training_history, visualize_reconstruction
from .utils import set_seed, get_device

__all__ = [
    "load_data",
    "split_data",
    "build_unet",
    "train_model",
    "compute_metrics",
    "plot_training_history",
    "visualize_reconstruction",
    "set_seed",
    "get_device",
]
