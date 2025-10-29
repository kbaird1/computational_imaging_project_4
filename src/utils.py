"""
utils.py
================
General utility functions for reproducibility, device management, and logging.
Used throughout the project for cleaner, modular code.
"""

import os
import random
import torch
import numpy as np
from datetime import datetime


# ============================================================
# 1. Reproducibility
# ============================================================
def set_seed(seed=42):
    """Set random seed for reproducibility across torch, numpy, and random."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"[utils] Seed set to {seed}")


# ============================================================
# 2. Device Selection
# ============================================================
def get_device():
    """Return the best available device: CUDA, MPS (Mac), or CPU."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        name = torch.cuda.get_device_name(0)
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        name = "Apple MPS (Metal)"
    else:
        device = torch.device("cpu")
        name = "CPU"
    print(f"[utils] Using device: {name}")
    return device


# ============================================================
# 3. Logging Helpers
# ============================================================
def ensure_dir_exists(path):
    """Ensure directory exists before saving files."""
    os.makedirs(os.path.dirname(path), exist_ok=True)

def log_message(message: str, log_path: str = None, console: bool = False):
    """
    Append a log message to file and optionally print to console.
    """
    timestamped = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}"
    if log_path:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        with open(log_path, "a") as f:
            f.write(timestamped + "\n")
    if console:
        print(timestamped)



# ============================================================
# 4. Time & Run Info
# ============================================================
def timestamp():
    """Return a formatted timestamp string."""
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

def make_run_dir(base_dir="results/runs"):
    """Create and return a new directory for the current run."""
    run_dir = os.path.join(base_dir, f"run_{timestamp()}")
    os.makedirs(run_dir, exist_ok=True)
    print(f"[utils] Created new run directory: {run_dir}")
    return run_dir


# ============================================================
# 5. Configuration Printing
# ============================================================
def print_config(config, log_path=None):
    """Nicely format and print a config dictionary. Optionally log to file."""
    output = ["\n[CONFIGURATION]"]
    for k, v in config.items():
        output.append(f"  {k:<20}: {v}")
    formatted = "\n".join(output)
    print(formatted)
    if log_path:
        ensure_dir_exists(log_path)
        with open(log_path, "a") as f:
            f.write(formatted + "\n")



# ============================================================
# 6. Timer Context Manager (optional)
# ============================================================
from contextlib import contextmanager
import time

@contextmanager
def timer(label="Operation"):
    """Context manager for timing code blocks."""
    start = time.time()
    yield
    end = time.time()
    print(f"[TIMER] {label}: {(end - start):.2f}s")
