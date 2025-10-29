"""
main.py
================
Main entry point for Project 4: Single-Pixel Image Reconstruction.

Pipeline stages:
  1. Baseline training run (active)
  2. Model architecture search (commented)
  3. Hyperparameter search (commented)
  4. Final training with best configuration (commented)
"""

import os
import torch
from datetime import datetime

from src.utils import (
    set_seed,
    make_run_dir,
    print_config,
    log_message,
)
from src.config import get_config
from src.data_helper import create_dataloaders
from src.unet_model import build_unet
from src.train import (
    train_model,
    evaluate_model,
    generate_training_report,
)
from src.visualize import (
    plot_training_history,
    visualize_batch_grid,
    plot_error_distribution,
    compare_model_metrics,
)


# ============================================================
# Logging Helper
# ============================================================
def setup_logging():
    """Creates timestamped log file inside ./results/logs/."""
    log_dir = "results/logs"
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(
        log_dir, f"run_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.txt"
    )
    return log_path


# ============================================================
# Baseline Pipeline Run
# ============================================================
def baseline_run():
    print("\n========== [BASELINE PIPELINE RUN] ==========")

    # --- Setup configuration and environment ---
    config = get_config("unet_baseline")
    set_seed(config["seed"])

    run_dir = make_run_dir("results/runs")
    log_file = setup_logging()

    # Inject log_file path into training config for unified logging
    if "training" in config and isinstance(config["training"], dict):
        config["training"]["log_file"] = log_file

    # Log config
    print_config(config, log_path=log_file)
    log_message(f"[RUN START] Baseline experiment in {run_dir}", log_file)

    # --- Data loading ---
    mat_path = "./data/Project4_Data.mat"
    train_loader, val_loader, test_loader = create_dataloaders(mat_path, config)
    log_message("[DATA] Dataloaders created successfully", log_file)

    # --- Model creation ---
    model_cfg = config["model"]
    model = build_unet(
        in_channels=model_cfg.get("in_channels", 1),
        out_channels=model_cfg.get("out_channels", 1),
        init_features=model_cfg.get("init_features", 64),
        depth=model_cfg.get("depth", 4),
        activation=model_cfg.get("activation", "ReLU"),
        dropout_rate=model_cfg.get("dropout_rate", 0.0),
        kernel_size=model_cfg.get("kernel_size", 3),
        padding_mode=model_cfg.get("padding_mode", "zeros"),
        output_activation=model_cfg.get("output_activation", "Sigmoid"),
    )
    log_message(
        f"[MODEL] U-Net initialized with {model_cfg['init_features']} base features", log_file
    )

    # --- Training ---
    ckpt_best = os.path.join(run_dir, "best_model.pt")
    ckpt_last = os.path.join(run_dir, "last_model.pt")
    model, history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        save_path=ckpt_best,
        save_last_path=ckpt_last,
        resume_path=None,
    )
    log_message("[TRAINING] Completed baseline training", log_file)

    # --- Evaluation ---
    test_metrics = evaluate_model(model, test_loader, config)
    log_message(f"[TEST] Final metrics: {test_metrics}", log_file)

    # --- Generate Report ---
    report_path = os.path.join(run_dir, "training_report.json")
    generate_training_report(
        output_path=report_path,
        config=config,
        history=history,
        test_metrics=test_metrics,
    )

    # --- Visualization Stage ---
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    model = model.to(device)

    # Loss / metric curves
    plot_training_history(history, save_dir=run_dir, show=False)

    # Example reconstructions
    X_batch, Y_batch = next(iter(test_loader))
    with torch.no_grad():
        X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
        preds = model(X_batch)

    visualize_batch_grid(X_batch, Y_batch, preds, n=5, save_dir=run_dir, show=False)

    # Error distribution visualization
    plot_error_distribution(Y_batch, preds, save_dir=run_dir, show=False)

    # Optional comparison placeholder (for multi-model runs)
    # compare_model_metrics({"unet_baseline": test_metrics["psnr"]}, metric="psnr")

    log_message(f"[OUTPUT] All figures and report saved in {run_dir}", log_file)
    print(f"\n All outputs saved in: {run_dir}")
    print(f"   → Training curves, reconstructions, report.json, checkpoints\n")


# ============================================================
# Entrypoint
# ============================================================
if __name__ == "__main__":
    baseline_run()