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
from src.utils import set_seed, make_run_dir, print_config, log_message
from src.config import get_config
from src.data_helper import create_dataloaders
from src.unet_model import build_unet
from src.train import train_model, evaluate_model
from src.visualize import plot_training_history, visualize_batch_grid


# ============================================================
# Logging helper
# ============================================================
def setup_logging():
    """Creates timestamped log file inside ./results/logs/"""
    log_dir = "results/logs"
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"run_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.txt")
    return log_path


# ============================================================
# Baseline Run (Verify Full Pipeline)
# ============================================================
def baseline_run():
    print("\n========== [BASELINE PIPELINE RUN] ==========")

    # --- Setup configuration and environment ---
    config = get_config("unet_baseline")
    set_seed(config["seed"])

    run_dir = make_run_dir("results/runs")
    log_file = setup_logging()

    # ✅ Inject log_file path into training config for global access
    if "training" in config and isinstance(config["training"], dict):
        config["training"]["log_file"] = log_file

    # Log config to both console and file
    print_config(config, log_path=log_file)
    log_message(f"[RUN START] Baseline experiment in {run_dir}", log_file)

    # --- Data loading ---
    mat_path = "./data/Project4_Data.mat"
    train_loader, val_loader, test_loader = create_dataloaders(mat_path, config)
    log_message("[DATA] Dataloaders created successfully", log_file)

    # --- Model creation ---
    model_cfg = config["model"]
    model = build_unet(init_features=model_cfg["init_features"])
    log_message(f"[MODEL] U-Net initialized with {model_cfg['init_features']} base features", log_file)

    # --- Training ---
    ckpt_best = os.path.join(run_dir, "best_model.pt")
    ckpt_last = os.path.join(run_dir, "last_model.pt")
    model, history = train_model(
        model,
        train_loader,
        val_loader,
        config,
        save_path=ckpt_best,
        save_last_path=ckpt_last,
        resume_path=None,
    )
    log_message("[TRAINING] Completed baseline training", log_file)

    # --- Evaluation ---
    metrics = evaluate_model(model, test_loader, config)
    log_message(f"[TEST] Final metrics: {metrics}", log_file)

    # --- Visualization ---
    device = torch.device("mps" if torch.backends.mps.is_available()
                        else "cuda" if torch.cuda.is_available()
                        else "cpu")
    model = model.to(device)

    plot_training_history(history, save_dir=run_dir, show=False)
    X_batch, Y_batch = next(iter(test_loader))
    with torch.no_grad():
        X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
        preds = model(X_batch)
    visualize_batch_grid(X_batch, Y_batch, preds, n=5, save_dir=run_dir, show=False)

# ============================================================
# Model Architecture Search (Commented)
# ============================================================
"""
def model_search():
    print("\n========== [MODEL ARCHITECTURE SEARCH] ==========")
    base_config = get_config("baseline")
    architectures = ["unet_small", "unet_base", "unet_deep"]

    for arch in architectures:
        cfg = get_config(arch)
        run_dir = make_run_dir(f"results/model_search/{arch}")
        log_file = setup_logging()

        # ✅ Inject log_file into config for consistent logging
        if "training" in cfg and isinstance(cfg["training"], dict):
            cfg["training"]["log_file"] = log_file

        print_config(cfg, log_path=log_file)
        log_message(f"[MODEL SEARCH] Testing {arch}", log_file)

        train_loader, val_loader, test_loader = create_dataloaders("./data/Project4_Data.mat", cfg)
        model = build_unet(init_features=cfg["model"]["init_features"])
        _, history = train_model(model, train_loader, val_loader, cfg, save_path=f"{run_dir}/best.pt")
        metrics = evaluate_model(model, test_loader, cfg)
        log_message(f"{arch} metrics: {metrics}", log_file)
"""


# ============================================================
#  Hyperparameter Search (Commented)
# ============================================================
"""
def hyperparameter_search():
    print("\n========== [HYPERPARAMETER SEARCH] ==========")
    import optuna
    cfg = get_config("baseline")
    mat_path = "./data/Project4_Data.mat"
    train_loader, val_loader, test_loader = create_dataloaders(mat_path, cfg)

    log_file = setup_logging()
    cfg["training"]["log_file"] = log_file  # ✅ Inject for consistency

    def objective(trial):
        lr = trial.suggest_loguniform("lr", 1e-5, 1e-2)
        batch_size = trial.suggest_categorical("batch_size", [8, 16, 32])
        optimizer = trial.suggest_categorical("optimizer", ["Adam", "AdamW", "RMSprop"])
        cfg["training"].update(lr=lr, batch_size=batch_size, optimizer=optimizer)

        log_message(f"Trial config: lr={lr}, batch_size={batch_size}, optimizer={optimizer}", log_file)

        model = build_unet(init_features=cfg["model"]["init_features"])
        _, _ = train_model(model, train_loader, val_loader, cfg)
        metrics = evaluate_model(model, test_loader, cfg)
        return metrics["psnr"]

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=10)
    print("Best hyperparameters:", study.best_params)
    log_message(f"Best hyperparameters: {study.best_params}", log_file)
"""


# ============================================================
# Final Training with Best Config (Commented)
# ============================================================
"""
def final_training():
    print("\n========== [ FINAL TRAINING] ==========")
    config = get_config("best_found")
    set_seed(config["seed"])
    run_dir = make_run_dir("results/final_training")
    log_file = setup_logging()

    config["training"]["log_file"] = log_file  # ✅ Inject for consistent logging

    print_config(config, log_path=log_file)
    log_message("[FINAL TRAIN] Starting final training run", log_file)

    train_loader, val_loader, test_loader = create_dataloaders("./data/Project4_Data.mat", config)
    model = build_unet(init_features=config["model"]["init_features"])
    model, history = train_model(model, train_loader, val_loader, config,
                                 save_path=f"{run_dir}/best.pt", save_last_path=f"{run_dir}/last.pt")
    metrics = evaluate_model(model, test_loader, config)
    log_message(f"[FINAL TRAIN] Best configuration metrics: {metrics}", log_file)
"""


# ============================================================
#  Entrypoint
# ============================================================
if __name__ == "__main__":
    # Run the baseline verification pipeline first
    baseline_run()

    # Uncomment the sections below as you progress:
    # model_search()
    # hyperparameter_search()
    # final_training()
