# ENGS 117 – Project 4: Deep Learning for Single-Pixel Image Reconstruction

## Overview
This repository implements a deep-learning pipeline for reconstructing images captured with a single-pixel camera using a U-Net architecture in PyTorch. The dataset provides paired images:
- **X** – low-quality initial guess (input)
- **Y** – ground-truth image (target)

The code includes modular configuration, training, evaluation, reporting, and visualization utilities.

---

## Project Goals
1. Implement and train a U-Net for single-pixel image reconstruction.
2. Support a variety of differentiable losses (MSE, MAE, DSSIM, LPIPS, TV, GDL, MS-SSIM).
3. Report comprehensive metrics (MSE, MAE, PSNR, SSIM, DSSIM, LPIPS, TV, GDL, MS-SSIM).
4. Generate figures and a JSON report for each run to enable easy model comparison.

---

## Repository Structure
```text
Project4/
│
├── data/
│   └── Project4_Data.mat              # dataset (X, Y)
│
├── misc/                              # MATLAB preprocessing utilities (optional)
│   ├── Project4_Data_Preprocessing.m
│   ├── RDHadamard.m
│   ├── RDHadamard_4096.mat
│   └── STL10_64.mat
│
├── report/
│   ├── figures/
│   └── project4_report.tex
│
├── results/
│   ├── logs/                          # training logs (.txt)
│   └── runs/                          # per-run outputs (timestamped)
│       ├── best_model.pt
│       ├── last_model.pt
│       ├── training_report.json
│       ├── loss_curve.png
│       ├── metric_*.png               # one file per metric tracked
│       ├── reconstruction_grid.png
│       └── error_distribution.png
│
├── src/
│   ├── __init__.py
│   ├── config.py                      # dataclasses + search spaces
│   ├── data_helper.py                 # .mat loading + DataLoaders (train/val/test)
│   ├── metrics.py                     # metrics & optional perceptual measures
│   ├── train.py                       # training/validation, evaluation, report
│   ├── unet_model.py                  # U-Net definition
│   ├── utils.py                       # device, logging, timers, seeding
│   └── visualize.py                   # plotting & comparison tools
│
├── main.py                            # main entry point (baseline run)
├── requirements.txt                   # Python dependencies
├── Project 4.pdf                      # assignment brief (reference)
└── README.md                          # this file
```

---

## Setup and Installation
```bash
# 1) Create and activate a virtual environment (Python 3.11 recommended)
python3 -m venv venv311
# macOS/Linux:
source venv311/bin/activate
# Windows (PowerShell):
# venv311\Scripts\Activate.ps1

# 2) Upgrade pip and install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

> If you plan to use optional perceptual metrics/losses, also install:
> ```bash
> pip install lpips piq
> ```

---

## How to Run
```bash
python main.py
```

This command executes the full reconstruction pipeline:

1. Loads `Project4_Data.mat` from the `./data/` directory.  
2. Builds and initializes the default **U-Net** model as defined in `src/config.py`.  
3. Trains the model using the parameters from `config.py` (with automatic mixed precision if GPU is available).  
4. Evaluates the trained model on the validation and test sets.  
5. Saves all checkpoints, metrics, and plots in a new timestamped directory under `results/runs/`.

Each run produces the following outputs:

- **Model checkpoints:**  
  `best_model.pt`, `last_model.pt`

- **Training curves:**  
  `loss_curve.png`, `metric_*.png`

- **Reconstruction visualizations:**  
  `reconstruction_grid.png`

- **Error histogram:**  
  `error_distribution.png`

- **Summary report:**  
  `training_report.json` – includes model configuration, timing, and performance metrics.

Example run directory structure:
```text
results/runs/run_2025-10-29_17-45-00/
├── best_model.pt
├── last_model.pt
├── training_report.json
├── loss_curve.png
├── metric_psnr.png
├── metric_ssim.png
├── reconstruction_grid.png
└── error_distribution.png
```

---

## Configuration
Edit or extend `src/config.py`.

### Baselines
- `unet_baseline` – default, reliable starting point
- `unet_small`, `unet_large` – capacity variants
- `unet_perceptual` – LPIPS-driven training
- `unet_structural` – DSSIM-driven training
- `unet_smooth` – TV-regularized training

### Search Spaces
`SEARCH_SPACES` defines candidate values for automated sweeps (e.g., Optuna/manual):
- Model: depth, init_features, activations, dropout, kernel_size, padding, etc.
- Training: lr, batch_size, optimizer, weight_decay, loss_fn, scheduler, epochs, patience, seed.
- `error_on_validation`: list of metric names to compute each epoch.

---

## Losses and Metrics

### Differentiable Losses (usable for training)
- `mse` (L2), `mae` (L1)
- `dssim` (1 − SSIM)/2 via differentiable implementation
- `lpips` (perceptual; requires `lpips`)
- `tv` (total variation regularization)
- `gdl` (gradient difference loss)
- `ms_ssim` (multi-scale SSIM; requires `piq`)

### Reported Metrics (for validation/testing)
- `mse`, `mae`, `psnr`, `ssim`, `dssim`, `lpips`, `tv`, `gdl`, `ms_ssim`

Configure which metrics to compute each epoch via:
```python
TrainingConfig.error_on_validation = [
    "mse", "mae", "psnr", "ssim", "dssim", "gdl", "tv", "lpips", "ms_ssim"
]
```

---

## Visualization
Key utilities in `src/visualize.py`:
- `plot_training_history(history, save_dir)`  
- `visualize_batch_grid(X, Y_true, Y_pred, n, save_dir)`  
- `plot_error_distribution(Y_true, Y_pred, save_dir)`  

All figures are saved in the run’s directory in `results/runs/...`.

---

## Training Reports
Each run saves `results/runs/<timestamp>/training_report.json` with:
- Model and training configuration
- Best and final validation loss
- Final validation metrics
- Test metrics
- Timing summary (average epoch, total training time)
- Best epoch index

---

## Data
Place the dataset at:
```
./data/Project4_Data.mat
```
`src/data_helper.py` automatically splits into 80% train, 10% val, 10% test.

---

## Reproducibility
- Global seed set through `src/utils.py:set_seed`.
- Device auto-detection (CUDA → MPS → CPU).
- Checkpoints saved each epoch (`last_model.pt`) and on best validation (`best_model.pt`).

---

## Troubleshooting
- **Out of memory (GPU):** Reduce `batch_size` or use `unet_small`.
- **LPIPS/MS-SSIM unavailable:** Install extras (`pip install lpips piq`).
- **Slow CPU training:** Lower model depth or disable heavy validation metrics.

---

## Author
Kaelen Baird  
ENGS 117 – Computational Imaging (Fall 2025)
