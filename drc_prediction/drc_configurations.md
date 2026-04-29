# DRC Prediction Configurations to Test

This document outlines 5 different training configurations for the DRC Prediction task in CircuitNet. The configurations override the default settings in `config/drc_train.yaml` using Hydra syntax. The training script logs metrics to `Aim`, allowing you to easily compare these runs in the Aim dashboard.

## Overview of Tunable Hyperparameters
- **Loss Function (`loss_type`)**: `MSELoss` (default) or `L1Loss`.
- **Learning Rate (`lr`)**: Default `2e-4`. Controls step size during optimization.
- **Weight Decay (`weight_decay`)**: Default `1e-4`. Controls L2 regularization.
- **Batch Size (`batch_size`)**: Default `8`. 
- **Metrics (`eval_metric`)**: Default `[NRMS, SSIM]`. Can include `PSNR` and `EMD`.

---

## 5 Configurations to Test

### 1. The Baseline (Current Default)
Uses standard MSE loss and the default hyperparameters. Good reference point.
```bash
python train.py loss_type=MSELoss lr=2e-4 batch_size=8
```

### 2. L1 Loss Alternative
Switches to L1 (Mean Absolute Error) loss, which is often less sensitive to extreme outliers in DRC maps than MSE.
```bash
python train.py loss_type=L1Loss lr=2e-4 batch_size=8
```

### 3. Higher Learning Rate & Higher Batch Size
Increases batch size for smoother, more stable gradients, and increases the learning rate accordingly. Good for finding broader minima.
*(Note: Requires more VRAM)*
```bash
python train.py loss_type=MSELoss lr=1e-3 batch_size=16
```

### 4. Lower Learning Rate & Higher Weight Decay
Applies stronger regularization (`weight_decay=1e-3` instead of `1e-4`) and a lower learning rate. This helps prevent overfitting on the training set.
```bash
python train.py loss_type=MSELoss lr=5e-5 weight_decay=1e-3 batch_size=8
```

### 5. Expanded Evaluation Metrics
Adds `PSNR` and `EMD` (Earth Mover's Distance) to the tracked metrics alongside `NRMS` and `SSIM` to get a comprehensive perspective on model performance.
```bash
python train.py loss_type=MSELoss eval_metric=[NRMS,SSIM,PSNR,EMD]
```

---

## How to Run and Evaluate

1. **Navigate to the directory**:
   ```bash
   cd drc_prediction
   ```

2. **Run a training configuration**:
   Simply append the hydra overrides to the command. For example:
   ```bash
   python train.py loss_type=L1Loss lr=2e-4 batch_size=8
   ```

3. **Run testing**:
   To evaluate a trained configuration, run the test script. Ensure the metrics match what you want to evaluate:
   ```bash
   python test.py eval_metric=[NRMS,SSIM,PSNR,EMD]
   ```

4. **Compare Results**:
   Because `train.py` uses `aim Run`, all hyperparameter overrides are saved into `run['hparams']`. You can use the Aim UI to compare the validation loss, NRMS, SSIM, PSNR, and EMD across all these runs.