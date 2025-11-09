# Classification-of-CDVM
Code for PLOS ONE manuscript:*Classification of Current Density Density Vector Map Using Transformer Hybrid Residual Network*

---
## Hyperparameter Optimization

1. Configure the `data_dir` (data path) and result saving paths in the script.
   
2. Run `xgc3.py` to start Bayesian optimization and experiments:
   ```bash
   python xgc3.py
   
3. Check results in generated Excel files and saved model weights (.pth files).

---
## Comparison and ablation studies

Experimental Steps

1. Open the main script (recommended filename: `xgct.py`).
   
2. Locate the `best_params` dictionary and confirm/modify the optimized hyperparameters:
   ```bash
   best_params = {
    'd_model_multiplier': 33.124159681292284,
    'dim_feedforward': 1334.7885171995422,
    'dropout': 0.6085392166316497,
    'lr': 2.972939149646677e-05,
    'nhead': 8,
    'num_layers': 1,
    'weight_decay': 0.000987018067234512}
   
3. Update `data_dir` (path to training/test data) and result saving paths (for Excel files and .pth weights) in the script to match your local environment.

Run Experiments

1. The experiments include 7 model variants:
 - Untrained ResNet
 - Pretrained ResNet
 - Untrained ResNet + Transformer
 - Pretrained ResNet + Transformer (proposed model)
 - Pretrained ResNet + LSTM
 - Simple CNN
 - Simple MLP

2. Check Results
 - Training history (loss, accuracy per epoch) is saved in Excel files (e.g., `pretrained_resnet_transformer_history1.xlsx`).
 - Best model weights are saved as .pth files (e.g., `rate000.pth`).
 - A comprehensive comparison table (`ablation_study_results1.xlsx`) includes train/validation/test accuracy for all models.
 - Real-time training logs and final results are printed in the console.

---
## Compatibility
 - Ubuntu 18.04.5 LTS / Windows 10 / macOS 10.15+
 - CUDA 11.3 (GPU support) / CPU
 - CUDNN 8.2.1
 - Python 3.8.10
 - numpy 1.21.6
 - torch 1.10.1
 - torchvision 0.11.2
 - pandas 1.3.5
 - scikit-learn 1.0.2
 - optuna 3.2.0
 - openpyxl 3.0.10



