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
## Comparison and Ablation Studies

Experimental Steps

1. Open the main script (recommended filename: `xgct.py`).
2. Locate the `best_params` dictionary and confirm/modify the optimized hyperparameters:
   ```bash
   python best_params = {
    'd_model_multiplier': 33.124159681292284,
    'dim_feedforward': 1334.7885171995422,
    'dropout': 0.6085392166316497,
    'lr': 2.972939149646677e-05,
    'nhead': 8,
    'num_layers': 1,
    'weight_decay': 0.000987018067234512}
   
4. 
5. 



