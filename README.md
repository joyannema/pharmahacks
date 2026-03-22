# Predicting Molecular Toxicity — PharmaHacks 2026

## Team
Joyanne Ma, Akhila Raj, Dorra Tray

## Approach
We predict rat oral LD50 from molecular SMILES strings using:
- Morgan fingerprints + physicochemical descriptors (RDKit)
- XGBoost + Random Forest ensemble
- SHAP analysis for interpretability

## Results
- Test R² = 0.6368
- Test MAE = 0.4241
- Test RMSE = 0.5697

## How to run
1. Open in Google Colab
2. Run all cells top to bottom
3. Final test evaluation is in the last section

## Requirements
pyTDC, rdkit, xgboost, scikit-learn, shap, pandas, numpy, matplotlib, seaborn, pillow
