# ============================================================
# LD50 Acute Toxicity Prediction — PharmaHacks 2026
# Predicting Molecular Toxicity with Machine Learning
# Team: Joyanne Ma, Akhila Raj, Dorra Tray
# Date: March 22, 2026
# Dataset: LD50_Zhu (Therapeutics Data Commons)
# Primary metric: R² on held-out test set
# Final Test R²: 0.6368
# ============================================================


# ── SECTION 0: Install packages ─────────────────────────────

import subprocess
subprocess.run(["pip", "install", "pyTDC", "rdkit", "xgboost",
                "scikit-learn", "pandas", "numpy", "matplotlib",
                "seaborn", "shap", "--quiet"])

# ── SECTION 1: Imports ──────────────────────────────────────

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
np.random.seed(42)

from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from rdkit import DataStructs

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

import xgboost as xgb
from tdc.single_pred import Tox

print("All imports successful!")


# ── SECTION 2: Load dataset ─────────────────────────────────
# We use the TDC standard split for comparability across teams
data = Tox(name='LD50_Zhu')
split = data.get_split()

train_df = split['train']
valid_df = split['valid']
test_df  = split['test']

print(f"Train: {len(train_df)} | Valid: {len(valid_df)} | Test: {len(test_df)}")
print(train_df.head())


# ── SECTION 3: EDA ──────────────────────────────────────────

def get_tox_category(y):
    if y < 1.5:   return 'Highly toxic'
    elif y < 3.0: return 'Moderately toxic'
    elif y < 5.0: return 'Slightly toxic'
    else:         return 'Practically non-toxic'

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, (df, label) in zip(axes, [(train_df,'Train'),(valid_df,'Valid'),(test_df,'Test')]):
    ax.hist(df['Y'], bins=40, color='steelblue', edgecolor='white', alpha=0.85)
    ax.axvline(1.5, color='red',    linestyle='--', label='<1.5 highly toxic')
    ax.axvline(3.0, color='orange', linestyle='--', label='1.5-3 moderate')
    ax.axvline(5.0, color='green',  linestyle='--', label='3-5 slight')
    ax.set_title(f'{label} — log(LD50) distribution')
    ax.set_xlabel('log(LD50)')
    ax.legend(fontsize=7)
plt.tight_layout()
plt.show()

train_df['tox_cat'] = train_df['Y'].apply(get_tox_category)
print(train_df['tox_cat'].value_counts())


# ── SECTION 4: Featurization ────────────────────────────────
# Convert SMILES strings into 2066 numerical features:
# - 2048 Morgan fingerprint bits (radius=2)
# - 18 physicochemical descriptors via RDKit

DESCRIPTOR_NAMES = [
    'MolWt', 'MolLogP', 'NumHDonors', 'NumHAcceptors', 'TPSA',
    'NumRotatableBonds', 'NumAromaticRings', 'NumAliphaticRings',
    'RingCount', 'FractionCSP3', 'NumHeavyAtoms', 'NumHeteroatoms',
    'BalabanJ', 'BertzCT', 'Chi0v', 'Chi1v', 'Kappa1', 'Kappa2', 'Kappa3',
]

def smiles_to_morgan(smiles, radius=2, n_bits=2048):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=n_bits)
    arr = np.zeros((n_bits,), dtype=np.float32)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr

def smiles_to_descriptors(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.full(len(DESCRIPTOR_NAMES), np.nan)
    desc_funcs = dict(Descriptors.descList)
    values = []
    for name in DESCRIPTOR_NAMES:
        try:
            val = desc_funcs[name](mol)
        except Exception:
            val = np.nan
        values.append(val)
    return np.array(values, dtype=np.float32)

def featurize_df(df, radius=2, n_bits=2048):
    fps, descs = [], []
    for smi in df['Drug'].tolist():
        fp = smiles_to_morgan(smi, radius, n_bits)
        d  = smiles_to_descriptors(smi)
        fps.append(fp if fp is not None else np.zeros(n_bits, dtype=np.float32))
        descs.append(d)
    return np.concatenate([np.stack(fps), np.stack(descs)], axis=1)

n_fp_bits = 2048

print("Featurizing train set...")
X_train_raw = featurize_df(train_df)
print("Featurizing valid set...")
X_valid_raw = featurize_df(valid_df)
print("Featurizing test set...")
X_test_raw  = featurize_df(test_df)

y_train = train_df['Y'].values
y_valid = valid_df['Y'].values
y_test  = test_df['Y'].values

# Impute NaNs using train medians only (no data leakage) 
train_medians = np.nanmedian(X_train_raw[:, n_fp_bits:], axis=0)

def impute(X, medians):
    X = X.copy()
    desc = X[:, n_fp_bits:]
    nan_mask = np.isnan(desc)
    desc[nan_mask] = np.take(medians, np.where(nan_mask)[1])
    X[:, n_fp_bits:] = desc
    return X

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='median')
X_train = imputer.fit_transform(X_train_raw)
X_valid = imputer.transform(X_valid_raw)
X_test  = imputer.transform(X_test_raw)

print(f"Feature matrix shape: {X_train.shape}")
print(f"   No NaNs: {not np.isnan(X_train).any()}")


# ── SECTION 5: Save arrays ─────

np.save('X_train.npy', X_train)
np.save('X_valid.npy', X_valid)
np.save('X_test.npy',  X_test)
np.save('y_train.npy', y_train)
np.save('y_valid.npy', y_valid)
np.save('y_test.npy',  y_test)


# ── SECTION 6: Evaluation helper ────────────────────────────

def evaluate(model, X, y_true, label=''):
    y_pred = model.predict(X)
    r2   = r2_score(y_true, y_pred)
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    print(f"[{label:8s}]  R²={r2:.4f}  MAE={mae:.4f}  RMSE={rmse:.4f}")
    return {'r2': r2, 'mae': mae, 'rmse': rmse, 'y_pred': y_pred}

results = {}


# ── SECTION 7: Baseline — Ridge regression ──────────────────
# Ridge regression as linear baseline

ridge = Pipeline([('scaler', StandardScaler()), ('model', Ridge(alpha=1.0))])
ridge.fit(X_train, y_train)
print("=== Ridge Regression ===")
evaluate(ridge, X_train, y_train, 'Train')
results['Ridge'] = evaluate(ridge, X_valid, y_valid, 'Valid')


# ── SECTION 8: Baseline — Random Forest ─────────────────────
# Random Forest as main baseline
rf = RandomForestRegressor(n_estimators=300, min_samples_leaf=2,
                            n_jobs=-1, random_state=42)
rf.fit(X_train, y_train)
print("=== Random Forest ===")
rf_train = evaluate(rf, X_train, y_train, 'Train')
results['RandomForest'] = evaluate(rf, X_valid, y_valid, 'Valid')
print(f"Train-Valid gap: {rf_train['r2'] - results['RandomForest']['r2']:.4f}  (aim for < 0.10)")


# ── SECTION 9: XGBoost ──────────────────────────────────────
# Gradient boosting with early stopping and regularization
xgb_model = xgb.XGBRegressor(
    n_estimators=1000, learning_rate=0.05, max_depth=6,
    subsample=0.8, colsample_bytree=0.8,
    reg_alpha=0.1, reg_lambda=1.0,
    early_stopping_rounds=50, eval_metric='rmse',
    random_state=42, n_jobs=-1, verbosity=0
)
# Early stopping monitors validation loss to prevent overfitting
xgb_model.fit(X_train, y_train,
              eval_set=[(X_valid, y_valid)],
              verbose=False)

print("=== XGBoost ===")
evaluate(xgb_model, X_train, y_train, 'Train')
results['XGBoost'] = evaluate(xgb_model, X_valid, y_valid, 'Valid')
print(f"Best iteration: {xgb_model.best_iteration}")


# ── SECTION 10: Ensemble ─────────────────────────────────────
# Weighted average of RF (0.2) and XGBoost (0.8)

class WeightedEnsemble:
    def __init__(self, models, weights):
        self.models  = models
        self.weights = weights
    def predict(self, X):
        preds = np.stack([m.predict(X) for m in self.models], axis=1)
        return np.average(preds, axis=1, weights=self.weights)

# Weight tuning: try all RF/XGB weight combinations, pick best on validation
best_w, best_r2 = 0.5, -np.inf
for w in np.arange(0.1, 1.0, 0.1):
    ens = WeightedEnsemble([rf, xgb_model], [w, 1-w])
    r2  = r2_score(y_valid, ens.predict(X_valid))
    if r2 > best_r2:
        best_r2, best_w = r2, w

ensemble = WeightedEnsemble([rf, xgb_model], [best_w, 1-best_w])
print(f"=== Ensemble (RF weight={best_w:.1f}, XGB weight={1-best_w:.1f}) ===")
evaluate(ensemble, X_train, y_train, 'Train')
results['Ensemble'] = evaluate(ensemble, X_valid, y_valid, 'Valid')


# ── SECTION 11: Model comparison chart ──────────────────────

comp = pd.DataFrame({
    'Model': list(results.keys()),
    'R²':   [v['r2']   for v in results.values()],
    'MAE':  [v['mae']  for v in results.values()],
    'RMSE': [v['rmse'] for v in results.values()],
}).sort_values('R²', ascending=False)

print("\n=== Validation set comparison ===")
print(comp.to_string(index=False))

fig, axes = plt.subplots(1, 3, figsize=(14, 4))
for ax, metric in zip(axes, ['R²', 'MAE', 'RMSE']):
    ax.bar(comp['Model'], comp[metric], color=['#1D9E75','#7F77DD','#D85A30','#BA7517'])
    ax.set_title(f'Validation {metric}', fontweight='bold')
    ax.tick_params(axis='x', rotation=20)
plt.tight_layout()
plt.show()


# ── SECTION 12: Feature Importance ──────────────────────────
importances    = xgb_model.feature_importances_
desc_imp       = importances[n_fp_bits:]
top_idx        = np.argsort(desc_imp)[::-1][:12]

fig, ax = plt.subplots(figsize=(9, 5))
ax.barh([DESCRIPTOR_NAMES[i] for i in top_idx][::-1],
         desc_imp[top_idx][::-1], color='#5DCAA5')
ax.set_title('Top descriptor importances (XGBoost)', fontweight='bold')
ax.set_xlabel('Feature importance')
plt.tight_layout()
plt.show()


# ── SECTION 13: Predicted vs actual + error analysis ────────

best_model   = ensemble
y_valid_pred = best_model.predict(X_valid)
residuals    = y_valid - y_valid_pred

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

ax = axes[0]
ax.scatter(y_valid, y_valid_pred, alpha=0.3, s=12, color='steelblue')
lims = [min(y_valid.min(), y_valid_pred.min()) - 0.2,
        max(y_valid.max(), y_valid_pred.max()) + 0.2]
ax.plot(lims, lims, 'r--', linewidth=1.5)
ax.set_xlabel('Actual log(LD50)')
ax.set_ylabel('Predicted log(LD50)')
ax.set_title(f'Predicted vs Actual  (R²={r2_score(y_valid,y_valid_pred):.4f})', fontweight='bold')

ax = axes[1]
ax.hist(residuals, bins=50, color='coral', edgecolor='white')
ax.axvline(0, color='black', linestyle='--')
ax.set_xlabel('Residual (actual − predicted)')
ax.set_title('Residual distribution', fontweight='bold')
plt.tight_layout()
plt.show()

# Error by toxicity category
valid_df2 = valid_df.copy().reset_index(drop=True)
valid_df2['abs_error'] = np.abs(residuals)
valid_df2['tox_cat']   = valid_df2['Y'].apply(get_tox_category)
print("\n=== Mean error by toxicity category ===")
print(valid_df2.groupby('tox_cat')['abs_error'].mean().round(3))


# ── SECTION 14: SHAP analysis ───────────────────────────────
import shap
from rdkit.Chem.Draw import rdMolDraw2D
from PIL import Image as PILImage
import io

# Sample 500 molecules for speed
idx      = np.random.choice(len(X_valid), 500, replace=False)
X_samp   = X_valid[idx]
y_samp   = y_valid[idx]
smi_samp = valid_df['Drug'].values[idx]
pred_samp = xgb_model.predict(X_samp)

print("Computing SHAP values (~2 min)...")
explainer   = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(X_samp)
print(f"SHAP values shape: {shap_values.shape}")

# Summary bar chart (descriptors only)
desc_shap = shap_values[:, n_fp_bits:]
mean_abs  = np.abs(desc_shap).mean(axis=0)
top10     = np.argsort(mean_abs)[::-1][:10]

fig, ax = plt.subplots(figsize=(9, 5))
ax.barh([DESCRIPTOR_NAMES[i] for i in top10][::-1],
         mean_abs[top10][::-1], color='#5DCAA5')
ax.set_title('Most influential descriptors (mean |SHAP|)', fontweight='bold')
ax.set_xlabel('Mean |SHAP value|')
plt.tight_layout()
plt.show()

# Full SHAP summary plot
fp_names = [f'Morgan_bit_{i}' for i in range(n_fp_bits)]
shap.summary_plot(desc_shap, X_samp[:, n_fp_bits:],
                  feature_names=DESCRIPTOR_NAMES, max_display=12, show=True)


# ── SECTION 15: Molecule heatmaps ───────────────────────────

def get_atom_shap_scores(smiles, shap_fp_vals, radius=2, n_bits=2048):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, {}
    bit_info = {}
    AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=n_bits, bitInfo=bit_info)
    atom_scores = {a.GetIdx(): 0.0 for a in mol.GetAtoms()}
    for bit, envs in bit_info.items():
        for center, _ in envs:
            atom_scores[center] += shap_fp_vals[bit]
    return mol, atom_scores

def shap_color(score, vmin, vmax):
    if score < 0:
        t = min(abs(score) / max(abs(vmin), 1e-9), 1.0)
        return (1.0, 1.0 - 0.7*t, 1.0 - 0.7*t)
    else:
        t = min(score / max(abs(vmax), 1e-9), 1.0)
        return (1.0 - 0.7*t, 1.0 - 0.7*t, 1.0)

def draw_mol_shap(smiles, shap_fp_vals, size=(400, 300)):
    mol, scores = get_atom_shap_scores(smiles, shap_fp_vals)
    if mol is None:
        return None
    vals = list(scores.values())
    vmin, vmax = min(vals), max(vals)
    colors = {i: shap_color(s, vmin, vmax) for i, s in scores.items()}
    radii  = {i: 0.4 + 0.4*min(abs(s)/max(abs(vmin),abs(vmax),1e-9),1.0)
               for i, s in scores.items()}
    drawer = rdMolDraw2D.MolDraw2DCairo(*size)
    rdMolDraw2D.PrepareAndDrawMolecule(
        drawer, mol,
        highlightAtoms=list(colors.keys()),
        highlightAtomColors=colors,
        highlightAtomRadii=radii,
        highlightBonds=[]
    )
    drawer.FinishDrawing()
    return PILImage.open(io.BytesIO(drawer.GetDrawingText())).convert('RGB')

# Draw 6 toxic + 6 safe molecules
def draw_grid(mol_indices, title, ncols=3):
    nrows = (len(mol_indices) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols*4, nrows*3.2))
    axes = np.array(axes).flatten()
    for ax_i, mol_i in enumerate(mol_indices):
        img = draw_mol_shap(smi_samp[mol_i], shap_values[mol_i, :n_fp_bits])
        if img:
            axes[ax_i].imshow(img)
        axes[ax_i].axis('off')
        c = '#C04828' if y_samp[mol_i] < 2.5 else '#3B6D11'
        axes[ax_i].set_title(
            f"actual={y_samp[mol_i]:.2f}  pred={pred_samp[mol_i]:.2f}",
            fontsize=9, color=c, fontweight='bold')
    for ax_i in range(len(mol_indices), len(axes)):
        axes[ax_i].axis('off')
    fig.suptitle(title, fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.show()

toxic_idx = np.where(y_samp < 2.0)[0][:6]
safe_idx  = np.where(y_samp > 4.5)[0][:6]

if len(toxic_idx) >= 2:
    draw_grid(toxic_idx, 'Highly toxic molecules — red = toxic fragments')
if len(safe_idx) >= 2:
    draw_grid(safe_idx, 'Low-toxicity molecules — blue = safe fragments')


# ── SECTION 16: FINAL TEST EVALUATION ──────────────────────
# Run once at the end — never used for tuning

print("FINAL TEST SET RESULTS")
test_results = evaluate(best_model, X_test, y_test, 'Test')
print(f"\nFinal R²  : {test_results['r2']:.4f}")
print(f"Final MAE : {test_results['mae']:.4f}")
print(f"Final RMSE: {test_results['rmse']:.4f}")

y_test_pred = test_results['y_pred']
fig, ax = plt.subplots(figsize=(6, 5))
ax.scatter(y_test, y_test_pred, alpha=0.3, s=12, color='steelblue')
lims = [min(y_test.min(), y_test_pred.min()) - 0.2,
        max(y_test.max(), y_test_pred.max()) + 0.2]
ax.plot(lims, lims, 'r--')
ax.set_xlabel('Actual log(LD50)')
ax.set_ylabel('Predicted log(LD50)')
ax.set_title(f'Test set: R² = {test_results["r2"]:.4f}', fontweight='bold')
plt.tight_layout()
plt.show()
