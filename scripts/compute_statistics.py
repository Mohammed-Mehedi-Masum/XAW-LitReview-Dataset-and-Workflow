import os
import pandas as pd
import numpy as np
from sklearn.metrics import brier_score_loss, accuracy_score
from statsmodels.stats.contingency_tables import mcnemar

gt_file = 'Dataset Annotated by the First Author.csv'
if not os.path.exists(gt_file):
    gt_file = os.path.join('..', gt_file)

gt_df = pd.read_csv(gt_file)
gt_map = gt_df.set_index('hw')['decision'].str.strip().str.capitalize()

def get_path(fname):
    p = os.path.join('experiments', fname)
    return p if os.path.exists(p) else os.path.join('..', p)

df_l60 = pd.read_csv(get_path('Experiment_results_0.60 - Sheet1.csv'))
df_l90 = pd.read_csv(get_path('Experiment_results_0.90 - Sheet1.csv'))
df_g60 = pd.read_csv(get_path('Experiment_results_0.60_gpt5.5mini - Sheet1.csv'))
df_g90 = pd.read_csv(get_path('Experiment_results_0.90_gpt5.5mini - Sheet1.csv'))

y_true = df_l60['hw'].map(gt_map).map({'Include': 1, 'Exclude': 0}).values

pred_l_raw = df_l60['predicted_decision'].str.strip().str.capitalize().map({'Include': 1, 'Exclude': 0}).values
pred_l_hitl = df_l90['predicted_decision'].str.strip().str.capitalize().map({'Include': 1, 'Exclude': 0}).values
pred_g_raw = df_g60['ai_proposed_decision'].str.strip().str.capitalize().map({'Include': 1, 'Exclude': 0}).values
pred_g_hitl = df_g90['predicted_decision'].str.strip().str.capitalize().map({'Include': 1, 'Exclude': 0}).values

def compute_ece(y_true, y_conf, n_bins=10):
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        b_low, b_high = bin_boundaries[i], bin_boundaries[i + 1]
        mask = (y_conf >= b_low) & (y_conf < b_high) if i < n_bins - 1 else (y_conf >= b_low) & (y_conf <= b_high)
        if mask.sum() > 0:
            bin_acc = y_true[mask].mean()
            bin_conf = y_conf[mask].mean()
            ece += (mask.sum() / len(y_true)) * np.abs(bin_acc - bin_conf)
    return ece

def bootstrap_ci(y_true, y_pred, n_resamples=2000, seed=42):
    rng = np.random.default_rng(seed)
    n = len(y_true)
    accs = [accuracy_score(y_true[idx], y_pred[idx]) for idx in (rng.choice(n, size=n, replace=True) for _ in range(n_resamples))]
    return np.percentile(accs, 2.5), np.percentile(accs, 97.5)

# Brier scores
p_l = np.where(df_l60['predicted_decision'].str.strip().str.capitalize() == 'Include', 
               df_l60['predicted_confidence'].fillna(0.5), 
               1.0 - df_l60['predicted_confidence'].fillna(0.5))
p_g = np.where(df_g60['ai_proposed_decision'].str.strip().str.capitalize() == 'Include', 
               df_g60['predicted_confidence'].fillna(0.5), 
               1.0 - df_g60['predicted_confidence'].fillna(0.5))

brier_l = brier_score_loss(y_true, p_l)
brier_g = brier_score_loss(y_true, p_g)

# ECE on confidence of predicted label
ece_l = compute_ece((y_true == pred_l_raw).astype(int), df_l60['predicted_confidence'].fillna(0.5).values)
ece_g = compute_ece((y_true == pred_g_raw).astype(int), df_g60['predicted_confidence'].fillna(0.5).values)

print("Calibration Metrics:")
print(f"  Llama-3.1-8B-Instant : Brier = {brier_l:.4f}, ECE = {ece_l:.4f}")
print(f"  GPT-5-mini           : Brier = {brier_g:.4f}, ECE = {ece_g:.4f}\n")

print("Bootstrap 95% CIs:")
for name, pred in [('Llama Standalone', pred_l_raw), 
                   ('Llama Live HITL (tau=0.90)', pred_l_hitl),
                   ('GPT Standalone', pred_g_raw), 
                   ('GPT Live HITL (tau=0.90)', pred_g_hitl)]:
    low, high = bootstrap_ci(y_true, pred)
    print(f"  {name:<28}: {accuracy_score(y_true, pred):.2f} [{low:.2f}, {high:.2f}]")

print("\nMcNemar Paired Tests:")
for name, p1, p2 in [("Llama Standalone vs. GPT Standalone", pred_l_raw, pred_g_raw),
                     ("Llama Standalone vs. Llama Live HITL", pred_l_raw, pred_l_hitl),
                     ("GPT Standalone vs. GPT Live HITL", pred_g_raw, pred_g_hitl)]:
    tb = pd.crosstab(p1 == y_true, p2 == y_true)
    res = mcnemar(tb, exact=True)
    print(f"  {name:<40}: p = {res.pvalue:.4e}")
