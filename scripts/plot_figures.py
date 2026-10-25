import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

gt_file = 'Dataset Annotated by the First Author.csv'
if not os.path.exists(gt_file):
    gt_file = os.path.join('..', gt_file)

gt_df = pd.read_csv(gt_file)
gt_map = gt_df.set_index('hw')['decision'].str.strip().str.capitalize()

def get_path(fname):
    p = os.path.join('experiments', fname)
    return p if os.path.exists(p) else os.path.join('..', p)

plt.style.use('seaborn-v0_8-whitegrid')

# -------------------------------------------------------------
# Figure 2: Confusion Matrices
# -------------------------------------------------------------
df_l90 = pd.read_csv(get_path('Experiment_results_0.90 - Sheet1.csv'))
df_g90 = pd.read_csv(get_path('Experiment_results_0.90_gpt5.5mini - Sheet1.csv'))

for df, out_name in [(df_l90, 'confusion_matrix_llama.pdf'), (df_g90, 'confusion_matrix_gpt.pdf')]:
    y_true = df['hw'].map(gt_map)
    y_pred = df['predicted_decision'].str.strip().str.capitalize()
    cm = confusion_matrix(y_true, y_pred, labels=['Include', 'Exclude'])
    
    plt.figure(figsize=(4, 3.2), dpi=300)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Include', 'Exclude'], yticklabels=['Include', 'Exclude'])
    plt.ylabel('Ground Truth')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(out_name)
    plt.close()
print("Saved Figure 2 confusion matrices.")

# -------------------------------------------------------------
# Figure 3: Threshold Curves
# -------------------------------------------------------------
df_l60 = pd.read_csv(get_path('Experiment_results_0.60 - Sheet1.csv'))
df_g60 = pd.read_csv(get_path('Experiment_results_0.60_gpt5.5mini - Sheet1.csv'))

df_l60['truth'] = df_l60['hw'].map(gt_map)
df_l60['pred']  = df_l60['predicted_decision'].str.strip().str.capitalize()
df_g60['truth'] = df_g60['hw'].map(gt_map)
df_g60['pred']  = df_g60['ai_proposed_decision'].str.strip().str.capitalize()

thresholds = [0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]

def get_curves(df):
    def_rates, sel_accs = [], []
    for tau in thresholds:
        deferred = df[df['predicted_confidence'].isna() | (df['predicted_confidence'] < tau)]
        auto = df[df['predicted_confidence'].notna() & (df['predicted_confidence'] >= tau)]
        def_rates.append(len(deferred) / len(df))
        sel_accs.append((auto['truth'] == auto['pred']).mean() if len(auto) > 0 else np.nan)
    return def_rates, sel_accs

l_def, l_sel = get_curves(df_l60)
g_def, g_sel = get_curves(df_g60)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), dpi=300)
ax1.plot(thresholds, g_def, marker='s', color='#0F4C81', label='GPT-5-mini')
ax1.plot(thresholds, l_def, marker='o', color='#D9531E', label='Llama-3.1-8B-Instant')
ax1.set_title('(a) Deferral Rate vs. Threshold', fontsize=11, fontweight='bold')
ax1.set_xlabel('Confidence Threshold (tau)')
ax1.set_ylabel('Deferral Rate')
ax1.set_xticks(thresholds)
ax1.set_ylim(-0.02, 1.02)
ax1.legend(loc='upper left')

ax2.plot(thresholds, g_sel, marker='s', color='#0F4C81', label='GPT-5-mini')
ax2.plot(thresholds, l_sel, marker='o', color='#D9531E', label='Llama-3.1-8B-Instant')
ax2.set_title('(b) Selective Accuracy on Accepted Papers', fontsize=11, fontweight='bold')
ax2.set_xlabel('Confidence Threshold (tau)')
ax2.set_ylabel('Selective Accuracy')
ax2.set_xticks(thresholds)
ax2.set_ylim(0.65, 1.02)
ax2.legend(loc='lower right')

plt.tight_layout()
plt.savefig('figure3_threshold_tradeoff.pdf')
plt.close()
print("Saved Figure 3.")

# -------------------------------------------------------------
# Figure 4: Ablation Study
# -------------------------------------------------------------
configs = ['LLM Only', 'LLM + Explainability', 'LLM + HITL', 'Full System']
x = np.arange(len(configs))
width = 0.35

fig, ax = plt.subplots(figsize=(6, 4), dpi=300)
rects1 = ax.bar(x - width/2, [0.72, 0.72, 0.94, 0.94], width, label='Llama-3.1-8B-Instant', color='#D9531E')
rects2 = ax.bar(x + width/2, [0.91, 0.91, 0.93, 0.93], width, label='GPT-5-mini', color='#0F4C81')

for rects in [rects1, rects2]:
    for rect in rects:
        h = rect.get_height()
        ax.annotate(f'{h:.2f}', xy=(rect.get_x() + rect.get_width() / 2, h),
                    xytext=(0, 2), textcoords="offset points", ha='center', va='bottom', fontsize=8.5)

ax.set_title('Ablation Decomposition Across System Configurations', fontsize=11, fontweight='bold')
ax.set_ylabel('Accuracy')
ax.set_xticks(x)
ax.set_xticklabels(configs, fontsize=9)
ax.set_ylim(0.5, 1.05)
ax.legend(loc='lower right')

plt.tight_layout()
plt.savefig('figure4_ablation.pdf')
plt.close()
print("Saved Figure 4.")
