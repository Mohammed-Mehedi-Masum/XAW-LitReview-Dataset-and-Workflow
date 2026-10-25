import os
import pandas as pd
from sklearn.metrics import accuracy_score

gt_file = 'Dataset Annotated by the First Author.csv'
if not os.path.exists(gt_file):
    gt_file = os.path.join('..', gt_file)

gt_df = pd.read_csv(gt_file)
gt_map = gt_df.set_index('hw')['decision'].str.strip().str.capitalize()

thresholds = [0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]

def evaluate(csv_path, model_name, pred_col='predicted_decision'):
    if not os.path.exists(csv_path):
        csv_path = os.path.join('..', csv_path)
    df = pd.read_csv(csv_path)
    df['truth'] = df['hw'].map(gt_map)
    df['pred'] = df[pred_col].str.strip().str.capitalize()
    
    acc = accuracy_score(df['truth'], df['pred'])
    print(f"\n{model_name} (Standalone accuracy: {acc:.4f})")
    print(f"{'Threshold':>10} | {'Score-based':>12} | {'Parse error':>12} | {'Total deferred':>14} | {'Coverage':>10} | {'Selective Acc':>14}")
    print("-" * 85)
    
    for tau in thresholds:
        nan_mask = df['predicted_confidence'].isna()
        score_mask = df['predicted_confidence'].notna() & (df['predicted_confidence'] < tau)
        auto_mask = df['predicted_confidence'].notna() & (df['predicted_confidence'] >= tau)
        
        n_nan = nan_mask.sum()
        n_score = score_mask.sum()
        total_def = n_nan + n_score
        coverage = auto_mask.mean() * 100
        
        auto_df = df[auto_mask]
        sel_acc = accuracy_score(auto_df['truth'], auto_df['pred']) if len(auto_df) > 0 else 0.0
        print(f"{tau:>10.2f} | {n_score:>12} | {n_nan:>12} | {total_def:>14} | {coverage:>9.1f}% | {sel_acc:>14.4f}")

if __name__ == '__main__':
    evaluate('experiments/Experiment_results_0.60 - Sheet1.csv', 'Llama-3.1-8B-Instant', 'predicted_decision')
    evaluate('experiments/Experiment_results_0.60_gpt5.5mini - Sheet1.csv', 'GPT-5-mini', 'ai_proposed_decision')
