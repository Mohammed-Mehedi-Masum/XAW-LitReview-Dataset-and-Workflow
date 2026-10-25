# XAW-LitReview: Explainable Agentic Workflow with Confidence-Gated Human-in-the-Loop for Automated Literature Screening

This repository contains the dataset, n8n workflow configuration, raw model experimental outputs, and evaluation scripts for the paper:

> **XAW-LitReview: Explainable Agentic Workflow with Confidence-Gated Human-in-the-Loop for Automated Literature Screening**  
> Mohammed Mehedi Masum and Mahfuzulhoq Chowdhury  
> Department of Computer Science and Engineering, Chittagong University of Engineering and Technology (CUET), Bangladesh.

---

## Repository Structure

- `Dataset Annotated by the First Author.csv`: Ground-truth labels from primary annotation (100 papers).
- `Dataset Annotated by the Supervisor.csv`: Independent second-reviewer annotations for inter-rater reliability.
- `Full workflow.json`: Exported n8n workflow for deployment.
- `requirements.txt`: Python package requirements.
- `experiments/`: Raw prediction and confidence CSVs for all experimental runs across threshold configurations ($\tau \in \{0.60, 0.75, 0.90\}$) for Llama-3.1-8B-Instant and GPT-5-mini.
- `scripts/`:
  - `evaluate_thresholds.py`: Evaluates the canonical predictions across thresholds ($\tau \in [0.60, 0.95]$), computing score-based deferrals, parse failures, coverage, and selective accuracy (reproduces Table 2).
  - `compute_statistics.py`: Calculates Brier scores, Expected Calibration Error (ECE), paired McNemar's tests, and 95% bootstrap confidence intervals.
  - `plot_figures.py`: Generates the confusion matrices (Figure 2), threshold trade-off curves (Figure 3), and ablation study plot (Figure 4).

---

## Setup and Reproduction

### 1. Requirements

Install dependencies via pip:

```bash
pip install -r requirements.txt
```

### 2. Run Evaluation and Statistical Scripts

From the repository root:

```bash
# Reproduce Table 2 (Threshold sweep and selective accuracy)
python scripts/evaluate_thresholds.py

# Compute calibration metrics and statistical significance
python scripts/compute_statistics.py

# Generate Figures 2, 3, and 4
python scripts/plot_figures.py
```

---

## Workflow Deployment (n8n)

1. Open your n8n instance and select **Import from File**, then upload `Full workflow.json`.
2. Connect your credentials:
   - Google Sheets OAuth2 (for paper title/abstract input and result logging)
   - Groq Cloud / OpenAI API keys (in the Chat Model node)
   - SMTP credentials (for email-based human review approval triggers)
3. Set the desired confidence threshold in the IF condition node (e.g., `0.90`) and execute.

---

## Citation

```bibtex
@article{masum2026xawlitreview,
  title={XAW-LitReview: Explainable Agentic Workflow with Confidence-Gated Human-in-the-Loop for Automated Literature Screening},
  author={Masum, Mohammed Mehedi and Chowdhury, Mahfuzulhoq},
  journal={Discover Applied Sciences},
  year={2026},
  publisher={Springer Nature}
}
```
