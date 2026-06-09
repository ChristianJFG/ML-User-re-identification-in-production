# Catch Joe — Target-User Detection

Binary detection of a specific user from web browsing sessions.  
Three model approaches, tracked end-to-end with MLflow.

---

## Project layout

```
catch_joe/
├── data/raw/          ← dataset.json (training), verify.json (unlabelled)
├── notebooks/
│   ├── data_exploration.ipynb    ← EDA
│   └── catch_joe_ml.ipynb        ← ML experiment (all three models)
├── scripts/
│   ├── train.py       ← production training entrypoint
│   └── predict.py     ← inference on new sessions
└── src/catch_joe/
    ├── data.py        ← loading, schema, target creation, splits
    ├── features.py    ← site indicators, TF-IDF, metadata preprocessing
    ├── evaluation.py  ← metrics, plots, MLflow logging helper
    └── modeling.py    ← CatBoost, LightGBM, TF-IDF+LR, Siamese encoder
```

---

## Setup

```bash
# Install all dependencies (including catboost, lightgbm, torch)
cd catch_joe
uv sync

# Verify the package imports correctly
uv run python -c "from catch_joe.data import load_sessions; print('OK')"
```

---

## MLflow UI

```bash
cd catch_joe
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5001
# Open http://localhost:5001
```

---

## Notebook experiment

Open `notebooks/catch_joe_ml.ipynb` in VS Code or JupyterLab.

Set `TARGET_USER_ID = 0` (default) at the top of the notebook, or choose a
well-represented user from the `get_user_stats()` cell.  
The notebook runs all three approaches with MLflow tracking and produces a
final comparison table.

---

## Production training

```bash
# Tree-based (CatBoost) — recommended default
uv run python scripts/train.py \
    --approach tree \
    --target-user-id 0 \
    --data-path data/raw/dataset.json \
    --top-k 1000

# TF-IDF + Logistic Regression
uv run python scripts/train.py \
    --approach tfidf \
    --target-user-id 0 \
    --data-path data/raw/dataset.json \
    --max-features 5000

# Siamese contrastive encoder
uv run python scripts/train.py \
    --approach siamese \
    --target-user-id 0 \
    --data-path data/raw/dataset.json \
    --epochs 20

# All runs land in the 'catch_joe_detection' MLflow experiment by default.
# Override with: --experiment-name my_experiment
```

Full CLI reference:

```
uv run python scripts/train.py --help
```

---

## Inference on new sessions

```bash
# Tree or TF-IDF model
uv run python scripts/predict.py \
    --run-id   <mlflow_run_id> \
    --model-type tree \
    --data-path  data/raw/verify.json \
    --output-path data/processed/predictions.csv

# Siamese model (needs --target-user-id to load stored target embeddings)
uv run python scripts/predict.py \
    --run-id   <mlflow_run_id> \
    --model-type siamese \
    --data-path  data/raw/verify.json \
    --target-user-id 0 \
    --output-path data/processed/predictions.csv
```

Output CSV contains one row per session with columns:
- `score` — continuous detection probability [0, 1]
- `predicted_target` — binary prediction at threshold 0.5

---

## Model recommendation

| Approach | Strengths | When to prefer |
|---|---|---|
| **CatBoost** | Handles all metadata categoricals natively; robust to imbalance; interpretable feature importances | **Default recommendation** — strong overall accuracy with low engineering overhead |
| TF-IDF + LR | Captures rare domain co-occurrence signal; fast to train; highly interpretable coefficients | Vocabulary is very large (> 10k unique domains) and domain pattern is the primary discriminator |
| Siamese encoder | Robust across sessions; detects novel behaviour; works with few labelled sessions at inference | Cross-time detection or when target user's sessions are temporally sparse |

**Use CatBoost** unless your evaluation shows TF-IDF or Siamese outperforming it on PR-AUC for the specific `TARGET_USER_ID`.

---

## Metrics tracked per run

| Category | Items |
|---|---|
| Params | `model_type`, `target_user_id`, `split_strategy`, `top_k`/`max_features`, hyperparams, `train_size`, `val_size`, `n_positive_train`, `n_negative_train` |
| Metrics | `pr_auc` *(primary)*, `roc_auc`, `precision`, `recall`, `f1`, `precision_at_K`, `recall_at_K` |
| Artifacts | `confusion_matrix.png`, `pr_curve.png`, `feature_importance.csv` / `top_coefficients.csv`, trained model, preprocessing metadata |
