"""Evaluation metrics, plot helpers, and unified MLflow logging."""
from __future__ import annotations

import json
import pickle
import tempfile
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

import mlflow
import mlflow.sklearn

# Optional MLflow model-flavour submodules — imported at module level so that
# function-level `import mlflow.X` statements don't accidentally shadow the
# global `mlflow` name (Python marks it as local for the whole function body).
try:
    import mlflow.catboost as _mlflow_catboost
except Exception:
    _mlflow_catboost = None  # type: ignore[assignment]

try:
    import mlflow.lightgbm as _mlflow_lightgbm
except Exception:
    _mlflow_lightgbm = None  # type: ignore[assignment]

try:
    import mlflow.pytorch as _mlflow_pytorch
except Exception:
    _mlflow_pytorch = None  # type: ignore[assignment]

# ── Core metrics ─────────────────────────────────────────────────────────────

def compute_all_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float = 0.5,
    k: int | None = None,
) -> dict[str, float]:
    """Compute PR-AUC (primary), ROC-AUC, precision, recall, F1, and optionally P@K / R@K."""
    y_pred = (y_score >= threshold).astype(int)
    metrics: dict[str, float] = {
        "pr_auc":    float(average_precision_score(y_true, y_score)),
        "roc_auc":   float(roc_auc_score(y_true, y_score)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall":    float(recall_score(y_true, y_pred, zero_division=0)),
        "f1":        float(f1_score(y_true, y_pred, zero_division=0)),
    }
    if k is not None:
        metrics[f"precision_at_{k}"] = precision_at_k(y_true, y_score, k)
        metrics[f"recall_at_{k}"]    = recall_at_k(y_true, y_score, k)
    return metrics


def precision_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int) -> float:
    """Fraction of positives among the top-K highest-scored sessions."""
    k = min(k, len(y_score))
    top_k_idx = np.argsort(y_score)[-k:]
    return float(y_true[top_k_idx].mean())


def recall_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int) -> float:
    """Fraction of all positives captured within the top-K scored sessions."""
    k = min(k, len(y_score))
    n_pos = int(y_true.sum())
    if n_pos == 0:
        return 0.0
    top_k_idx = np.argsort(y_score)[-k:]
    return float(y_true[top_k_idx].sum() / n_pos)


# ── Plots ────────────────────────────────────────────────────────────────────

def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Confusion Matrix",
) -> plt.Figure:
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax)
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Other", "Target"])
    ax.set_yticklabels(["Other", "Target"])
    thresh = cm.max() / 2.0
    for i in range(2):
        for j in range(2):
            ax.text(
                j, i, str(cm[i, j]), ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
            )
    ax.set_ylabel("True label")
    ax.set_xlabel("Predicted label")
    ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_pr_curve(
    y_true: np.ndarray,
    y_score: np.ndarray,
    label: str = "model",
) -> plt.Figure:
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    ap = average_precision_score(y_true, y_score)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(recall, precision, linewidth=2, label=f"{label} (AP={ap:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision–Recall Curve")
    ax.set_xlim([0.0, 1.05])
    ax.set_ylim([0.0, 1.05])
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


# ── Model comparison ─────────────────────────────────────────────────────────

def compare_models(results_dict: dict[str, dict]) -> pd.DataFrame:
    """Build a comparison DataFrame sorted by PR-AUC descending.

    Args:
        results_dict: {model_name: metrics_dict}
    """
    rows = [{"model": name, **metrics} for name, metrics in results_dict.items()]
    df = pd.DataFrame(rows).set_index("model")
    if "pr_auc" in df.columns:
        df = df.sort_values("pr_auc", ascending=False)
    return df


# ── MLflow helper ────────────────────────────────────────────────────────────

def log_run_to_mlflow(
    params: dict[str, Any],
    metrics: dict[str, float],
    figures: dict[str, plt.Figure] | None = None,
    model: Any = None,
    model_name: str = "model",
    extra_artifacts: dict[str, Any] | None = None,
    model_flavor: str = "sklearn",
) -> str:
    """Log params, metrics, figures, artifacts, and the model to the active MLflow run.

    model_flavor: 'sklearn' | 'catboost' | 'lightgbm' | 'pytorch'

    extra_artifacts values are auto-serialized:
        pd.DataFrame  → CSV
        np.ndarray    → .npy
        dict / list   → JSON
        anything else → pickle

    Returns the active run ID.
    """
    mlflow.log_params(params)
    mlflow.log_metrics(metrics)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)

        if figures:
            for name, fig in figures.items():
                fig_path = tmp / f"{name}.png"
                fig.savefig(fig_path, dpi=100, bbox_inches="tight")
                mlflow.log_artifact(str(fig_path))
                plt.close(fig)

        if extra_artifacts:
            for name, obj in extra_artifacts.items():
                if isinstance(obj, pd.DataFrame):
                    fname = name if name.endswith(".csv") else name + ".csv"
                    p = tmp / fname
                    obj.to_csv(p, index=False)
                    mlflow.log_artifact(str(p))
                elif isinstance(obj, np.ndarray):
                    fname = name if name.endswith(".npy") else name + ".npy"
                    p = tmp / fname
                    np.save(str(p), obj)
                    mlflow.log_artifact(str(p))
                elif isinstance(obj, (dict, list)):
                    fname = name if name.endswith(".json") else name + ".json"
                    p = tmp / fname
                    p.write_text(json.dumps(obj))
                    mlflow.log_artifact(str(p))
                else:
                    # Pickle fallback (sklearn scalers, etc.)
                    fname = name if "." in name else name + ".pkl"
                    p = tmp / fname
                    with open(p, "wb") as f:
                        pickle.dump(obj, f)
                    mlflow.log_artifact(str(p))

    if model is not None:
        # Use explicit `from` imports to avoid binding `mlflow` as a local
        # variable (which would cause UnboundLocalError at mlflow.log_params above).
        if model_flavor == "catboost":
            from mlflow.catboost import log_model as _log_model_fn
        elif model_flavor == "lightgbm":
            from mlflow.lightgbm import log_model as _log_model_fn
        elif model_flavor == "pytorch":
            from mlflow.pytorch import log_model as _log_model_fn
        else:
            from mlflow.sklearn import log_model as _log_model_fn
        _log_model_fn(model, model_name)

    return mlflow.active_run().info.run_id
