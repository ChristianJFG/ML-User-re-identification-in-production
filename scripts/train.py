"""Production training entrypoint with MLflow tracking.

Usage examples:
    uv run python scripts/train.py --approach tree   --target-user-id 0 --data-path data/raw/dataset.json
    uv run python scripts/train.py --approach tfidf  --target-user-id 0 --data-path data/raw/dataset.json
    uv run python scripts/train.py --approach siamese --target-user-id 0 --data-path data/raw/dataset.json
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import mlflow

# Allow running from the repo root or from catch_joe/
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH  = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from catch_joe.data import (
    create_target, get_user_stats, load_sessions,
    stratified_split, time_based_split, validate_schema,
)
from catch_joe.evaluation import (
    compute_all_metrics, log_run_to_mlflow,
    plot_confusion_matrix, plot_pr_curve,
)
from catch_joe.features import (
    METADATA_CAT_COLS, build_tree_features, extract_session_stats,
    get_top_k_domains,
)
from catch_joe.modeling import (
    SIAMESE_NUM_COLS, build_site_vocab, encode_sessions,
    predict_siamese_scores, train_catboost, train_siamese, train_tfidf_lr,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


# ── CLI ───────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Train a target-user detection model and log to MLflow."
    )
    p.add_argument("--approach", required=True, choices=["tree", "tfidf", "siamese"],
                   help="Model approach to train.")
    p.add_argument("--target-user-id", required=True, type=int,
                   help="user_id to use as the positive class.")
    p.add_argument("--data-path", required=True,
                   help="Path to dataset.json (relative to catch_joe/ or absolute).")
    p.add_argument("--split-strategy", default="time", choices=["time", "stratified"],
                   help="Train/val split strategy (default: time).")
    p.add_argument("--test-ratio", type=float, default=0.20,
                   help="Fraction of data held out for validation (default: 0.20).")
    p.add_argument("--experiment-name", default="catch_joe_detection",
                   help="MLflow experiment name.")
    # Tree params
    p.add_argument("--top-k", type=int, default=1000,
                   help="[tree] Number of top-K domain indicator features.")
    p.add_argument("--iterations", type=int, default=500,
                   help="[tree] CatBoost iterations.")
    p.add_argument("--depth", type=int, default=6,
                   help="[tree] CatBoost tree depth.")
    # TF-IDF params
    p.add_argument("--max-features", type=int, default=5000,
                   help="[tfidf] TF-IDF vocabulary size.")
    p.add_argument("--C", type=float, default=1.0,
                   help="[tfidf] Logistic Regression regularisation strength.")
    # Siamese params
    p.add_argument("--epochs", type=int, default=20,
                   help="[siamese] Training epochs.")
    p.add_argument("--lr", type=float, default=1e-3,
                   help="[siamese] Adam learning rate.")
    p.add_argument("--emb-dim", type=int, default=64,
                   help="[siamese] Site embedding dimension.")
    p.add_argument("--batch-size", type=int, default=256,
                   help="[siamese] Batch size.")
    p.add_argument("--neg-ratio", type=int, default=4,
                   help="[siamese] Negative-to-positive pair ratio.")
    return p


# ── Per-approach training functions ──────────────────────────────────────────

def run_tree(args, df_train, df_val, y_train, y_val) -> None:
    logger.info("Building tree features (top_k=%d) ...", args.top_k)
    X_train, X_val, cat_idx, feat_names = build_tree_features(
        df_train, df_val, top_k=args.top_k
    )

    params = {
        "model_type":       "catboost",
        "target_user_id":   args.target_user_id,
        "split_strategy":   args.split_strategy,
        "top_k":            args.top_k,
        "iterations":       args.iterations,
        "learning_rate":    0.05,
        "depth":            args.depth,
        "train_size":       len(df_train),
        "val_size":         len(df_val),
        "n_positive_train": int(y_train.sum()),
        "n_negative_train": int(len(y_train) - y_train.sum()),
    }

    with mlflow.start_run(run_name="catboost_tree"):
        model = train_catboost(
            X_train, y_train, X_val, y_val,
            cat_feature_indices=cat_idx,
            params={"iterations": args.iterations, "depth": args.depth},
        )

        y_score = model.predict_proba(X_val)[:, 1]
        y_pred  = (y_score >= 0.5).astype(int)

        k_eval  = min(500, max(1, int(y_val.sum())))
        metrics = compute_all_metrics(y_val, y_score, k=k_eval)

        top_domains = get_top_k_domains(df_train, args.top_k)
        feat_imp    = {
            "feature":    feat_names,
            "importance": model.get_feature_importance().tolist(),
        }
        import pandas as pd
        feat_imp_df = pd.DataFrame(feat_imp).sort_values(
            "importance", ascending=False
        ).reset_index(drop=True)

        fig_cm = plot_confusion_matrix(y_val, y_pred, title="CatBoost")
        fig_pr = plot_pr_curve(y_val, y_score, label="CatBoost")

        run_id = log_run_to_mlflow(
            params=params,
            metrics=metrics,
            figures={"confusion_matrix": fig_cm, "pr_curve": fig_pr},
            model=model,
            model_name="model",
            extra_artifacts={
                "feature_importance.csv": feat_imp_df,
                "top_domains":            top_domains,
                "feature_names":          feat_names,
            },
            model_flavor="catboost",
        )

    logger.info("Tree run complete — run_id=%s  PR-AUC=%.4f", run_id, metrics["pr_auc"])
    _log_metrics_table(metrics)


def run_tfidf(args, df_train, df_val, y_train, y_val) -> None:
    params = {
        "model_type":       "tfidf_lr",
        "target_user_id":   args.target_user_id,
        "split_strategy":   args.split_strategy,
        "max_features":     args.max_features,
        "C":                args.C,
        "train_size":       len(df_train),
        "val_size":         len(df_val),
        "n_positive_train": int(y_train.sum()),
        "n_negative_train": int(len(y_train) - y_train.sum()),
    }

    with mlflow.start_run(run_name="tfidf_lr"):
        pipeline = train_tfidf_lr(
            df_train, y_train,
            params={"max_features": args.max_features, "C": args.C},
        )

        y_score = pipeline.predict_proba(df_val)[:, 1]
        y_pred  = (y_score >= 0.5).astype(int)

        k_eval  = min(500, max(1, int(y_val.sum())))
        metrics = compute_all_metrics(y_val, y_score, k=k_eval)

        import pandas as pd
        import numpy as np
        feat_step  = pipeline.named_steps["features"]
        clf        = pipeline.named_steps["clf"]
        all_fnames = (
            list(feat_step.tfidf_.get_feature_names_out()) +
            list(feat_step.ohe_.get_feature_names_out()) +
            feat_step.num_cols
        )
        coef_df = pd.DataFrame({
            "feature":     all_fnames,
            "coefficient": clf.coef_[0],
        }).reindex(
            pd.Series(clf.coef_[0]).abs().nlargest(100).index
        ).reset_index(drop=True)

        fig_cm = plot_confusion_matrix(y_val, y_pred, title="TF-IDF+LR")
        fig_pr = plot_pr_curve(y_val, y_score, label="TF-IDF+LR")

        run_id = log_run_to_mlflow(
            params=params,
            metrics=metrics,
            figures={"confusion_matrix": fig_cm, "pr_curve": fig_pr},
            model=pipeline,
            model_name="model",
            extra_artifacts={"top_coefficients.csv": coef_df},
            model_flavor="sklearn",
        )

    logger.info("TF-IDF run complete — run_id=%s  PR-AUC=%.4f", run_id, metrics["pr_auc"])
    _log_metrics_table(metrics)


def run_siamese(args, df_train, df_val, y_train, y_val) -> None:
    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        logger.error("torch is not installed. Run: uv add torch")
        sys.exit(1)

    params = {
        "model_type":       "siamese",
        "target_user_id":   args.target_user_id,
        "split_strategy":   args.split_strategy,
        "emb_dim":          args.emb_dim,
        "hidden_dim":       256,
        "output_dim":       64,
        "epochs":           args.epochs,
        "batch_size":       args.batch_size,
        "lr":               args.lr,
        "neg_ratio":        args.neg_ratio,
        "max_pairs":        100_000,
        "max_sites":        20,
        "dropout":          0.2,
        "train_size":       len(df_train),
        "val_size":         len(df_val),
        "n_positive_train": int(y_train.sum()),
        "n_negative_train": int(len(y_train) - y_train.sum()),
    }

    siamese_model_params = {
        k: v for k, v in params.items()
        if k in {"emb_dim", "hidden_dim", "output_dim", "epochs",
                 "batch_size", "lr", "neg_ratio", "max_pairs", "max_sites", "dropout"}
    }

    with mlflow.start_run(run_name="siamese_encoder"):
        encoder, site_vocab, num_scaler = train_siamese(
            df_train, args.target_user_id,
            params=siamese_model_params, device=device,
        )

        df_target_train = df_train[df_train["user_id"] == args.target_user_id].copy()
        target_embs = encode_sessions(
            df_target_train, encoder, site_vocab, num_scaler,
            max_sites=siamese_model_params["max_sites"], device=device,
        )

        y_score = predict_siamese_scores(
            df_val, target_embs, encoder, site_vocab, num_scaler,
            max_sites=siamese_model_params["max_sites"], device=device, agg="max",
        )
        y_pred  = (y_score >= 0.5).astype(int)

        k_eval  = min(500, max(1, int(y_val.sum())))
        metrics = compute_all_metrics(y_val, y_score, k=k_eval)

        fig_cm = plot_confusion_matrix(y_val, y_pred, title="Siamese Encoder")
        fig_pr = plot_pr_curve(y_val, y_score, label="Siamese Encoder")

        run_id = log_run_to_mlflow(
            params=params,
            metrics=metrics,
            figures={"confusion_matrix": fig_cm, "pr_curve": fig_pr},
            model=encoder,
            model_name="model",
            extra_artifacts={
                "target_embeddings": target_embs,
                "site_vocab":        site_vocab,
                "numeric_scaler":    num_scaler,
            },
            model_flavor="pytorch",
        )

    logger.info("Siamese run complete — run_id=%s  PR-AUC=%.4f", run_id, metrics["pr_auc"])
    _log_metrics_table(metrics)


def _log_metrics_table(metrics: dict) -> None:
    width = 60
    logger.info("─" * width)
    for k, v in metrics.items():
        logger.info("  %-30s %s", k, f"{v:.4f}")
    logger.info("─" * width)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = build_parser().parse_args()

    # Resolve data path relative to the catch_joe/ project root
    data_path = Path(args.data_path)
    if not data_path.is_absolute():
        data_path = (REPO_ROOT / data_path).resolve()

    logger.info("Loading data from %s ...", data_path)
    df = load_sessions(data_path)
    validate_schema(df)
    df = create_target(df, args.target_user_id)
    df = extract_session_stats(df)

    # Quick sanity check on the chosen target user
    stats = get_user_stats(df)
    row   = stats[stats["user_id"] == args.target_user_id]
    if row.empty:
        logger.error("user_id=%d not found. Top users:\n%s",
                     args.target_user_id, stats.head(10).to_string(index=False))
        sys.exit(1)
    logger.info("Target user %d has %d sessions (%.2f%% of total)",
                args.target_user_id,
                int(row.iloc[0]["session_count"]),
                100 * int(row.iloc[0]["session_count"]) / len(df))

    # Split
    if args.split_strategy == "time":
        df_train, df_val = time_based_split(df, test_ratio=args.test_ratio)
    else:
        df_train, df_val = stratified_split(df, test_ratio=args.test_ratio)
    y_train = df_train["is_target_user"].values
    y_val   = df_val["is_target_user"].values

    logger.info("Train=%d  Val=%d  positives_train=%d  positives_val=%d",
                len(df_train), len(df_val), int(y_train.sum()), int(y_val.sum()))

    mlflow.set_tracking_uri(f"sqlite:///{REPO_ROOT}/mlflow.db")
    mlflow.set_experiment(args.experiment_name)

    dispatch = {"tree": run_tree, "tfidf": run_tfidf, "siamese": run_siamese}
    dispatch[args.approach](args, df_train, df_val, y_train, y_val)


if __name__ == "__main__":
    main()
