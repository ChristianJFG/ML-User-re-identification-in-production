"""Production inference script — load a trained MLflow model and score sessions.

Usage examples:
    # Tree / TF-IDF model
    uv run python scripts/predict.py \\
        --run-id  <mlflow_run_id> \\
        --model-type  tree \\
        --data-path   data/raw/verify.json \\
        --output-path data/processed/predictions.csv

    # Siamese model
    uv run python scripts/predict.py \\
        --run-id  <mlflow_run_id> \\
        --model-type  siamese \\
        --data-path   data/raw/verify.json \\
        --target-user-id 0 \\
        --output-path data/processed/predictions.csv
"""
from __future__ import annotations

import argparse
import json
import logging
import pickle
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

# Allow running from the repo root or from catch_joe/
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH  = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from catch_joe.data import load_sessions, validate_schema
from catch_joe.features import extract_session_stats, build_tree_features

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


# ── CLI ───────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run inference using a trained MLflow model."
    )
    p.add_argument("--run-id", required=True,
                   help="MLflow run ID that contains the trained model and artifacts.")
    p.add_argument("--model-type", required=True, choices=["tree", "tfidf", "siamese"],
                   help="Type of model stored in the run.")
    p.add_argument("--data-path", required=True,
                   help="Path to JSON file with sessions to score.")
    p.add_argument("--target-user-id", type=int, default=None,
                   help="[siamese only] Target user ID whose embeddings to load "
                        "from the run artifacts.")
    p.add_argument("--output-path", default="predictions.csv",
                   help="Output CSV path (default: predictions.csv).")
    p.add_argument("--has-user-id", action="store_true",
                   help="Flag if the data file contains a user_id column.")
    return p


# ── Artifact download helpers ────────────────────────────────────────────────

def _download_artifact(run_id: str, artifact_name: str, dst_dir: Path) -> Path:
    """Download a single artifact from an MLflow run to dst_dir."""
    import mlflow
    client = mlflow.tracking.MlflowClient()
    local  = client.download_artifacts(run_id, artifact_name, str(dst_dir))
    return Path(local)


# ── Per-approach predict functions ───────────────────────────────────────────

def predict_tree(run_id: str, df: pd.DataFrame) -> np.ndarray:
    """Load CatBoost model + top_domains / feature_names and score df."""
    import mlflow.catboost

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        td_path = _download_artifact(run_id, "top_domains.json", tmp)
        fn_path = _download_artifact(run_id, "feature_names.json", tmp)

        top_domains  = json.loads(td_path.read_text())
        feature_names = json.loads(fn_path.read_text())

    # Build feature matrix using the same top_domains from the training run
    from catch_joe.features import (
        METADATA_CAT_COLS, METADATA_NUM_COLS, build_site_indicators
    )
    df_feat = build_site_indicators(df, top_domains)
    for col in METADATA_CAT_COLS:
        df_feat[col] = df_feat[col].fillna("").astype(str)
    for col in METADATA_NUM_COLS:
        df_feat[col] = df_feat[col].fillna(0)

    # Ensure all feature columns exist (fill missing site indicators with 0)
    for col in feature_names:
        if col not in df_feat.columns:
            df_feat[col] = 0
    X = df_feat[feature_names]

    model_uri = f"runs:/{run_id}/model"
    model = mlflow.catboost.load_model(model_uri)
    return model.predict_proba(X)[:, 1]


def predict_tfidf(run_id: str, df: pd.DataFrame) -> np.ndarray:
    """Load sklearn Pipeline and score df."""
    import mlflow.sklearn
    model_uri = f"runs:/{run_id}/model"
    pipeline  = mlflow.sklearn.load_model(model_uri)
    return pipeline.predict_proba(df)[:, 1]


def predict_siamese(run_id: str, df: pd.DataFrame) -> np.ndarray:
    """Load encoder + vocab + scaler + target embeddings and score df."""
    import mlflow.pytorch
    from catch_joe.modeling import encode_sessions, predict_siamese_scores

    model_uri = f"runs:/{run_id}/model"
    encoder   = mlflow.pytorch.load_model(model_uri)
    encoder.eval()

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)

        vocab_path   = _download_artifact(run_id, "site_vocab.json", tmp)
        scaler_path  = _download_artifact(run_id, "numeric_scaler.pkl", tmp)
        emb_path     = _download_artifact(run_id, "target_embeddings.npy", tmp)

        site_vocab        = json.loads(vocab_path.read_text())
        with open(scaler_path, "rb") as f:
            numeric_scaler = pickle.load(f)
        target_embeddings = np.load(str(emb_path))

    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        device = "cpu"

    return predict_siamese_scores(
        df, target_embeddings, encoder, site_vocab, numeric_scaler,
        max_sites=20, device=device, agg="max",
    )


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = build_parser().parse_args()

    import mlflow
    mlflow.set_tracking_uri(f"sqlite:///{REPO_ROOT}/mlflow.db")

    data_path = Path(args.data_path)
    if not data_path.is_absolute():
        data_path = (REPO_ROOT / data_path).resolve()

    logger.info("Loading sessions from %s ...", data_path)
    df = load_sessions(data_path, has_user_id=args.has_user_id)
    validate_schema(df, require_user_id=False)
    df = extract_session_stats(df)

    logger.info("Loaded %d sessions", len(df))

    logger.info("Running inference with model_type=%s  run_id=%s",
                args.model_type, args.run_id)

    dispatch = {
        "tree":    predict_tree,
        "tfidf":   predict_tfidf,
        "siamese": predict_siamese,
    }
    scores = dispatch[args.model_type](args.run_id, df)

    output = df.copy()
    output["score"] = scores
    output["predicted_target"] = (scores >= 0.5).astype(int)

    # Drop list columns (not CSV-serialisable)
    for col in ["sites_visited", "site_lengths"]:
        if col in output.columns:
            output = output.drop(columns=[col])

    out_path = Path(args.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(out_path, index=False)
    logger.info("Predictions saved to %s  (n=%d, n_target=%d)",
                out_path, len(output), int(output["predicted_target"].sum()))


if __name__ == "__main__":
    main()
