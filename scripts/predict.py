"""Production inference script — load a trained MLflow model and score sessions.

By default loads the @production model from the registry (run promote.py first).
Pass --run-id and --model-type to target a specific run instead.

Usage examples:
    # Use the registered @production model (standalone — no run-id needed)
    uv run python scripts/predict.py --data-path data/raw/verify.json

    # Target a specific run
    uv run python scripts/predict.py \\
        --run-id     <mlflow_run_id> \\
        --model-type tree \\
        --data-path  data/raw/verify.json
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

REGISTRY_MODEL_NAME = "catch_joe_detector"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run inference using a trained MLflow model."
    )
    p.add_argument("--run-id", default=None,
                   help="MLflow run ID. Omit to load the @production registered model.")
    p.add_argument("--model-type", default=None, choices=["tree", "tfidf", "siamese"],
                   help="Model type. Auto-detected from registry when --run-id is omitted.")
    p.add_argument("--data-path", required=True,
                   help="Path to JSON file with sessions to score.")
    p.add_argument("--target-user-id", type=int, default=None,
                   help="[siamese only] Target user ID whose embeddings to load "
                        "from the run artifacts.")
    p.add_argument("--model-name", default=REGISTRY_MODEL_NAME,
                   help=f"Registered model name (default: {REGISTRY_MODEL_NAME}).")
    p.add_argument("--output-path", default="result.csv",
                   help="Output CSV path (default: result.csv).")
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

def _resolve_run_id_and_model_type(args) -> tuple[str, str]:
    """Return (run_id, model_type), resolving from the registry when not supplied."""
    import mlflow
    if args.run_id and args.model_type:
        return args.run_id, args.model_type

    client = mlflow.tracking.MlflowClient()
    try:
        mv = client.get_model_version_by_alias(args.model_name, "production")
    except Exception as exc:
        logger.error(
            "Could not load @production from registry '%s': %s. "
            "Run promote.py first, or pass --run-id and --model-type.",
            args.model_name, exc,
        )
        sys.exit(1)

    run_id     = mv.tags.get("run_id") or mv.run_id
    model_type = mv.tags.get("model_type")
    if not model_type:
        logger.error(
            "Registry version %s has no 'model_type' tag. Re-run promote.py.",
            mv.version,
        )
        sys.exit(1)

    logger.info(
        "Loaded @production from registry: model=%s  version=%s  model_type=%s  run_id=%s",
        args.model_name, mv.version, model_type, run_id,
    )
    return run_id, model_type


def main() -> None:
    args = build_parser().parse_args()

    import mlflow
    mlflow.set_tracking_uri(f"sqlite:///{REPO_ROOT}/mlflow.db")

    run_id, model_type = _resolve_run_id_and_model_type(args)

    data_path = Path(args.data_path)
    if not data_path.is_absolute():
        data_path = (REPO_ROOT / data_path).resolve()

    logger.info("Loading sessions from %s ...", data_path)
    df = load_sessions(data_path, has_user_id=args.has_user_id)
    validate_schema(df, require_user_id=False)
    df = extract_session_stats(df)

    logger.info("Loaded %d sessions", len(df))
    logger.info("Running inference with model_type=%s  run_id=%s", model_type, run_id)

    dispatch = {
        "tree":       predict_tree,
        "tfidf":      predict_tfidf,
        "tfidf_lr":   predict_tfidf,
        "siamese":    predict_siamese,
        "catboost":   predict_tree,
    }
    if model_type not in dispatch:
        logger.error("Unknown model_type '%s'. Expected one of: %s", model_type, list(dispatch))
        sys.exit(1)

    scores = dispatch[model_type](run_id, df)
    labels = (scores >= 0.5).astype(int)

    out_path = Path(args.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"predicted_label": labels}).to_csv(out_path, index=False)
    logger.info(
        "result.csv saved to %s  (n=%d, n_joe=%d, n_not_joe=%d)",
        out_path, len(labels), int((labels == 0).sum()), int((labels == 1).sum()),
    )


if __name__ == "__main__":
    main()
