"""Promote the best MLflow run to the Model Registry under the @production alias.

Queries the SQLite tracking store for the highest-scoring finished run in the
experiment, registers its model artifact, tags it with metadata, and sets the
@production alias so predict.py can load it without a run ID.

Usage:
    uv run python scripts/promote.py
    uv run python scripts/promote.py --experiment-name catch_joe_detection
    uv run python scripts/promote.py --model-name catch_joe_detector --metric pr_auc
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import mlflow

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH  = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

TRACKING_URI = f"sqlite:///{REPO_ROOT}/mlflow.db"
DEFAULT_MODEL_NAME = "catch_joe_detector"


# ── CLI ───────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Register the best run from an MLflow experiment and promote to @production."
    )
    p.add_argument("--experiment-name", default="catch_joe_detection",
                   help="MLflow experiment to search (default: catch_joe_detection).")
    p.add_argument("--model-name", default=DEFAULT_MODEL_NAME,
                   help=f"Registered model name (default: {DEFAULT_MODEL_NAME}).")
    p.add_argument("--metric", default="pr_auc",
                   help="Metric used to rank runs — higher is better (default: pr_auc).")
    return p


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = build_parser().parse_args()

    mlflow.set_tracking_uri(TRACKING_URI)
    client = mlflow.tracking.MlflowClient()

    # ── 1. Find best finished run ─────────────────────────────────────────────
    experiment = client.get_experiment_by_name(args.experiment_name)
    if experiment is None:
        logger.error(
            "Experiment '%s' not found in %s. "
            "Run train.py first or pass --experiment-name.",
            args.experiment_name, TRACKING_URI,
        )
        sys.exit(1)

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="attributes.status = 'FINISHED'",
        order_by=[f"metrics.{args.metric} DESC"],
        max_results=1,
    )
    if not runs:
        logger.error(
            "No finished runs found in experiment '%s'.", args.experiment_name
        )
        sys.exit(1)

    best       = runs[0]
    run_id     = best.info.run_id
    metric_val = best.data.metrics.get(args.metric, float("nan"))
    model_type = best.data.params.get("model_type", "unknown")

    logger.info(
        "Best run  run_id=%s  %s=%.4f  model_type=%s",
        run_id, args.metric, metric_val, model_type,
    )

    # ── 2. Register the model artifact ───────────────────────────────────────
    model_uri = f"runs:/{run_id}/model"
    mv = mlflow.register_model(model_uri, args.model_name)
    logger.info("Registered '%s' version %s", args.model_name, mv.version)

    # ── 3. Tag the version with metadata predict.py needs ────────────────────
    client.set_model_version_tag(args.model_name, mv.version, "model_type", model_type)
    client.set_model_version_tag(args.model_name, mv.version, "run_id",     run_id)
    client.set_model_version_tag(args.model_name, mv.version, args.metric,  str(round(metric_val, 4)))

    # ── 4. Promote to @production ─────────────────────────────────────────────
    client.set_registered_model_alias(args.model_name, "production", mv.version)
    logger.info(
        "Promoted '%s' v%s to @production  (%s=%.4f)",
        args.model_name, mv.version, args.metric, metric_val,
    )


if __name__ == "__main__":
    main()
