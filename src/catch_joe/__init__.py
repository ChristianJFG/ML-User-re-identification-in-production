"""catch_joe: Target-user detection from web session data."""

from catch_joe.data import (
    create_target,
    get_user_stats,
    load_sessions,
    stratified_split,
    time_based_split,
    validate_schema,
)
from catch_joe.evaluation import (
    compare_models,
    compute_all_metrics,
    log_run_to_mlflow,
    plot_confusion_matrix,
    plot_pr_curve,
    precision_at_k,
    recall_at_k,
)
from catch_joe.features import (
    METADATA_CAT_COLS,
    METADATA_NUM_COLS,
    build_tfidf_lr_pipeline,
    build_tree_features,
    extract_session_stats,
    get_top_k_domains,
    sessions_to_text,
)
from catch_joe.modeling import (
    SIAMESE_NUM_COLS,
    build_site_vocab,
    create_session_pairs,
    encode_sessions,
    predict_siamese_scores,
    train_catboost,
    train_lightgbm,
    train_siamese,
    train_tfidf_lr,
)

__all__ = [
    # data
    "load_sessions", "validate_schema", "create_target",
    "time_based_split", "stratified_split", "get_user_stats",
    # features
    "extract_session_stats", "get_top_k_domains", "build_tree_features",
    "sessions_to_text", "build_tfidf_lr_pipeline",
    "METADATA_CAT_COLS", "METADATA_NUM_COLS",
    # evaluation
    "compute_all_metrics", "precision_at_k", "recall_at_k",
    "plot_confusion_matrix", "plot_pr_curve", "compare_models", "log_run_to_mlflow",
    # modeling
    "train_catboost", "train_lightgbm", "train_tfidf_lr",
    "build_site_vocab", "create_session_pairs",
    "train_siamese", "encode_sessions", "predict_siamese_scores",
    "SIAMESE_NUM_COLS",
]
