"""Feature engineering: session stats, top-K site indicators, TF-IDF pipelines."""
from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ── Column name constants ─────────────────────────────────────────────────────

METADATA_CAT_COLS: list[str] = ["gender", "locale", "country", "city", "os", "browser"]
METADATA_NUM_COLS: list[str] = [
    "session_length", "unique_site_count", "hour_of_day", "day_of_week_num"
]


# ── Session-level statistics ──────────────────────────────────────────────────

def extract_session_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived numeric columns to a session DataFrame.

    New columns:
        session_length    : total dwell time (sum of site_lengths, seconds)
        unique_site_count : number of distinct domains visited
        hour_of_day       : 0–23, parsed from time string "HH:MM:SS"
        day_of_week_num   : 0=Monday … 6=Sunday
    """
    df = df.copy()
    df["session_length"]    = df["site_lengths"].apply(sum)
    df["unique_site_count"] = df["sites_visited"].apply(lambda s: len(set(s)))
    df["hour_of_day"]       = df["time"].str.split(":").str[0].astype(int)
    df["day_of_week_num"]   = pd.to_datetime(df["date"]).dt.dayofweek
    return df


# ── Top-K domain indicators ───────────────────────────────────────────────────

def get_top_k_domains(df_train: pd.DataFrame, k: int) -> list[str]:
    """Return the K most frequent domains in the training set.

    Frequency = number of sessions that contain the domain (not raw visit count),
    so a domain visited 10 times in one session counts once.
    """
    counter: Counter = Counter()
    for sites in df_train["sites_visited"]:
        counter.update(set(sites))  # per-session unique to avoid frequency inflation
    return [domain for domain, _ in counter.most_common(k)]


def build_site_indicators(df: pd.DataFrame, top_domains: list[str]) -> pd.DataFrame:
    """Add a binary `site_<domain>` column for each domain in top_domains.

    Column value is 1 if that domain appears in the session's sites_visited.
    Always call this with top_domains derived from the TRAINING set only.

    Uses a pre-allocated uint8 numpy matrix to avoid the O(K) Series
    intermediate allocation that causes OOM for large K.
    """
    if not top_domains:
        return df.copy()
    domain_to_idx = {d: j for j, d in enumerate(top_domains)}
    df_reset = df.reset_index(drop=True)
    n, k = len(df_reset), len(top_domains)
    mat = np.zeros((n, k), dtype=np.uint8)
    for i, sites in enumerate(df_reset["sites_visited"]):
        for s in set(sites):
            j = domain_to_idx.get(s)
            if j is not None:
                mat[i, j] = 1
    cols = [f"site_{d}" for d in top_domains]
    ind_df = pd.DataFrame(mat, columns=cols)
    return pd.concat([df_reset, ind_df], axis=1)


def build_tree_features(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    top_k: int = 1000,
    cat_cols: list[str] | None = None,
    num_cols: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, list[int], list[str]]:
    """Build feature matrices for CatBoost / LightGBM.

    Returns:
        X_train, X_val         DataFrames: cat + num + site indicator columns
        cat_feature_indices    column indices of categorical features (for CatBoost Pool)
        feature_names          ordered list of all feature column names
    """
    cat_cols = cat_cols or METADATA_CAT_COLS
    num_cols = num_cols or METADATA_NUM_COLS

    top_domains = get_top_k_domains(df_train, top_k)
    df_tr = build_site_indicators(df_train, top_domains)
    df_va = build_site_indicators(df_val,   top_domains)

    site_cols     = [f"site_{d}" for d in top_domains]
    feature_cols  = cat_cols + num_cols + site_cols
    cat_feature_indices = list(range(len(cat_cols)))

    X_train = df_tr[feature_cols].copy()
    X_val   = df_va[feature_cols].copy()

    # Fill NaN: empty string for categoricals, 0 for numerics/indicators
    for col in cat_cols:
        X_train[col] = X_train[col].fillna("").astype(str)
        X_val[col]   = X_val[col].fillna("").astype(str)
    for col in num_cols + site_cols:
        X_train[col] = X_train[col].fillna(0)
        X_val[col]   = X_val[col].fillna(0)

    return X_train, X_val, cat_feature_indices, feature_cols


# ── TF-IDF pipeline ───────────────────────────────────────────────────────────

def sessions_to_text(df: pd.DataFrame) -> pd.Series:
    """Convert each session's sites_visited list to a space-joined string."""
    return df["sites_visited"].apply(lambda sites: " ".join(sites))


class SessionFeatureTransformer(BaseEstimator, TransformerMixin):
    """Sklearn-compatible transformer that combines:

    1. TF-IDF over space-joined sites_visited (sparse)
    2. One-hot encoded categorical metadata (sparse)
    3. Standard-scaled numeric metadata (dense → sparse)

    Returns a scipy CSR sparse matrix of shape (n_sessions, n_features).
    Input X must be a DataFrame with the expected columns.
    """

    def __init__(
        self,
        max_features: int = 5000,
        cat_cols: list[str] | None = None,
        num_cols: list[str] | None = None,
    ) -> None:
        self.max_features = max_features
        self.cat_cols = cat_cols or METADATA_CAT_COLS
        self.num_cols = num_cols or METADATA_NUM_COLS

    def fit(self, X: pd.DataFrame, y=None) -> "SessionFeatureTransformer":
        text = sessions_to_text(X)
        self.tfidf_ = TfidfVectorizer(
            analyzer="word",
            tokenizer=str.split,
            preprocessor=None,
            token_pattern=None,
            max_features=self.max_features,
            sublinear_tf=True,
        )
        self.tfidf_.fit(text)

        self.ohe_ = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
        self.ohe_.fit(X[self.cat_cols].fillna("").astype(str))

        self.scaler_ = StandardScaler()
        self.scaler_.fit(X[self.num_cols].fillna(0))
        return self

    def transform(self, X: pd.DataFrame):
        text     = sessions_to_text(X)
        X_tfidf  = self.tfidf_.transform(text)
        X_ohe    = self.ohe_.transform(X[self.cat_cols].fillna("").astype(str))
        X_num    = sp.csr_matrix(self.scaler_.transform(X[self.num_cols].fillna(0)))
        return sp.hstack([X_tfidf, X_ohe, X_num], format="csr")


def build_tfidf_lr_pipeline(
    max_features: int = 5000,
    cat_cols: list[str] | None = None,
    num_cols: list[str] | None = None,
    C: float = 1.0,
) -> Pipeline:
    """Build a complete sklearn Pipeline ready to fit on a raw session DataFrame.

    Steps:
        features : SessionFeatureTransformer → sparse CSR matrix
        clf      : LogisticRegression(class_weight='balanced')
    """
    return Pipeline([
        ("features", SessionFeatureTransformer(
            max_features=max_features,
            cat_cols=cat_cols or METADATA_CAT_COLS,
            num_cols=num_cols or METADATA_NUM_COLS,
        )),
        ("clf", LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            C=C,
            solver="lbfgs",
        )),
    ])
