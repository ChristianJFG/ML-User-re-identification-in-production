"""Reusable training code for all three detection approaches."""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from catch_joe.features import build_tfidf_lr_pipeline, METADATA_NUM_COLS

logger = logging.getLogger(__name__)

# Numeric features consumed by the Siamese model
SIAMESE_NUM_COLS: list[str] = [
    "session_length", "unique_site_count", "hour_of_day", "day_of_week_num"
]

# ── Optional heavy deps ───────────────────────────────────────────────────────

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


# ── CatBoost ──────────────────────────────────────────────────────────────────

def train_catboost(
    X_train: pd.DataFrame,
    y_train: np.ndarray | pd.Series,
    X_val: pd.DataFrame,
    y_val: np.ndarray | pd.Series,
    cat_feature_indices: list[int],
    params: dict[str, Any] | None = None,
) -> Any:
    """Train a CatBoostClassifier with automatic class balancing.

    X_train / X_val must be DataFrames; categorical columns stay as strings —
    CatBoost handles them natively via cat_feature_indices.

    Returns the fitted CatBoostClassifier.
    """
    try:
        import catboost as cb
    except ImportError as exc:
        raise ImportError("catboost is not installed. Run: uv add catboost") from exc

    default_params: dict[str, Any] = {
        "iterations":            500,
        "learning_rate":         0.05,
        "depth":                 6,
        "auto_class_weights":    "Balanced",
        "eval_metric":           "AUC",
        "random_seed":           42,
        "verbose":               100,
        "early_stopping_rounds": 50,
    }
    if params:
        default_params.update(params)

    train_pool = cb.Pool(X_train, label=y_train, cat_features=cat_feature_indices)
    val_pool   = cb.Pool(X_val,   label=y_val,   cat_features=cat_feature_indices)

    model = cb.CatBoostClassifier(**default_params)
    model.fit(train_pool, eval_set=val_pool)
    return model


# ── LightGBM ──────────────────────────────────────────────────────────────────

def train_lightgbm(
    X_train: pd.DataFrame,
    y_train: np.ndarray | pd.Series,
    X_val: pd.DataFrame,
    y_val: np.ndarray | pd.Series,
    cat_feature_names: list[str],
    params: dict[str, Any] | None = None,
) -> Any:
    """Train a LightGBM binary classifier.

    cat_feature_names: column names that should be treated as categorical.
    Returns the fitted lgb.Booster.
    """
    try:
        import lightgbm as lgb
    except ImportError as exc:
        raise ImportError("lightgbm is not installed. Run: uv add lightgbm") from exc

    default_params: dict[str, Any] = {
        "objective":         "binary",
        "metric":            "auc",
        "learning_rate":     0.05,
        "num_leaves":        63,
        "min_child_samples": 20,
        "is_unbalance":      True,
        "random_state":      42,
        "verbosity":         -1,
    }
    if params:
        default_params.update(params)

    X_tr = X_train.copy()
    X_va = X_val.copy()
    for col in cat_feature_names:
        X_tr[col] = X_tr[col].astype("category")
        X_va[col] = X_va[col].astype("category")

    dtrain = lgb.Dataset(X_tr, label=y_train, free_raw_data=False)
    dval   = lgb.Dataset(X_va, label=y_val, reference=dtrain, free_raw_data=False)

    callbacks = [lgb.early_stopping(50, verbose=False), lgb.log_evaluation(100)]
    model = lgb.train(
        default_params,
        dtrain,
        num_boost_round=500,
        valid_sets=[dval],
        callbacks=callbacks,
    )
    return model


# ── TF-IDF + Logistic Regression ─────────────────────────────────────────────

def train_tfidf_lr(
    df_train: pd.DataFrame,
    y_train: np.ndarray | pd.Series,
    params: dict[str, Any] | None = None,
) -> Any:
    """Fit the TF-IDF + metadata + LogisticRegression sklearn Pipeline.

    df_train must contain METADATA_CAT_COLS, METADATA_NUM_COLS, and sites_visited.
    Returns the fitted Pipeline (fully self-contained for inference).
    """
    p = params or {}
    pipeline = build_tfidf_lr_pipeline(
        max_features=p.get("max_features", 5000),
        C=p.get("C", 1.0),
    )
    pipeline.fit(df_train, y_train)
    return pipeline


# ── Siamese site vocabulary ───────────────────────────────────────────────────

def build_site_vocab(df_train: pd.DataFrame) -> dict[str, int]:
    """Build domain → integer index mapping from training sessions.

    Index 0 is reserved for padding; domain indices start at 1.
    """
    all_domains: set[str] = set()
    for sites in df_train["sites_visited"]:
        all_domains.update(sites)
    return {domain: i + 1 for i, domain in enumerate(sorted(all_domains))}


# ── Siamese pair sampling ─────────────────────────────────────────────────────

def create_session_pairs(
    df: pd.DataFrame,
    target_user_id: int | str,
    neg_ratio: int = 4,
    max_pairs: int = 100_000,
    random_state: int = 42,
) -> list[tuple[int, int, int]]:
    """Sample (index_i, index_j, label) pairs for contrastive training.

    Positive pairs (label=1): two sessions from target_user_id.
    Negative pairs (label=0): one session from target_user_id + one from another user.
    """
    rng = np.random.default_rng(random_state)
    df_r = df.reset_index(drop=True)

    target_idx = df_r.index[df_r["user_id"] == target_user_id].tolist()
    other_idx  = df_r.index[df_r["user_id"] != target_user_id].tolist()

    if len(target_idx) < 2:
        raise ValueError(
            f"Target user {target_user_id} has fewer than 2 sessions in this split. "
            "Choose a user with more sessions or use a larger training set."
        )

    n_pos_max = min(
        len(target_idx) * (len(target_idx) - 1) // 2,
        max_pairs // (1 + neg_ratio),
    )

    # Positive pairs (deduplicated)
    pos_set: set[tuple[int, int]] = set()
    attempts = 0
    while len(pos_set) < n_pos_max and attempts < n_pos_max * 10:
        i, j = rng.choice(target_idx, size=2, replace=False)
        pos_set.add((min(int(i), int(j)), max(int(i), int(j))))
        attempts += 1
    pos_pairs = [(a, b, 1) for a, b in pos_set]

    # Negative pairs
    n_neg = min(len(pos_pairs) * neg_ratio, max_pairs - len(pos_pairs))
    neg_seen: set[tuple[int, int]] = set()
    neg_pairs: list[tuple[int, int, int]] = []
    attempts = 0
    while len(neg_pairs) < n_neg and attempts < n_neg * 5:
        ti = int(rng.choice(target_idx))
        oj = int(rng.choice(other_idx))
        key = (ti, oj)
        if key not in neg_seen:
            neg_seen.add(key)
            neg_pairs.append((ti, oj, 0))
        attempts += 1

    all_pairs = pos_pairs + neg_pairs
    rng.shuffle(all_pairs)
    return all_pairs


# ── PyTorch Session Encoder & Dataset ────────────────────────────────────────

if _TORCH_AVAILABLE:
    class SessionEncoder(nn.Module):
        """Encodes a browsing session into a fixed-dimension embedding.

        Architecture:
            site_embedding (vocab+1, emb_dim) → mean-pool over visited sites
            → concat numeric features
            → MLP (hidden_dim → hidden_dim//2 → output_dim)
        """

        def __init__(
            self,
            vocab_size: int,
            emb_dim: int = 64,
            num_numeric: int = 4,
            hidden_dim: int = 256,
            output_dim: int = 64,
            dropout: float = 0.2,
        ) -> None:
            super().__init__()
            self.site_emb = nn.Embedding(vocab_size + 1, emb_dim, padding_idx=0)
            self.mlp = nn.Sequential(
                nn.Linear(emb_dim + num_numeric, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, output_dim),
            )

        def forward(self, site_idx: "torch.Tensor", numeric: "torch.Tensor") -> "torch.Tensor":
            # site_idx : (B, max_sites)  — 0 = padding
            emb    = self.site_emb(site_idx)                          # (B, S, emb_dim)
            mask   = (site_idx != 0).unsqueeze(-1).float()            # (B, S, 1)
            pooled = (emb * mask).sum(1) / mask.sum(1).clamp(min=1)   # (B, emb_dim)
            x = torch.cat([pooled, numeric], dim=1)
            return self.mlp(x)                                        # (B, output_dim)

    class SiameseDataset(Dataset):
        """Pairs of encoded sessions for contrastive training."""

        def __init__(
            self,
            pairs: list[tuple[int, int, int]],
            df: pd.DataFrame,
            site_vocab: dict[str, int],
            numeric_scaler: Any,
            max_sites: int = 20,
        ) -> None:
            self.pairs          = pairs
            self.df             = df.reset_index(drop=True)
            self.site_vocab     = site_vocab
            self.max_sites      = max_sites
            self.numeric_matrix = numeric_scaler.transform(
                self.df[SIAMESE_NUM_COLS].fillna(0).values
            ).astype(np.float32)

        def _encode_sites(self, row_idx: int) -> list[int]:
            sites   = self.df.iloc[row_idx]["sites_visited"][: self.max_sites]
            indices = [self.site_vocab.get(s, 0) for s in sites]
            indices += [0] * (self.max_sites - len(indices))
            return indices

        def __len__(self) -> int:
            return len(self.pairs)

        def __getitem__(self, idx: int):
            i, j, label = self.pairs[idx]
            site_i = torch.tensor(self._encode_sites(i), dtype=torch.long)
            site_j = torch.tensor(self._encode_sites(j), dtype=torch.long)
            num_i  = torch.tensor(self.numeric_matrix[i], dtype=torch.float32)
            num_j  = torch.tensor(self.numeric_matrix[j], dtype=torch.float32)
            return site_i, num_i, site_j, num_j, torch.tensor(float(label))

else:  # Stubs when PyTorch is not installed

    class SessionEncoder:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            raise ImportError("torch is not installed. Run: uv add torch")

    class SiameseDataset:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            raise ImportError("torch is not installed. Run: uv add torch")


# ── Siamese training ──────────────────────────────────────────────────────────

def train_siamese(
    df_train: pd.DataFrame,
    target_user_id: int | str,
    params: dict[str, Any] | None = None,
    device: str | None = None,
) -> tuple[Any, dict[str, int], Any]:
    """Train the Siamese session encoder.

    df_train must already contain SIAMESE_NUM_COLS (call extract_session_stats first).

    Returns:
        encoder        : trained SessionEncoder (nn.Module)
        site_vocab     : domain → index mapping used by the encoder
        numeric_scaler : fitted StandardScaler for numeric features
    """
    if not _TORCH_AVAILABLE:
        raise ImportError("torch is not installed. Run: uv add torch")

    from sklearn.preprocessing import StandardScaler

    p = params or {}
    emb_dim    = p.get("emb_dim",    64)
    hidden_dim = p.get("hidden_dim", 256)
    output_dim = p.get("output_dim", 64)
    epochs     = p.get("epochs",     20)
    batch_size = p.get("batch_size", 256)
    lr         = p.get("lr",         1e-3)
    neg_ratio  = p.get("neg_ratio",  4)
    max_pairs  = p.get("max_pairs",  100_000)
    max_sites  = p.get("max_sites",  20)
    dropout    = p.get("dropout",    0.2)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)

    site_vocab = build_site_vocab(df_train)
    vocab_size = len(site_vocab)

    numeric_scaler = StandardScaler()
    numeric_scaler.fit(df_train[SIAMESE_NUM_COLS].fillna(0).values)

    pairs = create_session_pairs(
        df_train, target_user_id, neg_ratio=neg_ratio, max_pairs=max_pairs
    )
    logger.info("Created %d pairs (%d pos, %d neg)",
                len(pairs),
                sum(1 for _, _, l in pairs if l == 1),
                sum(1 for _, _, l in pairs if l == 0))

    dataset = SiameseDataset(pairs, df_train, site_vocab, numeric_scaler, max_sites=max_sites)
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    encoder = SessionEncoder(
        vocab_size=vocab_size,
        emb_dim=emb_dim,
        num_numeric=len(SIAMESE_NUM_COLS),
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        dropout=dropout,
    ).to(dev)

    optimizer = torch.optim.Adam(encoder.parameters(), lr=lr)
    criterion = torch.nn.BCEWithLogitsLoss()

    encoder.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for site_i, num_i, site_j, num_j, labels in loader:
            site_i, num_i = site_i.to(dev), num_i.to(dev)
            site_j, num_j = site_j.to(dev), num_j.to(dev)
            labels = labels.to(dev)

            emb_i = encoder(site_i, num_i)
            emb_j = encoder(site_j, num_j)

            # Cosine similarity scaled to logit space for BCEWithLogitsLoss
            sim  = F.cosine_similarity(emb_i, emb_j) * 5.0
            loss = criterion(sim, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(labels)

        avg_loss = total_loss / len(dataset)
        if (epoch + 1) % 5 == 0 or epoch == 0:
            logger.info("Epoch %d/%d — loss: %.4f", epoch + 1, epochs, avg_loss)

    encoder.eval()
    return encoder, site_vocab, numeric_scaler


# ── Session encoding & inference ─────────────────────────────────────────────

def encode_sessions(
    df: pd.DataFrame,
    encoder: Any,
    site_vocab: dict[str, int],
    numeric_scaler: Any,
    max_sites: int = 20,
    device: str | None = None,
    batch_size: int = 512,
) -> np.ndarray:
    """Encode all sessions in df into fixed-dimension embedding vectors.

    Returns np.ndarray of shape (n_sessions, output_dim).
    """
    if not _TORCH_AVAILABLE:
        raise ImportError("torch is not installed. Run: uv add torch")

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)

    df_r = df.reset_index(drop=True)
    numeric_matrix = numeric_scaler.transform(
        df_r[SIAMESE_NUM_COLS].fillna(0).values
    ).astype(np.float32)

    all_embs: list[np.ndarray] = []
    encoder.eval()
    with torch.no_grad():
        for start in range(0, len(df_r), batch_size):
            end = min(start + batch_size, len(df_r))
            batch_sites = []
            for i in range(start, end):
                sites   = df_r.iloc[i]["sites_visited"][: max_sites]
                indices = [site_vocab.get(s, 0) for s in sites]
                indices += [0] * (max_sites - len(indices))
                batch_sites.append(indices)
            site_t = torch.tensor(batch_sites, dtype=torch.long).to(dev)
            num_t  = torch.tensor(numeric_matrix[start:end], dtype=torch.float32).to(dev)
            emb    = encoder(site_t, num_t)
            all_embs.append(emb.cpu().numpy())

    return np.vstack(all_embs)


def predict_siamese_scores(
    df_new: pd.DataFrame,
    target_embeddings: np.ndarray,
    encoder: Any,
    site_vocab: dict[str, int],
    numeric_scaler: Any,
    max_sites: int = 20,
    device: str | None = None,
    agg: str = "max",
) -> np.ndarray:
    """Score new sessions against the target user's historical embeddings.

    For each new session, computes cosine similarity against every stored
    target-user embedding and aggregates with `agg` ('max' or 'mean').
    Higher score → more likely to be the target user.

    Returns np.ndarray of shape (n_sessions,), values in [0, 1].
    """
    new_embs = encode_sessions(
        df_new, encoder, site_vocab, numeric_scaler,
        max_sites=max_sites, device=device
    )

    def _normalize(x: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(x, axis=1, keepdims=True)
        return x / np.where(norms == 0, 1.0, norms)

    new_norm    = _normalize(new_embs)           # (N, D)
    target_norm = _normalize(target_embeddings)  # (T, D)
    sims        = new_norm @ target_norm.T        # (N, T)

    if agg == "max":
        scores = sims.max(axis=1)
    elif agg == "mean":
        scores = sims.mean(axis=1)
    else:
        raise ValueError(f"Unknown aggregation '{agg}'. Choose 'max' or 'mean'.")

    return np.clip(scores, 0.0, 1.0)
