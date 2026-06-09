"""Data loading, schema validation, target creation, and split helpers."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit


# ── Public API ────────────────────────────────────────────────────────────────

def load_sessions(path: str | Path, has_user_id: bool = True) -> pd.DataFrame:
    """Load dataset.json (or verify.json) → flat session-level DataFrame.

    Columns produced:
        user_id (when has_user_id=True), browser, os, locale, gender,
        country, city, sites_visited (list[str]), site_lengths (list[int]),
        time, date, datetime
    """
    with open(path) as f:
        raw = json.load(f)

    records = []
    for r in raw:
        sites = r.get("sites", [])
        location = r.get("location", "/")
        parts = location.split("/", 1)
        record: dict = {
            "browser":       r["browser"],
            "os":            r["os"],
            "locale":        r["locale"],
            "gender":        r["gender"],
            "country":       parts[0],
            "city":          parts[1] if len(parts) > 1 else "",
            "sites_visited": [s["site"] for s in sites],
            "site_lengths":  [s["length"] for s in sites],
            "time":          r["time"],
            "date":          r["date"],
        }
        if has_user_id:
            record["user_id"] = r["user_id"]
        records.append(record)

    df = pd.DataFrame(records)
    df["datetime"] = pd.to_datetime(
        df["date"] + " " + df["time"], format="%Y-%m-%d %H:%M:%S", errors="coerce"
    )
    return df


def validate_schema(df: pd.DataFrame, require_user_id: bool = True) -> None:
    """Raise ValueError if required columns are missing."""
    expected = {
        "browser", "os", "locale", "gender", "country", "city",
        "sites_visited", "time", "date",
    }
    if require_user_id:
        expected.add("user_id")
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")


def create_target(df: pd.DataFrame, target_user_id: int | str) -> pd.DataFrame:
    """Append is_target_user binary column.

    user_id is NOT removed; it remains for split/evaluation purposes but
    must never be passed as a model feature.
    """
    df = df.copy()
    df["is_target_user"] = (df["user_id"] == target_user_id).astype(int)
    return df


def time_based_split(
    df: pd.DataFrame,
    test_ratio: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Sort by datetime; use earliest (1-test_ratio) fraction as training data.

    Limitation: a user active only in one period may appear only in train or
    only in validation. Use stratified_split when temporal ordering is irrelevant.
    """
    if "datetime" not in df.columns or df["datetime"].isna().all():
        raise ValueError("No valid datetime column; use stratified_split instead.")
    df_sorted = df.sort_values("datetime").reset_index(drop=True)
    split_idx = int(len(df_sorted) * (1.0 - test_ratio))
    return df_sorted.iloc[:split_idx].copy(), df_sorted.iloc[split_idx:].copy()


def stratified_split(
    df: pd.DataFrame,
    target_col: str = "is_target_user",
    test_ratio: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stratified train/val split preserving the positive-class ratio.

    Limitation: ignores temporal ordering; use only when datetime is unavailable
    or as an explicit ablation.
    """
    sss = StratifiedShuffleSplit(
        n_splits=1, test_size=test_ratio, random_state=random_state
    )
    idx_train, idx_val = next(sss.split(df, df[target_col]))
    return (
        df.iloc[idx_train].reset_index(drop=True),
        df.iloc[idx_val].reset_index(drop=True),
    )


def get_user_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Return per-user session counts sorted descending.

    Useful for choosing a well-represented TARGET_USER_ID.
    """
    if "user_id" not in df.columns:
        raise ValueError("DataFrame has no user_id column.")
    return (
        df.groupby("user_id")
        .size()
        .rename("session_count")
        .sort_values(ascending=False)
        .reset_index()
    )
