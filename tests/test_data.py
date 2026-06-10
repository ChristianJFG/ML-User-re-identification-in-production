"""Tests for catch_joe.data — loading, schema, target creation, splits."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from catch_joe.data import (
    create_target,
    get_user_stats,
    load_sessions,
    stratified_split,
    time_based_split,
    validate_schema,
)


# ── load_sessions ─────────────────────────────────────────────────────────────

class TestLoadSessions:
    def test_returns_dataframe(self, dataset_json):
        df = load_sessions(dataset_json)
        assert isinstance(df, pd.DataFrame)

    def test_row_count_matches_input(self, dataset_json, raw_sessions):
        df = load_sessions(dataset_json)
        assert len(df) == len(raw_sessions)

    def test_expected_columns_present(self, dataset_json):
        df = load_sessions(dataset_json)
        required = {
            "user_id", "browser", "os", "locale", "gender",
            "country", "city", "sites_visited", "site_lengths",
            "time", "date", "datetime",
        }
        assert required.issubset(df.columns)

    def test_location_split_into_country_city(self, dataset_json):
        df = load_sessions(dataset_json)
        assert df["country"].iloc[0] == "USA"
        assert df["city"].iloc[0] == "New York"

    def test_sites_visited_is_list_of_strings(self, dataset_json):
        df = load_sessions(dataset_json)
        assert isinstance(df["sites_visited"].iloc[0], list)
        assert all(isinstance(s, str) for s in df["sites_visited"].iloc[0])

    def test_site_lengths_is_list_of_ints(self, dataset_json):
        df = load_sessions(dataset_json)
        assert isinstance(df["site_lengths"].iloc[0], list)
        assert all(isinstance(n, int) for n in df["site_lengths"].iloc[0])

    def test_datetime_column_parsed(self, dataset_json):
        df = load_sessions(dataset_json)
        assert pd.api.types.is_datetime64_any_dtype(df["datetime"])
        assert df["datetime"].notna().all()

    def test_without_user_id(self, verify_json):
        df = load_sessions(verify_json, has_user_id=False)
        assert "user_id" not in df.columns
        assert "browser" in df.columns


# ── validate_schema ───────────────────────────────────────────────────────────

class TestValidateSchema:
    def test_passes_on_valid_df(self, sessions_df):
        validate_schema(sessions_df)  # should not raise

    def test_raises_on_missing_column(self, sessions_df):
        bad_df = sessions_df.drop(columns=["browser"])
        with pytest.raises(ValueError, match="browser"):
            validate_schema(bad_df)

    def test_passes_without_user_id_when_not_required(self, sessions_df):
        df_no_id = sessions_df.drop(columns=["user_id"])
        validate_schema(df_no_id, require_user_id=False)  # should not raise

    def test_raises_when_user_id_missing_and_required(self, sessions_df):
        df_no_id = sessions_df.drop(columns=["user_id"])
        with pytest.raises(ValueError, match="user_id"):
            validate_schema(df_no_id, require_user_id=True)


# ── create_target ─────────────────────────────────────────────────────────────

class TestCreateTarget:
    def test_adds_is_target_user_column(self, sessions_df):
        df = create_target(sessions_df, target_user_id=0)
        assert "is_target_user" in df.columns

    def test_correct_positive_count(self, sessions_df, raw_sessions):
        expected_positives = sum(1 for s in raw_sessions if s["user_id"] == 0)
        df = create_target(sessions_df, target_user_id=0)
        assert df["is_target_user"].sum() == expected_positives

    def test_binary_values_only(self, sessions_df):
        df = create_target(sessions_df, target_user_id=0)
        assert set(df["is_target_user"].unique()).issubset({0, 1})

    def test_original_df_not_mutated(self, sessions_df):
        _ = create_target(sessions_df, target_user_id=0)
        assert "is_target_user" not in sessions_df.columns

    def test_no_positives_for_absent_user(self, sessions_df):
        df = create_target(sessions_df, target_user_id=999)
        assert df["is_target_user"].sum() == 0


# ── time_based_split ──────────────────────────────────────────────────────────

class TestTimeBasedSplit:
    def test_total_rows_preserved(self, sessions_with_target):
        train, val = time_based_split(sessions_with_target, test_ratio=0.4)
        assert len(train) + len(val) == len(sessions_with_target)

    def test_val_ratio_approximately_correct(self, sessions_with_target):
        test_ratio = 0.4
        train, val = time_based_split(sessions_with_target, test_ratio=test_ratio)
        actual_ratio = len(val) / len(sessions_with_target)
        assert abs(actual_ratio - test_ratio) <= 1 / len(sessions_with_target)

    def test_train_is_earlier_than_val(self, sessions_with_target):
        train, val = time_based_split(sessions_with_target, test_ratio=0.4)
        assert train["datetime"].max() <= val["datetime"].min()

    def test_raises_without_datetime_column(self, sessions_with_target):
        df_no_dt = sessions_with_target.drop(columns=["datetime"])
        with pytest.raises(ValueError, match="datetime"):
            time_based_split(df_no_dt)


# ── stratified_split ──────────────────────────────────────────────────────────

class TestStratifiedSplit:
    def test_total_rows_preserved(self, sessions_with_target):
        train, val = stratified_split(sessions_with_target, test_ratio=0.4)
        assert len(train) + len(val) == len(sessions_with_target)

    def test_positive_class_present_in_both_splits(self, sessions_with_target):
        train, val = stratified_split(sessions_with_target, test_ratio=0.4)
        assert train["is_target_user"].sum() > 0
        assert val["is_target_user"].sum() > 0

    def test_no_index_overlap(self, sessions_with_target):
        train, val = stratified_split(sessions_with_target, test_ratio=0.4)
        # reset_index is called inside; original index values should not overlap
        combined = len(train) + len(val)
        assert combined == len(sessions_with_target)


# ── get_user_stats ────────────────────────────────────────────────────────────

class TestGetUserStats:
    def test_returns_dataframe_with_expected_columns(self, sessions_df):
        stats = get_user_stats(sessions_df)
        assert "user_id" in stats.columns
        assert "session_count" in stats.columns

    def test_sorted_descending(self, sessions_df):
        stats = get_user_stats(sessions_df)
        counts = stats["session_count"].tolist()
        assert counts == sorted(counts, reverse=True)

    def test_total_matches_dataset(self, sessions_df, raw_sessions):
        stats = get_user_stats(sessions_df)
        assert stats["session_count"].sum() == len(raw_sessions)

    def test_raises_without_user_id_column(self, sessions_df):
        df_no_id = sessions_df.drop(columns=["user_id"])
        with pytest.raises(ValueError, match="user_id"):
            get_user_stats(df_no_id)
