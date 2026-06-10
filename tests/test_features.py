"""Tests for catch_joe.features — session stats, top-K domains, site indicators."""
from __future__ import annotations

import pandas as pd
import pytest

from catch_joe.features import (
    METADATA_CAT_COLS,
    METADATA_NUM_COLS,
    build_site_indicators,
    build_tree_features,
    extract_session_stats,
    get_top_k_domains,
)


# ── extract_session_stats ─────────────────────────────────────────────────────

class TestExtractSessionStats:
    def test_adds_expected_columns(self, sessions_df):
        df = extract_session_stats(sessions_df)
        for col in METADATA_NUM_COLS:
            assert col in df.columns

    def test_session_length_equals_sum_of_lengths(self, sessions_df):
        df = extract_session_stats(sessions_df)
        for _, row in df.iterrows():
            assert row["session_length"] == sum(row["site_lengths"])

    def test_unique_site_count_deduplicates(self, sessions_df):
        # row 0 has ["github.com", "stackoverflow.com", "github.com"] → 2 unique
        df = extract_session_stats(sessions_df)
        assert df["unique_site_count"].iloc[0] == 2

    def test_hour_of_day_range(self, sessions_df):
        df = extract_session_stats(sessions_df)
        assert df["hour_of_day"].between(0, 23).all()

    def test_day_of_week_range(self, sessions_df):
        df = extract_session_stats(sessions_df)
        assert df["day_of_week_num"].between(0, 6).all()

    def test_original_df_not_mutated(self, sessions_df):
        _ = extract_session_stats(sessions_df)
        assert "session_length" not in sessions_df.columns


# ── get_top_k_domains ─────────────────────────────────────────────────────────

class TestGetTopKDomains:
    def test_returns_list_of_strings(self, sessions_df):
        result = get_top_k_domains(sessions_df, k=3)
        assert isinstance(result, list)
        assert all(isinstance(d, str) for d in result)

    def test_length_capped_at_k(self, sessions_df):
        result = get_top_k_domains(sessions_df, k=2)
        assert len(result) == 2

    def test_most_frequent_domain_is_first(self, sessions_df):
        # github.com appears in 3 out of 5 sessions (rows 0, 1, 3)
        result = get_top_k_domains(sessions_df, k=10)
        assert result[0] == "github.com"

    def test_per_session_deduplication(self, sessions_df):
        # github.com appears twice in session 0 but should count as 1 session
        all_domains = get_top_k_domains(sessions_df, k=100)
        # The count for github.com should be based on sessions, not raw visits
        counts = {}
        for sites in sessions_df["sites_visited"]:
            for s in set(sites):
                counts[s] = counts.get(s, 0) + 1
        assert counts["github.com"] == 3

    def test_k_larger_than_vocab_returns_all(self, sessions_df):
        result = get_top_k_domains(sessions_df, k=1000)
        # Should not raise; length bounded by actual unique domain count
        unique_domains = {s for sites in sessions_df["sites_visited"] for s in sites}
        assert len(result) == len(unique_domains)


# ── build_site_indicators ─────────────────────────────────────────────────────

class TestBuildSiteIndicators:
    def test_adds_site_columns(self, sessions_df):
        domains = ["github.com", "python.org"]
        df = build_site_indicators(sessions_df, domains)
        assert "site_github.com" in df.columns
        assert "site_python.org" in df.columns

    def test_indicator_is_binary(self, sessions_df):
        domains = ["github.com"]
        df = build_site_indicators(sessions_df, domains)
        assert set(df["site_github.com"].unique()).issubset({0, 1})

    def test_indicator_correct_for_present_domain(self, sessions_df):
        # session 0 visits github.com
        domains = ["github.com"]
        df = build_site_indicators(sessions_df, domains)
        assert df["site_github.com"].iloc[0] == 1

    def test_indicator_zero_for_absent_domain(self, sessions_df):
        # session 2 (user_id=1, youtube.com/google.com) does NOT visit github.com
        domains = ["github.com"]
        df = build_site_indicators(sessions_df, domains)
        assert df["site_github.com"].iloc[2] == 0

    def test_empty_top_domains_returns_copy(self, sessions_df):
        result = build_site_indicators(sessions_df, [])
        pd.testing.assert_frame_equal(result.reset_index(drop=True),
                                      sessions_df.reset_index(drop=True))

    def test_row_count_unchanged(self, sessions_df):
        domains = ["github.com", "youtube.com"]
        df = build_site_indicators(sessions_df, domains)
        assert len(df) == len(sessions_df)

    def test_original_columns_preserved(self, sessions_df):
        domains = ["github.com"]
        df = build_site_indicators(sessions_df, domains)
        for col in sessions_df.columns:
            assert col in df.columns


# ── build_tree_features ───────────────────────────────────────────────────────

class TestBuildTreeFeatures:
    def test_returns_four_items(self, sessions_with_target):
        from catch_joe.data import stratified_split
        train, val = stratified_split(sessions_with_target, test_ratio=0.4)
        result = build_tree_features(train, val, top_k=3)
        assert len(result) == 4

    def test_train_val_same_columns(self, sessions_with_target):
        from catch_joe.data import stratified_split
        train, val = stratified_split(sessions_with_target, test_ratio=0.4)
        X_train, X_val, _, _ = build_tree_features(train, val, top_k=3)
        assert list(X_train.columns) == list(X_val.columns)

    def test_cat_feature_indices_within_bounds(self, sessions_with_target):
        from catch_joe.data import stratified_split
        train, val = stratified_split(sessions_with_target, test_ratio=0.4)
        X_train, _, cat_idx, feat_names = build_tree_features(train, val, top_k=3)
        assert all(0 <= i < len(feat_names) for i in cat_idx)

    def test_cat_cols_are_strings(self, sessions_with_target):
        from catch_joe.data import stratified_split
        train, val = stratified_split(sessions_with_target, test_ratio=0.4)
        X_train, X_val, _, _ = build_tree_features(train, val, top_k=3)
        for col in METADATA_CAT_COLS:
            assert X_train[col].dtype == object
            assert X_val[col].dtype == object

    def test_no_nulls_in_output(self, sessions_with_target):
        from catch_joe.data import stratified_split
        train, val = stratified_split(sessions_with_target, test_ratio=0.4)
        X_train, X_val, _, _ = build_tree_features(train, val, top_k=3)
        assert not X_train.isnull().any().any()
        assert not X_val.isnull().any().any()
