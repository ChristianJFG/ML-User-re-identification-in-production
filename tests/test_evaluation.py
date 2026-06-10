"""Tests for catch_joe.evaluation — metrics computation."""
from __future__ import annotations

import numpy as np
import pytest

from catch_joe.evaluation import compute_all_metrics


class TestComputeAllMetrics:
    def _perfect_inputs(self):
        y_true = np.array([1, 1, 0, 0, 0])
        y_score = np.array([0.9, 0.8, 0.1, 0.2, 0.05])
        return y_true, y_score

    def _all_negative_score(self):
        y_true = np.array([1, 0, 0, 0])
        y_score = np.array([0.1, 0.1, 0.1, 0.1])
        return y_true, y_score

    def test_returns_dict_with_required_keys(self):
        y_true, y_score = self._perfect_inputs()
        metrics = compute_all_metrics(y_true, y_score)
        for key in ("pr_auc", "roc_auc", "precision", "recall", "f1"):
            assert key in metrics

    def test_perfect_scores_give_auc_one(self):
        y_true, y_score = self._perfect_inputs()
        metrics = compute_all_metrics(y_true, y_score)
        assert metrics["pr_auc"] == pytest.approx(1.0)
        assert metrics["roc_auc"] == pytest.approx(1.0)

    def test_metrics_are_floats(self):
        y_true, y_score = self._perfect_inputs()
        metrics = compute_all_metrics(y_true, y_score)
        for v in metrics.values():
            assert isinstance(v, float)

    def test_pr_auc_between_zero_and_one(self):
        y_true, y_score = self._all_negative_score()
        metrics = compute_all_metrics(y_true, y_score)
        assert 0.0 <= metrics["pr_auc"] <= 1.0

    def test_roc_auc_between_zero_and_one(self):
        y_true, y_score = self._all_negative_score()
        metrics = compute_all_metrics(y_true, y_score)
        assert 0.0 <= metrics["roc_auc"] <= 1.0

    def test_precision_recall_f1_between_zero_and_one(self):
        y_true, y_score = self._perfect_inputs()
        metrics = compute_all_metrics(y_true, y_score)
        for key in ("precision", "recall", "f1"):
            assert 0.0 <= metrics[key] <= 1.0

    def test_k_precision_and_recall_returned_when_k_given(self):
        y_true, y_score = self._perfect_inputs()
        metrics = compute_all_metrics(y_true, y_score, k=2)
        assert "precision_at_2" in metrics
        assert "recall_at_2" in metrics

    def test_no_k_metrics_when_k_not_given(self):
        y_true, y_score = self._perfect_inputs()
        metrics = compute_all_metrics(y_true, y_score)
        assert "precision_at_k" not in metrics
        assert "recall_at_k" not in metrics

    def test_custom_threshold_affects_precision_recall(self):
        y_true = np.array([1, 1, 0, 0])
        y_score = np.array([0.6, 0.4, 0.3, 0.2])
        # threshold=0.5 → only top prediction is positive
        m_high = compute_all_metrics(y_true, y_score, threshold=0.5)
        # threshold=0.35 → two predictions are positive
        m_low = compute_all_metrics(y_true, y_score, threshold=0.35)
        assert m_low["recall"] >= m_high["recall"]
