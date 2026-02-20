"""Tests for fairness metrics module."""

import pytest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.fairness.fairness_metrics import (
    demographic_parity,
    equalized_odds,
    predictive_parity,
    disparate_impact_ratio,
    compute_all_fairness_metrics,
)
from src.data.data_loader import load_config


class TestDemographicParity:
    def test_equal_rates(self):
        y_pred = np.array([1, 0, 1, 1, 0, 1])
        groups = np.array(["A", "A", "A", "B", "B", "B"])
        dp = demographic_parity(y_pred, groups)
        assert abs(dp["A"] - dp["B"]) < 0.01

    def test_unequal_rates(self):
        y_pred = np.array([1, 1, 1, 0, 0, 0])
        groups = np.array(["A", "A", "A", "B", "B", "B"])
        dp = demographic_parity(y_pred, groups)
        assert dp["A"] == 1.0
        assert dp["B"] == 0.0


class TestEqualizedOdds:
    def test_returns_tpr_fpr(self):
        y_true = np.array([1, 0, 1, 0, 1, 0])
        y_pred = np.array([1, 0, 1, 1, 0, 0])
        groups = np.array(["A", "A", "A", "B", "B", "B"])
        eo = equalized_odds(y_true, y_pred, groups)
        assert "tpr" in eo["A"]
        assert "fpr" in eo["A"]

    def test_perfect_classifier(self):
        y_true = np.array([1, 0, 1, 0])
        y_pred = np.array([1, 0, 1, 0])
        groups = np.array(["A", "A", "B", "B"])
        eo = equalized_odds(y_true, y_pred, groups)
        assert eo["A"]["tpr"] == 1.0
        assert eo["A"]["fpr"] == 0.0


class TestPredictiveParity:
    def test_perfect_precision(self):
        y_true = np.array([1, 1, 1, 0])
        y_pred = np.array([1, 1, 1, 0])
        groups = np.array(["A", "A", "B", "B"])
        pp = predictive_parity(y_true, y_pred, groups)
        assert pp["A"] == 1.0


class TestDisparateImpact:
    def test_equal_impact(self):
        y_pred = np.array([1, 0, 1, 0])
        groups = np.array(["A", "A", "B", "B"])
        ratio = disparate_impact_ratio(y_pred, groups, "A", "B")
        assert abs(ratio - 1.0) < 0.01

    def test_disparate_impact_detected(self):
        y_pred = np.array([1, 1, 1, 1, 0, 0, 0, 1])
        groups = np.array(["A", "A", "A", "A", "B", "B", "B", "B"])
        ratio = disparate_impact_ratio(y_pred, groups, "A", "B")
        assert ratio < 0.80


class TestComputeAllFairnessMetrics:
    def test_returns_all_metrics(self):
        np.random.seed(42)
        n = 1000
        y_true = np.random.binomial(1, 0.3, n)
        y_pred = np.random.binomial(1, 0.3, n)
        groups = np.random.choice(["Low", "High"], n)
        config = load_config("config/config.yaml")

        results = compute_all_fairness_metrics(y_true, y_pred, groups, "test_attr", config)

        assert "demographic_parity" in results
        assert "equalized_odds" in results
        assert "predictive_parity" in results
        assert "disparate_impact" in results
        assert "violations" in results
