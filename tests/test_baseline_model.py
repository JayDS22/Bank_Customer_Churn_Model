"""Tests for baseline model module."""

import pytest
import numpy as np
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.data_loader import load_config, generate_synthetic_data
from src.data.feature_engineer import engineer_features, get_model_features
from src.models.baseline_model import (
    train_logistic_regression,
    evaluate_on_test,
    get_feature_importance,
)


@pytest.fixture
def sample_data():
    config = load_config("config/config.yaml")
    config["data"]["sample_size"] = 2000
    df = generate_synthetic_data(n=2000, random_state=42)
    df = engineer_features(df, config)

    from sklearn.model_selection import train_test_split
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["default"])

    X_train, y_train = get_model_features(train_df, config)
    X_test, y_test = get_model_features(test_df, config)

    return X_train, y_train, X_test, y_test, config


class TestLogisticRegression:
    def test_trains_successfully(self, sample_data):
        X_train, y_train, _, _, config = sample_data
        model, metrics = train_logistic_regression(X_train, y_train, config)
        assert model is not None
        assert "auc_roc" in metrics
        assert 0 < metrics["auc_roc"] <= 1.0

    def test_auc_above_random(self, sample_data):
        X_train, y_train, _, _, config = sample_data
        _, metrics = train_logistic_regression(X_train, y_train, config)
        assert metrics["auc_roc"] > 0.5

    def test_evaluate_on_test(self, sample_data):
        X_train, y_train, X_test, y_test, config = sample_data
        model, _ = train_logistic_regression(X_train, y_train, config)
        test_metrics = evaluate_on_test(model, X_test, y_test, "LR")
        assert "predictions" in test_metrics
        assert len(test_metrics["predictions"]) == len(y_test)

    def test_feature_importance(self, sample_data):
        X_train, y_train, _, _, config = sample_data
        model, _ = train_logistic_regression(X_train, y_train, config)
        fi = get_feature_importance(model, list(X_train.columns), "logistic")
        assert len(fi) > 0
        assert "feature" in fi.columns
        assert "importance" in fi.columns
