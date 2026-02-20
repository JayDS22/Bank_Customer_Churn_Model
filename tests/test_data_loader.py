"""Tests for data loader module."""

import pytest
import numpy as np
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.data_loader import generate_synthetic_data, load_config, load_data


class TestSyntheticDataGeneration:
    """Tests for synthetic data generation."""

    def test_generates_correct_number_of_rows(self):
        df = generate_synthetic_data(n=1000, random_state=42)
        assert len(df) == 1000

    def test_has_required_columns(self):
        df = generate_synthetic_data(n=100, random_state=42)
        required = [
            "loan_amnt", "term", "int_rate", "installment", "grade",
            "annual_inc", "dti", "addr_state", "loan_status",
            "state_unemployment_rate",
        ]
        for col in required:
            assert col in df.columns, f"Missing column: {col}"

    def test_loan_status_values(self):
        df = generate_synthetic_data(n=5000, random_state=42)
        assert set(df["loan_status"].unique()) == {"Fully Paid", "Charged Off"}

    def test_income_positive(self):
        df = generate_synthetic_data(n=1000, random_state=42)
        assert (df["annual_inc"] > 0).all()

    def test_interest_rate_range(self):
        df = generate_synthetic_data(n=1000, random_state=42)
        assert df["int_rate"].min() >= 5
        assert df["int_rate"].max() <= 31

    def test_default_rate_reasonable(self):
        df = generate_synthetic_data(n=10000, random_state=42)
        default_rate = (df["loan_status"] == "Charged Off").mean()
        assert 0.05 < default_rate < 0.50, f"Default rate {default_rate} seems unreasonable"

    def test_reproducibility(self):
        df1 = generate_synthetic_data(n=100, random_state=42)
        df2 = generate_synthetic_data(n=100, random_state=42)
        pd.testing.assert_frame_equal(df1, df2)

    def test_state_unemployment_present(self):
        df = generate_synthetic_data(n=1000, random_state=42)
        assert "state_unemployment_rate" in df.columns
        assert df["state_unemployment_rate"].notna().all()


class TestLoadConfig:
    """Tests for configuration loading."""

    def test_load_default_config(self):
        config = load_config("config/config.yaml")
        assert "data" in config
        assert "model" in config
        assert "fairness" in config
        assert "causal" in config
        assert "bayesian" in config

    def test_config_has_thresholds(self):
        config = load_config("config/config.yaml")
        assert config["fairness"]["disparate_impact_ratio_threshold"] == 0.80


class TestLoadData:
    """Tests for the main load_data function."""

    def test_load_data_synthetic_fallback(self):
        config = load_config("config/config.yaml")
        config["data"]["raw_path"] = None
        config["data"]["sample_size"] = 500
        df = load_data(config)
        assert len(df) == 500
