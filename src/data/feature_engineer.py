"""
Feature Engineering for Consumer Lending Fairness Audit.

Creates income segments, geographic regions, derived financial ratios,
and temporal features needed for fairness analysis and causal inference.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)


def add_income_segments(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Bin annual income into Low/Medium/High groups.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset with 'annual_inc' column.
    config : dict
        Configuration with income thresholds.

    Returns
    -------
    pd.DataFrame
        Dataset with 'income_group' column added.
    """
    seg = config["segments"]["income"]
    low = seg["low_threshold"]
    high = seg["high_threshold"]

    conditions = [
        df["annual_inc"] < low,
        (df["annual_inc"] >= low) & (df["annual_inc"] < high),
        df["annual_inc"] >= high,
    ]
    df["income_group"] = np.select(conditions, seg["labels"], default="Medium")

    counts = df["income_group"].value_counts()
    logger.info(f"Income segments: {counts.to_dict()}")
    return df


def add_geographic_segments(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Map states to Census regions.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset with 'addr_state' column.
    config : dict
        Configuration with state-to-region mapping.

    Returns
    -------
    pd.DataFrame
        Dataset with 'state_region' column added.
    """
    region_map = {}
    for region, state_list in config["segments"]["geography"]["regions"].items():
        for state in state_list:
            region_map[state] = region

    df["state_region"] = df["addr_state"].map(region_map).fillna("Other")

    counts = df["state_region"].value_counts()
    logger.info(f"Geographic segments: {counts.to_dict()}")
    return df


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create derived financial ratios and interaction features.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset with raw lending features.

    Returns
    -------
    pd.DataFrame
        Dataset with additional engineered features.
    """
    # Payment-to-income ratio
    df["payment_to_income"] = np.where(
        df["annual_inc"] > 0,
        (df["installment"] * 12) / df["annual_inc"],
        0,
    )

    # Loan-to-income ratio
    df["loan_to_income"] = np.where(
        df["annual_inc"] > 0,
        df["loan_amnt"] / df["annual_inc"],
        0,
    )

    # Credit utilization bucket
    df["high_utilization"] = (df["revol_util"] > 75).astype(int)

    # Account diversity
    df["open_acc_ratio"] = np.where(
        df["total_acc"] > 0,
        df["open_acc"] / df["total_acc"],
        0,
    )

    # Log-transformed income (for modeling)
    df["log_annual_inc"] = np.log1p(df["annual_inc"])

    # Income × DTI interaction
    df["income_dti_interaction"] = df["log_annual_inc"] * df["dti"]

    # Grade as numeric ordinal
    grade_map = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7}
    df["grade_num"] = df["grade"].map(grade_map).fillna(4)

    logger.info(f"Added {7} derived features. Total columns: {df.shape[1]}")
    return df


def add_temporal_features(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Add temporal features for DiD analysis.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset with 'issue_d' column.
    config : dict
        Configuration with policy change date.

    Returns
    -------
    pd.DataFrame
        Dataset with 'post_policy', 'issue_year', 'issue_quarter' columns.
    """
    if "issue_d" not in df.columns:
        logger.warning("No issue_d column found. Skipping temporal features.")
        return df

    policy_date = pd.Timestamp(config["causal"]["did"]["policy_change_date"])
    df["post_policy"] = (df["issue_d"] >= policy_date).astype(int)
    df["issue_year"] = df["issue_d"].dt.year
    df["issue_quarter"] = df["issue_d"].dt.quarter

    pre_count = (df["post_policy"] == 0).sum()
    post_count = (df["post_policy"] == 1).sum()
    logger.info(f"Temporal split: {pre_count} pre-policy, {post_count} post-policy")
    return df


def get_model_features(df: pd.DataFrame, config: dict) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Prepare feature matrix X and target vector y for modeling.

    Encodes categorical variables and selects features specified in config.

    Parameters
    ----------
    df : pd.DataFrame
        Fully engineered dataset.
    config : dict
        Configuration dictionary.

    Returns
    -------
    Tuple[pd.DataFrame, pd.Series]
        (X, y) ready for sklearn.
    """
    target = "default"
    if target not in df.columns:
        positive = config["data"]["positive_label"]
        df[target] = (df[config["data"]["target_column"]] == positive).astype(int)

    y = df[target]

    # Numeric features
    num_feats = [
        f for f in config["data"]["numeric_features"] if f in df.columns
    ]

    # Add derived numeric features
    derived = [
        "payment_to_income", "loan_to_income", "high_utilization",
        "open_acc_ratio", "log_annual_inc", "income_dti_interaction", "grade_num",
    ]
    num_feats += [f for f in derived if f in df.columns]

    # Categorical features → one-hot
    cat_feats = [
        f for f in config["data"]["categorical_features"] if f in df.columns
    ]

    X_numeric = df[num_feats].copy()
    X_categorical = pd.get_dummies(df[cat_feats], drop_first=True, dtype=float)
    X = pd.concat([X_numeric, X_categorical], axis=1)

    # Fill any remaining NaN
    X = X.fillna(0)

    logger.info(f"Feature matrix: {X.shape[1]} features, {X.shape[0]} samples")
    return X, y


def engineer_features(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Run the full feature engineering pipeline.

    Parameters
    ----------
    df : pd.DataFrame
        Raw or cleaned loan dataset.
    config : dict
        Configuration dictionary.

    Returns
    -------
    pd.DataFrame
        Fully engineered dataset.
    """
    logger.info("Starting feature engineering pipeline...")

    # Create target variable
    positive = config["data"]["positive_label"]
    df["default"] = (df[config["data"]["target_column"]] == positive).astype(int)

    df = add_income_segments(df, config)
    df = add_geographic_segments(df, config)
    df = add_derived_features(df)
    df = add_temporal_features(df, config)

    logger.info(f"Feature engineering complete. Shape: {df.shape}")
    return df
