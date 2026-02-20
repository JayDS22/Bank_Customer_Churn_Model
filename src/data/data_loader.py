"""
Data Loader for Consumer Lending Fairness Audit.

Handles loading LendingClub data or generating realistic synthetic data
for testing. Performs cleaning, type conversion, and basic validation.
"""

import logging
import numpy as np
import pandas as pd
import yaml
from pathlib import Path
from typing import Optional, Tuple, Dict

logger = logging.getLogger(__name__)


def load_config(config_path: str = "config/config.yaml") -> dict:
    """Load YAML configuration."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def generate_synthetic_data(n: int = 50000, random_state: int = 42) -> pd.DataFrame:
    """
    Generate realistic synthetic lending data mimicking LendingClub distributions.

    The synthetic data preserves realistic correlations between income, geography,
    loan characteristics, and default outcomes — including deliberate bias patterns
    that the fairness audit should detect.

    Parameters
    ----------
    n : int
        Number of records to generate.
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Synthetic loan dataset.
    """
    rng = np.random.RandomState(random_state)
    logger.info(f"Generating synthetic lending data with {n} records...")

    # --- State assignment with population-weighted probabilities ---
    states = [
        "CA", "TX", "FL", "NY", "PA", "IL", "OH", "GA", "NC", "MI",
        "NJ", "VA", "WA", "AZ", "MA", "TN", "IN", "MO", "MD", "WI",
        "CO", "MN", "SC", "AL", "LA", "KY", "OR", "OK", "CT", "UT",
        "NV", "AR", "MS", "KS", "NM", "NE", "ID", "HI", "ME", "NH",
        "RI", "MT", "DE", "SD", "ND", "AK", "VT", "WY", "DC"
    ]
    state_weights = np.array([
        12, 9, 7, 6, 4, 4, 3.5, 3.2, 3.1, 3,
        2.8, 2.6, 2.4, 2.2, 2.1, 2, 2, 1.9, 1.8, 1.8,
        1.7, 1.7, 1.5, 1.5, 1.4, 1.3, 1.3, 1.2, 1.1, 1,
        0.9, 0.9, 0.9, 0.9, 0.6, 0.6, 0.5, 0.4, 0.4, 0.4,
        0.3, 0.3, 0.3, 0.3, 0.2, 0.2, 0.2, 0.2, 0.2
    ])
    state_weights = state_weights / state_weights.sum()
    addr_state = rng.choice(states, size=n, p=state_weights)

    # --- State-level unemployment rate (instrument for IV) ---
    state_unemployment = {
        "CA": 7.5, "TX": 6.2, "FL": 6.8, "NY": 7.8, "PA": 6.5,
        "IL": 7.1, "OH": 5.8, "GA": 6.3, "NC": 6.1, "MI": 7.0,
        "NJ": 7.2, "VA": 5.3, "WA": 6.4, "AZ": 6.9, "MA": 5.9,
        "TN": 5.7, "IN": 5.5, "MO": 5.6, "MD": 5.8, "WI": 5.2,
        "CO": 5.1, "MN": 4.8, "SC": 6.0, "AL": 6.4, "LA": 6.7,
        "KY": 5.9, "OR": 6.3, "OK": 5.4, "CT": 6.6, "UT": 4.5,
        "NV": 7.5, "AR": 5.8, "MS": 7.2, "KS": 5.0, "NM": 6.5,
        "NE": 4.2, "ID": 5.0, "HI": 4.8, "ME": 5.3, "NH": 4.3,
        "RI": 6.8, "MT": 5.1, "DE": 5.7, "SD": 4.0, "ND": 3.8,
        "AK": 6.5, "VT": 4.5, "WY": 4.7, "DC": 6.9,
    }
    state_unemp = np.array([state_unemployment.get(s, 5.5) for s in addr_state])

    # --- Annual income (correlated with state unemployment) ---
    base_income = rng.lognormal(mean=10.8, sigma=0.7, size=n)
    # Lower income in high-unemployment states (creates IV relevance)
    income_adjustment = 1 - 0.03 * (state_unemp - 5.5)
    annual_inc = np.clip(base_income * income_adjustment, 10000, 500000).round(0)

    # --- Employment length ---
    emp_length_num = rng.choice(range(0, 11), size=n, p=[
        0.08, 0.12, 0.11, 0.10, 0.09, 0.08, 0.07, 0.06, 0.06, 0.06, 0.17
    ])

    # --- Loan characteristics ---
    loan_amnt = rng.choice(
        np.arange(1000, 40001, 500), size=n,
        p=np.ones(79) / 79
    ).astype(float)

    grade_probs = [0.08, 0.22, 0.28, 0.20, 0.12, 0.07, 0.03]
    grade = rng.choice(["A", "B", "C", "D", "E", "F", "G"], size=n, p=grade_probs)

    # Interest rate depends on grade
    grade_rate_map = {"A": 7, "B": 10, "C": 13, "D": 17, "E": 21, "F": 25, "G": 28}
    int_rate = np.array([grade_rate_map[g] + rng.normal(0, 1.5) for g in grade])
    int_rate = np.clip(int_rate, 5, 31).round(2)

    term = rng.choice(["36 months", "60 months"], size=n, p=[0.7, 0.3])
    term_months = np.where(term == "36 months", 36, 60)

    installment = (loan_amnt * (int_rate / 100 / 12) /
                   (1 - (1 + int_rate / 100 / 12) ** (-term_months))).round(2)

    # --- Credit profile ---
    dti = np.clip(rng.normal(17, 8, n), 0, 50).round(2)
    open_acc = rng.poisson(11, n).clip(1, 40)
    total_acc = open_acc + rng.poisson(14, n).clip(0, 60)
    revol_bal = (rng.lognormal(9, 1.2, n)).clip(0, 200000).round(0)
    revol_util = np.clip(rng.beta(2, 3, n) * 100, 0, 100).round(1)

    home_ownership = rng.choice(
        ["RENT", "MORTGAGE", "OWN", "OTHER"], size=n,
        p=[0.40, 0.42, 0.12, 0.06]
    )
    verification_status = rng.choice(
        ["Not Verified", "Source Verified", "Verified"], size=n,
        p=[0.35, 0.35, 0.30]
    )
    purpose = rng.choice(
        ["debt_consolidation", "credit_card", "home_improvement",
         "major_purchase", "small_business", "car", "medical", "other"],
        size=n,
        p=[0.45, 0.15, 0.10, 0.07, 0.06, 0.06, 0.05, 0.06]
    )

    # --- Issue date (for DiD temporal structure) ---
    start = pd.Timestamp("2012-01-01")
    end = pd.Timestamp("2018-12-31")
    days_range = (end - start).days
    issue_d = pd.to_datetime(
        start + pd.to_timedelta(rng.randint(0, days_range, n), unit="D")
    )

    # --- DEFAULT OUTCOME (with deliberate bias for audit detection) ---
    # Base default probability from credit characteristics
    logit = (
        -2.5
        + 0.03 * int_rate
        + 0.015 * dti
        + 0.00001 * revol_bal
        - 0.00001 * annual_inc  # income is protective
        - 0.05 * emp_length_num
        + 0.3 * (term_months == 60).astype(float)
        + 0.2 * np.isin(grade, ["E", "F", "G"]).astype(float)
    )

    # INJECT BIAS: low-income applicants get extra default probability
    # This creates the disparate impact the audit should detect
    income_group_num = np.where(annual_inc < 40000, 1, np.where(annual_inc < 80000, 0.3, 0))
    logit += 0.25 * income_group_num  # bias: income group affects outcome beyond credit risk

    # INJECT GEOGRAPHIC BIAS: Southern states have slightly higher denial
    south_states = ["AL", "AR", "FL", "GA", "KY", "LA", "MS", "NC", "OK", "SC", "TN", "TX", "VA", "WV"]
    logit += 0.1 * np.isin(addr_state, south_states).astype(float)

    prob_default = 1 / (1 + np.exp(-logit))
    default = rng.binomial(1, prob_default)
    loan_status = np.where(default == 1, "Charged Off", "Fully Paid")

    df = pd.DataFrame({
        "loan_amnt": loan_amnt,
        "term": term,
        "int_rate": int_rate,
        "installment": installment,
        "grade": grade,
        "emp_length_num": emp_length_num,
        "home_ownership": home_ownership,
        "annual_inc": annual_inc,
        "verification_status": verification_status,
        "purpose": purpose,
        "addr_state": addr_state,
        "dti": dti,
        "open_acc": open_acc,
        "revol_bal": revol_bal,
        "revol_util": revol_util,
        "total_acc": total_acc,
        "issue_d": issue_d,
        "loan_status": loan_status,
        "state_unemployment_rate": state_unemp,
    })

    logger.info(f"Synthetic data generated: {df.shape[0]} rows, {df.shape[1]} columns")
    logger.info(f"Default rate: {default.mean():.3f}")
    return df


def load_lendingclub_data(filepath: str) -> pd.DataFrame:
    """
    Load and clean LendingClub CSV data.

    Parameters
    ----------
    filepath : str
        Path to the LendingClub CSV file.

    Returns
    -------
    pd.DataFrame
        Cleaned loan dataset.
    """
    logger.info(f"Loading LendingClub data from {filepath}...")

    df = pd.read_csv(filepath, low_memory=False)
    logger.info(f"Raw data: {df.shape[0]} rows, {df.shape[1]} columns")

    # Filter to completed loans only
    valid_statuses = ["Fully Paid", "Charged Off"]
    df = df[df["loan_status"].isin(valid_statuses)].copy()

    # Clean int_rate (remove % sign if present)
    if df["int_rate"].dtype == object:
        df["int_rate"] = df["int_rate"].str.replace("%", "").astype(float)

    # Clean emp_length -> numeric
    if "emp_length" in df.columns and "emp_length_num" not in df.columns:
        emp_map = {"< 1 year": 0, "1 year": 1, "10+ years": 10}
        for i in range(2, 10):
            emp_map[f"{i} years"] = i
        df["emp_length_num"] = df["emp_length"].map(emp_map).fillna(0).astype(int)

    # Parse issue date
    if "issue_d" in df.columns:
        df["issue_d"] = pd.to_datetime(df["issue_d"], format="mixed", errors="coerce")

    # Drop rows with critical nulls
    critical_cols = ["loan_amnt", "annual_inc", "dti", "int_rate"]
    df = df.dropna(subset=[c for c in critical_cols if c in df.columns])

    # Fill remaining numeric nulls with median
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

    logger.info(f"Cleaned data: {df.shape[0]} rows")
    return df


def load_data(config: dict) -> pd.DataFrame:
    """
    Main entry point: load real data if path provided, else generate synthetic.

    Parameters
    ----------
    config : dict
        Configuration dictionary.

    Returns
    -------
    pd.DataFrame
        Loan dataset ready for feature engineering.
    """
    raw_path = config["data"].get("raw_path")

    if raw_path and Path(raw_path).exists():
        return load_lendingclub_data(raw_path)
    else:
        logger.info("No data file found. Generating synthetic data...")
        return generate_synthetic_data(
            n=config["data"].get("sample_size", 50000),
            random_state=config["data"].get("random_state", 42),
        )


def train_test_split_data(
    df: pd.DataFrame, config: dict
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data into train and test sets with stratification on target.

    Parameters
    ----------
    df : pd.DataFrame
        Full dataset with target column.
    config : dict
        Configuration dictionary.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        (train_df, test_df)
    """
    from sklearn.model_selection import train_test_split

    target = config["data"]["target_column"]
    positive = config["data"]["positive_label"]
    df["default"] = (df[target] == positive).astype(int)

    train_df, test_df = train_test_split(
        df,
        test_size=config["data"]["test_size"],
        random_state=config["data"]["random_state"],
        stratify=df["default"],
    )

    logger.info(f"Train: {train_df.shape[0]}, Test: {test_df.shape[0]}")
    logger.info(f"Train default rate: {train_df['default'].mean():.3f}")
    logger.info(f"Test default rate: {test_df['default'].mean():.3f}")

    return train_df, test_df
