"""
Instrumental Variables (IV / 2SLS) Analysis for Lending Fairness Audit.

Uses state-level unemployment rate as an instrument for income to isolate
the causal effect of income on default, separating it from geographic confounders.
"""

import logging
import numpy as np
import pandas as pd
import statsmodels.api as sm
from typing import Dict, Optional

logger = logging.getLogger(__name__)


def estimate_iv_2sls(
    df: pd.DataFrame,
    config: dict,
) -> Dict:
    """
    Estimate IV/2SLS model using linearmodels or manual 2SLS.

    First stage: income ~ unemployment_rate + controls
    Second stage: default ~ income_hat + controls

    Parameters
    ----------
    df : pd.DataFrame
        Dataset with instrument, endogenous, and control variables.
    config : dict
        Configuration with IV settings.

    Returns
    -------
    Dict
        IV estimation results.
    """
    logger.info("\n" + "=" * 60)
    logger.info("CAUSAL ANALYSIS: Instrumental Variables (2SLS)")
    logger.info("=" * 60)

    iv_cfg = config["causal"]["iv"]
    instrument = iv_cfg["instrument"]
    endogenous = iv_cfg["endogenous"]
    controls = [c for c in iv_cfg["controls"] if c in df.columns]

    # Clean data
    all_vars = [instrument, endogenous, "default"] + controls
    iv_df = df[all_vars].dropna().copy()

    # Log-transform income for better linearity
    iv_df["log_income"] = np.log1p(iv_df[endogenous])

    logger.info(f"IV data: {iv_df.shape[0]} observations")
    logger.info(f"Instrument: {instrument}")
    logger.info(f"Endogenous: {endogenous} (log-transformed)")
    logger.info(f"Controls: {controls}")

    # Try linearmodels first, fall back to manual 2SLS
    try:
        return _estimate_with_linearmodels(iv_df, instrument, controls, config)
    except ImportError:
        logger.info("linearmodels not available. Using manual 2SLS...")
        return _estimate_manual_2sls(iv_df, instrument, controls)


def _estimate_with_linearmodels(
    iv_df: pd.DataFrame,
    instrument: str,
    controls: list,
    config: dict,
) -> Dict:
    """Estimate IV using linearmodels.IV2SLS."""
    from linearmodels.iv import IV2SLS

    # Prepare variables
    dependent = iv_df["default"]
    endog = iv_df[["log_income"]]
    instruments = iv_df[[instrument]]
    exog = sm.add_constant(iv_df[controls]) if controls else sm.add_constant(pd.DataFrame(index=iv_df.index))

    model = IV2SLS(dependent, exog, endog, instruments).fit(cov_type="robust")

    logger.info(f"\n2SLS Results:")
    logger.info(f"{'='*50}")
    logger.info(str(model.summary))

    # First stage diagnostics
    first_stage = _first_stage_diagnostics(iv_df, instrument, controls)

    # Extract key coefficient
    income_coef = float(model.params.get("log_income", 0))
    income_pval = float(model.pvalues.get("log_income", 1))

    logger.info(f"\nKey Result: log(income) coefficient = {income_coef:.4f} (p={income_pval:.4f})")
    if income_pval < 0.05:
        direction = "reduces" if income_coef < 0 else "increases"
        logger.info(f"  ⚠️  Income causally {direction} default probability")
    else:
        logger.info(f"  Income effect not statistically significant at 5%")

    return {
        "model": model,
        "income_coefficient": income_coef,
        "income_pvalue": income_pval,
        "first_stage": first_stage,
        "method": "linearmodels.IV2SLS",
    }


def _estimate_manual_2sls(
    iv_df: pd.DataFrame,
    instrument: str,
    controls: list,
) -> Dict:
    """Manual two-stage least squares estimation."""

    # --- First Stage: log_income ~ instrument + controls ---
    logger.info("\n--- First Stage ---")
    first_regressors = [instrument] + controls
    X1 = sm.add_constant(iv_df[first_regressors])
    y1 = iv_df["log_income"]

    first_stage = sm.OLS(y1, X1).fit(cov_type="HC1")

    # F-statistic on excluded instrument
    f_stat = first_stage.fvalue
    logger.info(f"First-stage F-statistic: {f_stat:.2f}")
    if f_stat < 10:
        logger.warning("  ⚠️  Weak instrument (F < 10)")
    else:
        logger.info("  ✓ Instrument is strong (F ≥ 10)")

    instrument_coef = first_stage.params[instrument]
    instrument_pval = first_stage.pvalues[instrument]
    logger.info(f"Instrument coefficient: {instrument_coef:.4f} (p={instrument_pval:.4f})")

    # Predicted (fitted) values
    iv_df["log_income_hat"] = first_stage.fittedvalues

    # --- Second Stage: default ~ log_income_hat + controls ---
    logger.info("\n--- Second Stage ---")
    second_regressors = ["log_income_hat"] + controls
    X2 = sm.add_constant(iv_df[second_regressors])
    y2 = iv_df["default"]

    second_stage = sm.OLS(y2, X2).fit(cov_type="HC1")

    income_coef = float(second_stage.params["log_income_hat"])
    income_se = float(second_stage.bse["log_income_hat"])
    income_pval = float(second_stage.pvalues["log_income_hat"])

    logger.info(f"\n2SLS Results:")
    logger.info(f"  log(income) coefficient: {income_coef:.4f}")
    logger.info(f"  Standard error: {income_se:.4f}")
    logger.info(f"  p-value: {income_pval:.4f}")

    return {
        "first_stage_model": first_stage,
        "second_stage_model": second_stage,
        "income_coefficient": income_coef,
        "income_se": income_se,
        "income_pvalue": income_pval,
        "first_stage_f": float(f_stat),
        "instrument_coefficient": float(instrument_coef),
        "instrument_pvalue": float(instrument_pval),
        "method": "manual_2sls",
    }


def _first_stage_diagnostics(
    iv_df: pd.DataFrame,
    instrument: str,
    controls: list,
) -> Dict:
    """
    Run first-stage instrument diagnostics.

    Checks relevance (F-stat) and correlation patterns.
    """
    first_regressors = [instrument] + controls
    X = sm.add_constant(iv_df[first_regressors])
    y = iv_df["log_income"]

    model = sm.OLS(y, X).fit(cov_type="HC1")

    f_stat = float(model.fvalue)
    r_squared = float(model.rsquared)
    instrument_coef = float(model.params[instrument])
    instrument_pval = float(model.pvalues[instrument])

    # Correlation between instrument and endogenous
    corr = iv_df[instrument].corr(iv_df["log_income"])

    logger.info(f"\nFirst-Stage Diagnostics:")
    logger.info(f"  F-statistic: {f_stat:.2f} ({'Strong' if f_stat >= 10 else 'Weak'})")
    logger.info(f"  R²: {r_squared:.4f}")
    logger.info(f"  Instrument-endogenous correlation: {corr:.4f}")

    return {
        "f_statistic": f_stat,
        "r_squared": r_squared,
        "instrument_coefficient": instrument_coef,
        "instrument_pvalue": instrument_pval,
        "correlation": float(corr),
        "is_strong": f_stat >= 10,
    }


def refutation_test(
    df: pd.DataFrame,
    config: dict,
    n_permutations: int = 100,
) -> Dict:
    """
    Placebo/refutation test: permute instrument assignment.

    If the IV result is causal, randomizing the instrument should
    destroy the second-stage effect.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset.
    config : dict
        Configuration.
    n_permutations : int
        Number of permutations.

    Returns
    -------
    Dict
        Refutation test results.
    """
    logger.info(f"\nRunning placebo refutation test ({n_permutations} permutations)...")

    iv_cfg = config["causal"]["iv"]
    instrument = iv_cfg["instrument"]
    controls = [c for c in iv_cfg["controls"] if c in df.columns]

    all_vars = [instrument, "annual_inc", "default"] + controls
    iv_df = df[all_vars].dropna().copy()
    iv_df["log_income"] = np.log1p(iv_df["annual_inc"])

    rng = np.random.RandomState(42)
    placebo_coefs = []

    for i in range(n_permutations):
        shuffled = iv_df.copy()
        shuffled[instrument] = rng.permutation(shuffled[instrument].values)

        try:
            X1 = sm.add_constant(shuffled[[instrument] + controls])
            first = sm.OLS(shuffled["log_income"], X1).fit()
            shuffled["log_income_hat"] = first.fittedvalues

            X2 = sm.add_constant(shuffled[["log_income_hat"] + controls])
            second = sm.OLS(shuffled["default"], X2).fit()
            placebo_coefs.append(second.params["log_income_hat"])
        except Exception:
            continue

    placebo_coefs = np.array(placebo_coefs)
    placebo_mean = placebo_coefs.mean()
    placebo_std = placebo_coefs.std()

    logger.info(f"  Placebo coefficient mean: {placebo_mean:.6f}")
    logger.info(f"  Placebo coefficient std: {placebo_std:.6f}")
    logger.info(f"  (Should be centered near 0 if original result is causal)")

    return {
        "placebo_mean": float(placebo_mean),
        "placebo_std": float(placebo_std),
        "placebo_coefs": placebo_coefs,
        "n_permutations": len(placebo_coefs),
    }


def run_iv_analysis(df: pd.DataFrame, config: dict) -> Dict:
    """
    Run the complete IV analysis pipeline.

    Parameters
    ----------
    df : pd.DataFrame
        Full engineered dataset.
    config : dict
        Configuration dictionary.

    Returns
    -------
    Dict
        Complete IV results including refutation.
    """
    iv_results = estimate_iv_2sls(df, config)
    refutation = refutation_test(df, config, n_permutations=50)

    return {
        "iv": iv_results,
        "refutation": refutation,
    }
