"""
Difference-in-Differences (DiD) Analysis for Lending Fairness Audit.

Uses temporal variation in lending policies to estimate causal effects
of income-group membership on loan outcomes, controlling for time trends.
"""

import logging
import numpy as np
import pandas as pd
import statsmodels.api as sm
from typing import Dict, Optional

logger = logging.getLogger(__name__)


def prepare_did_data(
    df: pd.DataFrame,
    config: dict,
) -> pd.DataFrame:
    """
    Prepare panel data for Difference-in-Differences estimation.

    Creates treatment/control indicators and time period flags.

    Parameters
    ----------
    df : pd.DataFrame
        Full dataset with income_group and temporal features.
    config : dict
        Configuration with DiD settings.

    Returns
    -------
    pd.DataFrame
        Dataset with DiD variables added.
    """
    did_cfg = config["causal"]["did"]
    treatment = did_cfg["treatment_group"]
    control = did_cfg["control_group"]

    # Filter to treatment and control groups only
    did_df = df[df["income_group"].isin([treatment, control])].copy()

    # Treatment indicator: 1 if in treatment group (low income)
    did_df["treated"] = (did_df["income_group"] == treatment).astype(int)

    # Interaction term (the DiD estimator)
    did_df["did_interaction"] = did_df["treated"] * did_df["post_policy"]

    logger.info(f"DiD data prepared:")
    logger.info(f"  Treatment ({treatment}): {(did_df['treated'] == 1).sum()}")
    logger.info(f"  Control ({control}): {(did_df['treated'] == 0).sum()}")
    logger.info(f"  Pre-policy: {(did_df['post_policy'] == 0).sum()}")
    logger.info(f"  Post-policy: {(did_df['post_policy'] == 1).sum()}")

    return did_df


def estimate_did(
    did_df: pd.DataFrame,
    outcome: str = "default",
    controls: Optional[list] = None,
) -> Dict:
    """
    Estimate Difference-in-Differences via OLS.

    Model: Y = α + β₁·Post + β₂·Treated + β₃·(Post × Treated) + γ·X + ε

    The coefficient β₃ is the causal DiD estimate.

    Parameters
    ----------
    did_df : pd.DataFrame
        Prepared DiD dataset.
    outcome : str
        Outcome variable name.
    controls : list, optional
        Additional control variables.

    Returns
    -------
    Dict
        DiD estimation results.
    """
    logger.info("Estimating Difference-in-Differences model...")

    # Build regressor matrix
    regressors = ["post_policy", "treated", "did_interaction"]

    if controls:
        available_controls = [c for c in controls if c in did_df.columns]
        regressors.extend(available_controls)

    X = did_df[regressors].copy()
    X = sm.add_constant(X)
    y = did_df[outcome]

    # OLS with heteroskedasticity-robust standard errors
    model = sm.OLS(y, X).fit(cov_type="HC1")

    logger.info(f"\nDiD Regression Results:")
    logger.info(f"{'='*50}")
    logger.info(f"Dependent variable: {outcome}")
    logger.info(f"N = {model.nobs:.0f}, R² = {model.rsquared:.4f}")
    logger.info(f"")
    logger.info(f"{'Variable':<25} {'Coef':>10} {'SE':>10} {'t':>8} {'p':>8}")
    logger.info(f"{'-'*61}")

    for var in model.params.index:
        coef = model.params[var]
        se = model.bse[var]
        t = model.tvalues[var]
        p = model.pvalues[var]
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        logger.info(f"{var:<25} {coef:>10.4f} {se:>10.4f} {t:>8.3f} {p:>8.4f} {sig}")

    # Extract DiD coefficient
    did_coef = model.params["did_interaction"]
    did_se = model.bse["did_interaction"]
    did_pval = model.pvalues["did_interaction"]
    did_ci = model.conf_int().loc["did_interaction"]

    logger.info(f"\nDiD Estimate (Post × Treated):")
    logger.info(f"  Coefficient: {did_coef:.4f}")
    logger.info(f"  Standard Error: {did_se:.4f}")
    logger.info(f"  p-value: {did_pval:.4f}")
    logger.info(f"  95% CI: [{did_ci[0]:.4f}, {did_ci[1]:.4f}]")

    if did_pval < 0.05:
        direction = "increased" if did_coef > 0 else "decreased"
        logger.info(f"  ⚠️  Statistically significant: policy {direction} default disparity")
    else:
        logger.info(f"  ✓ Not statistically significant at 5% level")

    return {
        "model": model,
        "did_coefficient": float(did_coef),
        "did_se": float(did_se),
        "did_pvalue": float(did_pval),
        "did_ci_lower": float(did_ci[0]),
        "did_ci_upper": float(did_ci[1]),
        "r_squared": float(model.rsquared),
        "n_obs": int(model.nobs),
        "summary": model.summary(),
    }


def parallel_trends_test(
    did_df: pd.DataFrame,
    outcome: str = "default",
) -> Dict:
    """
    Test the parallel trends assumption (pre-treatment).

    Checks whether treatment and control groups had similar trends
    before the policy change.

    Parameters
    ----------
    did_df : pd.DataFrame
        Prepared DiD dataset with issue_year.
    outcome : str
        Outcome variable.

    Returns
    -------
    Dict
        Parallel trends test results.
    """
    logger.info("\nTesting parallel trends assumption (pre-treatment)...")

    pre_data = did_df[did_df["post_policy"] == 0].copy()

    if "issue_year" not in pre_data.columns or pre_data.empty:
        logger.warning("Insufficient temporal data for parallel trends test.")
        return {"test_passed": None, "message": "Insufficient data"}

    # Group-year means
    trends = (
        pre_data.groupby(["issue_year", "treated"])[outcome]
        .mean()
        .unstack(fill_value=0)
    )

    # Test: regress outcome on year × treated interaction (pre-period only)
    pre_data["year_trend"] = pre_data["issue_year"] - pre_data["issue_year"].min()
    pre_data["trend_x_treated"] = pre_data["year_trend"] * pre_data["treated"]

    X = sm.add_constant(pre_data[["year_trend", "treated", "trend_x_treated"]])
    y = pre_data[outcome]
    model = sm.OLS(y, X).fit(cov_type="HC1")

    interaction_pval = model.pvalues.get("trend_x_treated", 1.0)
    passed = interaction_pval > 0.05

    if passed:
        logger.info(f"  ✓ Parallel trends assumption holds (p={interaction_pval:.4f})")
    else:
        logger.warning(f"  ⚠️  Parallel trends may be violated (p={interaction_pval:.4f})")

    return {
        "test_passed": passed,
        "interaction_pvalue": float(interaction_pval),
        "trend_data": trends,
        "model_summary": model.summary(),
    }


def run_did_analysis(df: pd.DataFrame, config: dict) -> Dict:
    """
    Run the complete DiD analysis pipeline.

    Parameters
    ----------
    df : pd.DataFrame
        Full engineered dataset.
    config : dict
        Configuration dictionary.

    Returns
    -------
    Dict
        Complete DiD results.
    """
    logger.info("\n" + "=" * 60)
    logger.info("CAUSAL ANALYSIS: Difference-in-Differences")
    logger.info("=" * 60)

    did_df = prepare_did_data(df, config)

    # Controls from IV config (reuse)
    controls = config["causal"]["iv"].get("controls", [])
    available_controls = [c for c in controls if c in did_df.columns]

    # Main DiD estimation
    did_results = estimate_did(did_df, outcome="default", controls=available_controls)

    # Parallel trends test
    trends_results = parallel_trends_test(did_df)

    return {
        "did": did_results,
        "parallel_trends": trends_results,
        "data": did_df,
    }
