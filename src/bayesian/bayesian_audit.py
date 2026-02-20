"""
Bayesian Uncertainty Quantification for Lending Fairness Audit.

Estimates posterior distributions of fairness gap parameters using PyMC.
Provides credible intervals to assess statistical robustness of bias findings.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)


def bayesian_fairness_gap(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    groups: np.ndarray,
    group_a: str,
    group_b: str,
    config: dict,
) -> Dict:
    """
    Estimate the Bayesian posterior of the fairness gap between two groups.

    Uses a Beta-Binomial model for each group's positive prediction rate,
    then computes the posterior of the difference.

    Parameters
    ----------
    y_true : np.ndarray
        True labels.
    y_pred : np.ndarray
        Predicted labels.
    groups : np.ndarray
        Group labels.
    group_a : str
        First group name.
    group_b : str
        Second group name.
    config : dict
        Bayesian configuration.

    Returns
    -------
    Dict
        Posterior summary with HDI.
    """
    try:
        import pymc as pm
        import arviz as az
    except ImportError:
        logger.warning("PyMC not installed. Using analytical Beta-Binomial approximation.")
        return _analytical_bayesian_gap(y_pred, groups, group_a, group_b)

    logger.info(f"\nBayesian fairness gap estimation: {group_a} vs {group_b}")

    bayes_cfg = config["bayesian"]
    mask_a = groups == group_a
    mask_b = groups == group_b

    n_a = mask_a.sum()
    k_a = y_pred[mask_a].sum()
    n_b = mask_b.sum()
    k_b = y_pred[mask_b].sum()

    logger.info(f"  {group_a}: {int(k_a)}/{n_a} positive predictions")
    logger.info(f"  {group_b}: {int(k_b)}/{n_b} positive predictions")

    with pm.Model() as model:
        # Priors: weakly informative Beta
        rate_a = pm.Beta("rate_a", alpha=1, beta=1)
        rate_b = pm.Beta("rate_b", alpha=1, beta=1)

        # Likelihood
        obs_a = pm.Binomial("obs_a", n=int(n_a), p=rate_a, observed=int(k_a))
        obs_b = pm.Binomial("obs_b", n=int(n_b), p=rate_b, observed=int(k_b))

        # Derived: fairness gap
        gap = pm.Deterministic("fairness_gap", rate_a - rate_b)

        # Derived: disparate impact ratio
        di_ratio = pm.Deterministic("di_ratio", rate_a / pm.math.maximum(rate_b, 0.001))

        # Sample
        trace = pm.sample(
            draws=bayes_cfg["samples"],
            tune=bayes_cfg["tune"],
            chains=bayes_cfg["chains"],
            target_accept=bayes_cfg["target_accept"],
            return_inferencedata=True,
            progressbar=True,
        )

    # Extract posterior summaries
    gap_posterior = trace.posterior["fairness_gap"].values.flatten()
    di_posterior = trace.posterior["di_ratio"].values.flatten()

    gap_mean = float(gap_posterior.mean())
    gap_hdi = az.hdi(trace, var_names=["fairness_gap"], hdi_prob=0.95)
    gap_lower = float(gap_hdi["fairness_gap"].values[0])
    gap_upper = float(gap_hdi["fairness_gap"].values[1])

    di_mean = float(di_posterior.mean())
    di_hdi = az.hdi(trace, var_names=["di_ratio"], hdi_prob=0.95)

    # Probability that gap > 0 (evidence of disparity)
    prob_gap_positive = float((gap_posterior > 0).mean())
    # Probability that DI ratio < 0.80
    prob_di_violation = float((di_posterior < 0.80).mean())

    logger.info(f"\nBayesian Posterior Results:")
    logger.info(f"  Fairness Gap ({group_a} - {group_b}):")
    logger.info(f"    Mean: {gap_mean:.4f}")
    logger.info(f"    95% HDI: [{gap_lower:.4f}, {gap_upper:.4f}]")
    logger.info(f"    P(gap > 0): {prob_gap_positive:.4f}")
    logger.info(f"  Disparate Impact Ratio:")
    logger.info(f"    Mean: {di_mean:.4f}")
    logger.info(f"    P(DI < 0.80): {prob_di_violation:.4f}")

    if prob_gap_positive > 0.95:
        logger.warning(f"  ⚠️  Strong evidence of fairness gap (P > 0.95)")

    return {
        "group_a": group_a,
        "group_b": group_b,
        "gap_mean": gap_mean,
        "gap_hdi_lower": gap_lower,
        "gap_hdi_upper": gap_upper,
        "gap_posterior": gap_posterior,
        "prob_gap_positive": prob_gap_positive,
        "di_mean": di_mean,
        "di_posterior": di_posterior,
        "prob_di_violation": prob_di_violation,
        "trace": trace,
    }


def _analytical_bayesian_gap(
    y_pred: np.ndarray,
    groups: np.ndarray,
    group_a: str,
    group_b: str,
) -> Dict:
    """
    Analytical Beta-Binomial approximation when PyMC is not available.

    Uses conjugate Beta posterior for each group's rate, then
    Monte Carlo sampling for the gap distribution.
    """
    logger.info("Using analytical Beta-Binomial approximation...")

    mask_a = groups == group_a
    mask_b = groups == group_b

    n_a, k_a = int(mask_a.sum()), int(y_pred[mask_a].sum())
    n_b, k_b = int(mask_b.sum()), int(y_pred[mask_b].sum())

    # Beta posterior parameters (uniform prior: alpha=1, beta=1)
    alpha_a, beta_a = 1 + k_a, 1 + (n_a - k_a)
    alpha_b, beta_b = 1 + k_b, 1 + (n_b - k_b)

    # Monte Carlo sampling from posterior
    rng = np.random.RandomState(42)
    n_samples = 10000
    rate_a_samples = rng.beta(alpha_a, beta_a, n_samples)
    rate_b_samples = rng.beta(alpha_b, beta_b, n_samples)

    gap_samples = rate_a_samples - rate_b_samples
    di_samples = rate_a_samples / np.maximum(rate_b_samples, 0.001)

    gap_mean = float(gap_samples.mean())
    gap_lower = float(np.percentile(gap_samples, 2.5))
    gap_upper = float(np.percentile(gap_samples, 97.5))
    prob_gap_positive = float((gap_samples > 0).mean())

    di_mean = float(di_samples.mean())
    prob_di_violation = float((di_samples < 0.80).mean())

    logger.info(f"\nAnalytical Bayesian Results:")
    logger.info(f"  Fairness Gap ({group_a} - {group_b}):")
    logger.info(f"    Mean: {gap_mean:.4f}")
    logger.info(f"    95% CI: [{gap_lower:.4f}, {gap_upper:.4f}]")
    logger.info(f"    P(gap > 0): {prob_gap_positive:.4f}")
    logger.info(f"  Disparate Impact Ratio mean: {di_mean:.4f}")
    logger.info(f"  P(DI < 0.80): {prob_di_violation:.4f}")

    return {
        "group_a": group_a,
        "group_b": group_b,
        "gap_mean": gap_mean,
        "gap_hdi_lower": gap_lower,
        "gap_hdi_upper": gap_upper,
        "gap_posterior": gap_samples,
        "prob_gap_positive": prob_gap_positive,
        "di_mean": di_mean,
        "di_posterior": di_samples,
        "prob_di_violation": prob_di_violation,
        "trace": None,
        "method": "analytical_beta_binomial",
    }


def bayesian_coefficient_stability(
    X: pd.DataFrame,
    y: pd.Series,
    feature_names: list,
    config: dict,
) -> Dict:
    """
    Bayesian logistic regression for coefficient uncertainty quantification.

    Estimates posterior distributions of model coefficients to identify
    which features have stable vs. uncertain effects.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix (standardized).
    y : pd.Series
        Binary target.
    feature_names : list
        Names of top features to analyze.
    config : dict
        Bayesian configuration.

    Returns
    -------
    Dict
        Posterior summaries for each coefficient.
    """
    try:
        import pymc as pm
        import arviz as az
    except ImportError:
        logger.warning("PyMC not installed. Skipping Bayesian coefficient stability.")
        return {"method": "skipped", "reason": "PyMC not installed"}

    logger.info("\nBayesian coefficient stability analysis...")

    # Use top N features to keep sampling tractable
    top_n = min(10, len(feature_names))
    features = feature_names[:top_n]
    X_sub = X[features].values if isinstance(X, pd.DataFrame) else X[:, :top_n]
    y_vals = y.values if hasattr(y, "values") else y

    # Standardize
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_sub)

    bayes_cfg = config["bayesian"]

    with pm.Model() as model:
        # Priors on coefficients
        intercept = pm.Normal("intercept", mu=0, sigma=2)
        betas = pm.Normal("betas", mu=0, sigma=1, shape=top_n)

        # Linear predictor
        mu = intercept + pm.math.dot(X_scaled, betas)

        # Likelihood
        obs = pm.Bernoulli("obs", logit_p=mu, observed=y_vals)

        # Sample
        trace = pm.sample(
            draws=bayes_cfg["samples"],
            tune=bayes_cfg["tune"],
            chains=bayes_cfg["chains"],
            target_accept=bayes_cfg["target_accept"],
            return_inferencedata=True,
            progressbar=True,
        )

    # Extract summaries
    summary = az.summary(trace, var_names=["betas"], hdi_prob=0.95)
    summary.index = features

    logger.info(f"\nCoefficient Posterior Summaries:")
    logger.info(summary.to_string())

    return {
        "summary": summary,
        "trace": trace,
        "features": features,
    }


def run_bayesian_audit(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    df_test: pd.DataFrame,
    X: Optional[pd.DataFrame],
    y: Optional[pd.Series],
    config: dict,
) -> Dict:
    """
    Run the complete Bayesian fairness analysis.

    Parameters
    ----------
    y_true : np.ndarray
        True test labels.
    y_pred : np.ndarray
        Predicted test labels.
    df_test : pd.DataFrame
        Test data with segment columns.
    X : pd.DataFrame, optional
        Full feature matrix (for coefficient stability).
    y : pd.Series, optional
        Full target vector.
    config : dict
        Configuration dictionary.

    Returns
    -------
    Dict
        Complete Bayesian audit results.
    """
    logger.info("\n" + "=" * 60)
    logger.info("BAYESIAN UNCERTAINTY QUANTIFICATION")
    logger.info("=" * 60)

    results = {}

    # Fairness gap analysis per protected attribute
    if "income_group" in df_test.columns:
        results["income_gap"] = bayesian_fairness_gap(
            y_true, y_pred, df_test["income_group"].values,
            group_a="Low", group_b="High", config=config,
        )

    if "state_region" in df_test.columns:
        results["geography_gap"] = bayesian_fairness_gap(
            y_true, y_pred, df_test["state_region"].values,
            group_a="South", group_b="Northeast", config=config,
        )

    # Coefficient stability (optional, requires full data)
    if X is not None and y is not None:
        top_features = list(X.columns[:10])
        results["coefficient_stability"] = bayesian_coefficient_stability(
            X, y, top_features, config
        )

    return results
