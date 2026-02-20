"""
Fairness Metrics for Consumer Lending Model Audit.

Computes demographic parity, equalized odds, predictive parity,
and disparate impact ratio across protected attribute segments.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional

logger = logging.getLogger(__name__)


def demographic_parity(
    y_pred: np.ndarray,
    groups: np.ndarray,
) -> Dict[str, float]:
    """
    Compute demographic parity: P(Ŷ=1 | G=g) for each group.

    A fair model has equal positive prediction rates across groups.

    Parameters
    ----------
    y_pred : np.ndarray
        Predicted labels (0/1).
    groups : np.ndarray
        Group membership labels.

    Returns
    -------
    Dict[str, float]
        Positive prediction rate per group.
    """
    result = {}
    for g in np.unique(groups):
        mask = groups == g
        rate = y_pred[mask].mean()
        result[g] = float(rate)

    return result


def equalized_odds(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    groups: np.ndarray,
) -> Dict[str, Dict[str, float]]:
    """
    Compute equalized odds: TPR and FPR per group.

    A fair model has equal TPR and FPR across groups.

    Parameters
    ----------
    y_true : np.ndarray
        True labels.
    y_pred : np.ndarray
        Predicted labels.
    groups : np.ndarray
        Group membership labels.

    Returns
    -------
    Dict[str, Dict[str, float]]
        {group: {"tpr": ..., "fpr": ...}}
    """
    result = {}
    for g in np.unique(groups):
        mask = groups == g
        y_t = y_true[mask]
        y_p = y_pred[mask]

        # True Positive Rate (Recall)
        pos_mask = y_t == 1
        tpr = y_p[pos_mask].mean() if pos_mask.sum() > 0 else 0.0

        # False Positive Rate
        neg_mask = y_t == 0
        fpr = y_p[neg_mask].mean() if neg_mask.sum() > 0 else 0.0

        result[g] = {"tpr": float(tpr), "fpr": float(fpr)}

    return result


def predictive_parity(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    groups: np.ndarray,
) -> Dict[str, float]:
    """
    Compute predictive parity: PPV (precision) per group.

    A fair model has equal precision across groups.

    Parameters
    ----------
    y_true : np.ndarray
        True labels.
    y_pred : np.ndarray
        Predicted labels.
    groups : np.ndarray
        Group membership labels.

    Returns
    -------
    Dict[str, float]
        Precision per group.
    """
    result = {}
    for g in np.unique(groups):
        mask = groups == g
        y_t = y_true[mask]
        y_p = y_pred[mask]

        pred_pos = y_p == 1
        if pred_pos.sum() > 0:
            ppv = y_t[pred_pos].mean()
        else:
            ppv = 0.0

        result[g] = float(ppv)

    return result


def disparate_impact_ratio(
    y_pred: np.ndarray,
    groups: np.ndarray,
    privileged_group: str,
    unprivileged_group: str,
) -> float:
    """
    Compute disparate impact ratio: P(Ŷ=1|G=unprivileged) / P(Ŷ=1|G=privileged).

    The 4/5 (80%) rule: ratio should be ≥ 0.80.

    Parameters
    ----------
    y_pred : np.ndarray
        Predicted labels (0/1).
    groups : np.ndarray
        Group membership labels.
    privileged_group : str
        Label of the privileged group.
    unprivileged_group : str
        Label of the unprivileged group.

    Returns
    -------
    float
        Disparate impact ratio.
    """
    priv_mask = groups == privileged_group
    unpriv_mask = groups == unprivileged_group

    priv_rate = y_pred[priv_mask].mean() if priv_mask.sum() > 0 else 0.0
    unpriv_rate = y_pred[unpriv_mask].mean() if unpriv_mask.sum() > 0 else 0.0

    if priv_rate == 0:
        return float("inf") if unpriv_rate > 0 else 1.0

    ratio = unpriv_rate / priv_rate
    return float(ratio)


def compute_all_fairness_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    groups: np.ndarray,
    group_name: str,
    config: dict,
) -> Dict:
    """
    Compute all fairness metrics for a given protected attribute.

    Parameters
    ----------
    y_true : np.ndarray
        True labels.
    y_pred : np.ndarray
        Predicted labels (0/1).
    groups : np.ndarray
        Group membership labels.
    group_name : str
        Name of the attribute being audited (for logging).
    config : dict
        Configuration with fairness thresholds.

    Returns
    -------
    Dict
        Complete fairness audit results.
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"FAIRNESS AUDIT: {group_name}")
    logger.info(f"{'='*60}")

    unique_groups = sorted(np.unique(groups))

    # --- Demographic Parity ---
    dp = demographic_parity(y_pred, groups)
    dp_values = list(dp.values())
    dp_gap = max(dp_values) - min(dp_values)

    logger.info(f"\n1. Demographic Parity (positive prediction rate):")
    for g, rate in sorted(dp.items()):
        logger.info(f"   {g}: {rate:.4f}")
    logger.info(f"   Gap: {dp_gap:.4f} (threshold: {config['fairness']['demographic_parity_threshold']})")

    # --- Equalized Odds ---
    eo = equalized_odds(y_true, y_pred, groups)
    tpr_values = [v["tpr"] for v in eo.values()]
    fpr_values = [v["fpr"] for v in eo.values()]
    tpr_gap = max(tpr_values) - min(tpr_values)
    fpr_gap = max(fpr_values) - min(fpr_values)

    logger.info(f"\n2. Equalized Odds:")
    for g, rates in sorted(eo.items()):
        logger.info(f"   {g}: TPR={rates['tpr']:.4f}, FPR={rates['fpr']:.4f}")
    logger.info(f"   TPR gap: {tpr_gap:.4f}, FPR gap: {fpr_gap:.4f}")

    # --- Predictive Parity ---
    pp = predictive_parity(y_true, y_pred, groups)
    pp_values = list(pp.values())
    pp_gap = max(pp_values) - min(pp_values) if pp_values else 0

    logger.info(f"\n3. Predictive Parity (PPV):")
    for g, ppv in sorted(pp.items()):
        logger.info(f"   {g}: {ppv:.4f}")
    logger.info(f"   Gap: {pp_gap:.4f}")

    # --- Disparate Impact ---
    di_results = {}
    logger.info(f"\n4. Disparate Impact Ratio:")
    for i, g1 in enumerate(unique_groups):
        for g2 in unique_groups[i + 1:]:
            ratio = disparate_impact_ratio(y_pred, groups, g2, g1)
            di_results[f"{g1}_vs_{g2}"] = ratio
            flag = "⚠️  BELOW 4/5 RULE" if ratio < config["fairness"]["disparate_impact_ratio_threshold"] else "✓"
            logger.info(f"   {g1} vs {g2}: {ratio:.4f} {flag}")

    # --- Summary ---
    flags = []
    if dp_gap > config["fairness"]["demographic_parity_threshold"]:
        flags.append("Demographic Parity violation")
    if tpr_gap > config["fairness"]["equalized_odds_threshold"]:
        flags.append("Equalized Odds (TPR) violation")
    if fpr_gap > config["fairness"]["equalized_odds_threshold"]:
        flags.append("Equalized Odds (FPR) violation")
    for key, ratio in di_results.items():
        if ratio < config["fairness"]["disparate_impact_ratio_threshold"]:
            flags.append(f"Disparate Impact violation ({key})")

    if flags:
        logger.warning(f"\n⚠️  FAIRNESS VIOLATIONS DETECTED for {group_name}:")
        for f in flags:
            logger.warning(f"   - {f}")
    else:
        logger.info(f"\n✓ No fairness violations detected for {group_name}")

    return {
        "group_name": group_name,
        "groups": unique_groups,
        "demographic_parity": dp,
        "demographic_parity_gap": dp_gap,
        "equalized_odds": eo,
        "tpr_gap": tpr_gap,
        "fpr_gap": fpr_gap,
        "predictive_parity": pp,
        "predictive_parity_gap": pp_gap,
        "disparate_impact": di_results,
        "violations": flags,
    }


def run_fairness_audit(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    df_test: pd.DataFrame,
    config: dict,
) -> Dict[str, Dict]:
    """
    Run the complete fairness audit across all protected attributes.

    Parameters
    ----------
    y_true : np.ndarray
        True labels.
    y_pred : np.ndarray
        Predicted labels.
    df_test : pd.DataFrame
        Test dataset with segment columns.
    config : dict
        Configuration dictionary.

    Returns
    -------
    Dict[str, Dict]
        Fairness results keyed by attribute name.
    """
    results = {}
    protected = config["fairness"]["protected_attributes"]

    for attr in protected:
        if attr in df_test.columns:
            groups = df_test[attr].values
            results[attr] = compute_all_fairness_metrics(
                y_true, y_pred, groups, attr, config
            )
        else:
            logger.warning(f"Protected attribute '{attr}' not found in data.")

    # Summary across all attributes
    total_violations = sum(len(r["violations"]) for r in results.values())
    logger.info(f"\n{'='*60}")
    logger.info(f"AUDIT SUMMARY: {total_violations} total violation(s) across {len(results)} attribute(s)")
    logger.info(f"{'='*60}")

    return results
