"""
Visualization Utilities for Lending Fairness Audit.

Generates publication-quality plots for fairness metrics, causal analysis,
and Bayesian posterior distributions.
"""

import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Optional, List
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.calibration import calibration_curve

logger = logging.getLogger(__name__)

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "figure.titlesize": 14,
})
COLORS = ["#2196F3", "#FF5722", "#4CAF50", "#9C27B0", "#FF9800", "#607D8B"]


def save_figure(fig, filename: str, output_dir: str) -> str:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    path = Path(output_dir) / filename
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {path}")
    return str(path)


def plot_fairness_dashboard(fairness_results: Dict, output_dir: str) -> List[str]:
    """Generate fairness metric bar charts for each protected attribute."""
    saved = []
    for attr, results in fairness_results.items():
        groups = results["groups"]
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f"Fairness Metrics by {attr}", fontsize=15, fontweight="bold")

        # 1. Demographic Parity
        ax = axes[0, 0]
        dp = results["demographic_parity"]
        bars = ax.bar([str(g) for g in groups], [dp[g] for g in groups],
                      color=COLORS[:len(groups)], edgecolor="white", linewidth=1.5)
        ax.set_title("Demographic Parity (Positive Prediction Rate)")
        ax.set_ylabel("P(Y_hat=1)")
        ax.set_ylim(0, 1)
        for bar, g in zip(bars, groups):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                    f"{dp[g]:.3f}", ha="center", va="bottom", fontsize=10)

        # 2. Equalized Odds
        ax = axes[0, 1]
        eo = results["equalized_odds"]
        x = np.arange(len(groups))
        w = 0.35
        tpr_vals = [eo[g]["tpr"] for g in groups]
        fpr_vals = [eo[g]["fpr"] for g in groups]
        ax.bar(x - w / 2, tpr_vals, w, label="TPR", color="#2196F3", edgecolor="white")
        ax.bar(x + w / 2, fpr_vals, w, label="FPR", color="#FF5722", edgecolor="white")
        ax.set_xticks(x)
        ax.set_xticklabels([str(g) for g in groups])
        ax.set_title("Equalized Odds (TPR & FPR)")
        ax.set_ylabel("Rate")
        ax.set_ylim(0, 1)
        ax.legend()

        # 3. Predictive Parity
        ax = axes[1, 0]
        pp = results["predictive_parity"]
        bars = ax.bar([str(g) for g in groups], [pp[g] for g in groups],
                      color=COLORS[:len(groups)], edgecolor="white", linewidth=1.5)
        ax.set_title("Predictive Parity (PPV)")
        ax.set_ylabel("Precision")
        ax.set_ylim(0, 1)
        for bar, g in zip(bars, groups):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                    f"{pp[g]:.3f}", ha="center", va="bottom", fontsize=10)

        # 4. Disparate Impact
        ax = axes[1, 1]
        di = results["disparate_impact"]
        pairs = list(di.keys())
        ratios = list(di.values())
        colors = ["#FF5722" if r < 0.80 else "#4CAF50" for r in ratios]
        bars = ax.barh(pairs, ratios, color=colors, edgecolor="white")
        ax.axvline(x=0.80, color="red", linestyle="--", linewidth=2, label="4/5 Rule (0.80)")
        ax.set_title("Disparate Impact Ratio")
        ax.set_xlabel("Ratio")
        ax.legend()
        for bar, ratio in zip(bars, ratios):
            ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                    f"{ratio:.3f}", va="center", fontsize=10)

        plt.tight_layout()
        path = save_figure(fig, f"fairness_dashboard_{attr}.png", output_dir)
        saved.append(path)
    return saved


def plot_roc_by_group(y_true, y_probs, groups, group_name, output_dir):
    """Plot overlaid ROC curves per group."""
    fig, ax = plt.subplots(figsize=(8, 8))
    for i, g in enumerate(sorted(np.unique(groups))):
        mask = groups == g
        if mask.sum() < 10:
            continue
        fpr, tpr, _ = roc_curve(y_true[mask], y_probs[mask])
        auc = roc_auc_score(y_true[mask], y_probs[mask])
        ax.plot(fpr, tpr, color=COLORS[i % len(COLORS)], linewidth=2,
                label=f"{g} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curves by {group_name}")
    ax.legend(loc="lower right")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    return save_figure(fig, f"roc_by_{group_name}.png", output_dir)


def plot_calibration_by_group(y_true, y_probs, groups, group_name, output_dir):
    """Plot calibration curves per group."""
    fig, ax = plt.subplots(figsize=(8, 8))
    for i, g in enumerate(sorted(np.unique(groups))):
        mask = groups == g
        if mask.sum() < 50:
            continue
        prob_true, prob_pred = calibration_curve(y_true[mask], y_probs[mask],
                                                  n_bins=10, strategy="uniform")
        ax.plot(prob_pred, prob_true, marker="o", color=COLORS[i % len(COLORS)],
                linewidth=2, label=str(g))
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5, label="Perfectly calibrated")
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction of Positives")
    ax.set_title(f"Calibration Curves by {group_name}")
    ax.legend()
    return save_figure(fig, f"calibration_by_{group_name}.png", output_dir)


def plot_did_results(did_data, did_results, output_dir):
    """Plot DiD parallel trends and treatment effect."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax = axes[0]
    if "issue_year" in did_data.columns:
        trends = did_data.groupby(["issue_year", "treated"])["default"].mean().reset_index()
        for treated_val, label, color in [(0, "Control (High Income)", "#2196F3"),
                                           (1, "Treatment (Low Income)", "#FF5722")]:
            subset = trends[trends["treated"] == treated_val]
            ax.plot(subset["issue_year"], subset["default"],
                    marker="o", linewidth=2, color=color, label=label)
        ax.axvline(x=2015, color="gray", linestyle="--", linewidth=2, alpha=0.7,
                   label="Policy Change")
    ax.set_xlabel("Year")
    ax.set_ylabel("Default Rate")
    ax.set_title("Parallel Trends: Default Rate Over Time")
    ax.legend()

    ax = axes[1]
    did_info = did_results.get("did", {})
    did_coef = did_info.get("did_coefficient", 0)
    did_ci_l = did_info.get("did_ci_lower", 0)
    did_ci_u = did_info.get("did_ci_upper", 0)
    ax.barh(["DiD Effect\n(Post x Treated)"], [did_coef],
            xerr=[[did_coef - did_ci_l], [did_ci_u - did_coef]],
            color="#FF5722" if did_coef > 0 else "#4CAF50",
            edgecolor="white", capsize=10, height=0.4)
    ax.axvline(x=0, color="black", linewidth=1)
    ax.set_xlabel("Coefficient (effect on default probability)")
    ax.set_title("DiD Treatment Effect with 95% CI")

    plt.tight_layout()
    return save_figure(fig, "did_analysis.png", output_dir)


def plot_bayesian_posterior(bayesian_results, output_dir):
    """Plot Bayesian posterior distributions for fairness gaps."""
    saved = []
    for key, result in bayesian_results.items():
        if "gap_posterior" not in result:
            continue

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Fairness gap posterior
        ax = axes[0]
        gap = result["gap_posterior"]
        ax.hist(gap, bins=50, density=True, color="#2196F3", alpha=0.7, edgecolor="white")
        ax.axvline(x=0, color="red", linestyle="--", linewidth=2, label="No Gap")
        ax.axvline(x=result["gap_mean"], color="black", linewidth=2,
                   label=f"Mean={result['gap_mean']:.4f}")
        hdi_l = result["gap_hdi_lower"]
        hdi_u = result["gap_hdi_upper"]
        ax.axvspan(hdi_l, hdi_u, alpha=0.15, color="blue",
                   label=f"95% HDI [{hdi_l:.4f}, {hdi_u:.4f}]")
        ax.set_xlabel("Fairness Gap")
        ax.set_ylabel("Density")
        ax.set_title(f"Posterior: {result['group_a']} vs {result['group_b']} Gap")
        ax.legend(fontsize=9)

        # DI ratio posterior
        ax = axes[1]
        di = result["di_posterior"]
        di_clipped = di[di < 3]
        ax.hist(di_clipped, bins=50, density=True, color="#4CAF50", alpha=0.7, edgecolor="white")
        ax.axvline(x=0.80, color="red", linestyle="--", linewidth=2, label="4/5 Rule (0.80)")
        ax.axvline(x=result["di_mean"], color="black", linewidth=2,
                   label=f"Mean={result['di_mean']:.4f}")
        ax.set_xlabel("Disparate Impact Ratio")
        ax.set_ylabel("Density")
        ax.set_title("Posterior: Disparate Impact Ratio")
        ax.legend(fontsize=9)

        prob_violation = result["prob_di_violation"]
        ax.text(0.05, 0.95, f"P(DI < 0.80) = {prob_violation:.3f}",
                transform=ax.transAxes, fontsize=11, verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

        plt.tight_layout()
        path = save_figure(fig, f"bayesian_posterior_{key}.png", output_dir)
        saved.append(path)
    return saved


def plot_feature_importance_with_proxy_flags(
    feature_importance: pd.DataFrame,
    proxy_features: List[str],
    output_dir: str,
) -> str:
    """
    Plot feature importance with proxy features highlighted.

    Parameters
    ----------
    feature_importance : pd.DataFrame
        DataFrame with 'feature' and 'importance' columns.
    proxy_features : list
        Feature names flagged as proxies.
    output_dir : str
        Directory to save plot.

    Returns
    -------
    str
        Path to saved figure.
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    top_n = min(15, len(feature_importance))
    fi = feature_importance.head(top_n).copy()
    fi = fi.sort_values("importance", ascending=True)

    colors = ["#FF5722" if f in proxy_features else "#2196F3" for f in fi["feature"]]
    ax.barh(fi["feature"], fi["importance"], color=colors, edgecolor="white")
    ax.set_xlabel("Feature Importance")
    ax.set_title("Feature Importance with Proxy Flags")

    import matplotlib.patches as mpatches
    causal_patch = mpatches.Patch(color="#2196F3", label="Causal Feature")
    proxy_patch = mpatches.Patch(color="#FF5722", label="Proxy Feature (bias risk)")
    ax.legend(handles=[causal_patch, proxy_patch], loc="lower right")

    return save_figure(fig, "feature_importance_proxy_flags.png", output_dir)


def plot_disparate_impact_heatmap(fairness_results: Dict, output_dir: str) -> str:
    """Plot heatmap of disparate impact ratios across all segment pairs."""
    fig, ax = plt.subplots(figsize=(10, 8))

    all_ratios = {}
    for attr, results in fairness_results.items():
        for pair, ratio in results["disparate_impact"].items():
            all_ratios[f"{attr}: {pair}"] = ratio

    if not all_ratios:
        logger.warning("No disparate impact data to plot.")
        plt.close(fig)
        return ""

    pairs = list(all_ratios.keys())
    ratios = list(all_ratios.values())

    colors = ["#FF5722" if r < 0.80 else "#4CAF50" for r in ratios]
    bars = ax.barh(pairs, ratios, color=colors, edgecolor="white", height=0.6)
    ax.axvline(x=0.80, color="red", linestyle="--", linewidth=2, label="4/5 Rule Threshold")
    ax.axvline(x=1.0, color="gray", linestyle=":", linewidth=1, alpha=0.5)
    ax.set_xlabel("Disparate Impact Ratio")
    ax.set_title("Disparate Impact Across All Segment Pairs")
    ax.legend()

    for bar, ratio in zip(bars, ratios):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{ratio:.3f}", va="center", fontsize=10)

    plt.tight_layout()
    return save_figure(fig, "disparate_impact_heatmap.png", output_dir)


def generate_all_plots(
    fairness_results: Dict,
    y_true: np.ndarray,
    y_probs: np.ndarray,
    df_test: pd.DataFrame,
    did_results: Optional[Dict],
    bayesian_results: Optional[Dict],
    feature_importance: Optional[pd.DataFrame],
    proxy_features: List[str],
    output_dir: str,
) -> List[str]:
    """Generate all audit plots and return file paths."""
    all_paths = []

    # Fairness dashboard
    all_paths.extend(plot_fairness_dashboard(fairness_results, output_dir))

    # ROC and calibration by group
    for attr in fairness_results:
        if attr in df_test.columns:
            groups = df_test[attr].values
            all_paths.append(plot_roc_by_group(y_true, y_probs, groups, attr, output_dir))
            all_paths.append(plot_calibration_by_group(y_true, y_probs, groups, attr, output_dir))

    # Disparate impact heatmap
    path = plot_disparate_impact_heatmap(fairness_results, output_dir)
    if path:
        all_paths.append(path)

    # DiD
    if did_results and "data" in did_results:
        all_paths.append(plot_did_results(did_results["data"], did_results, output_dir))

    # Bayesian posteriors
    if bayesian_results:
        all_paths.extend(plot_bayesian_posterior(bayesian_results, output_dir))

    # Feature importance
    if feature_importance is not None:
        all_paths.append(
            plot_feature_importance_with_proxy_flags(feature_importance, proxy_features, output_dir)
        )

    logger.info(f"\nGenerated {len(all_paths)} plots in {output_dir}")
    return all_paths
