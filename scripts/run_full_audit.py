#!/usr/bin/env python3
"""
Run Full Audit Pipeline
=======================

End-to-end consumer lending model fairness audit:
1. Load/generate data
2. Feature engineering
3. Train baseline models
4. Run fairness metrics
5. Causal inference (DiD + IV)
6. Bayesian uncertainty quantification
7. Generate visualizations
8. Produce recommendations report

Usage:
    python scripts/run_full_audit.py
    python scripts/run_full_audit.py --use-sample-data
    python scripts/run_full_audit.py --config config/config.yaml --skip-bayesian
"""

import argparse
import logging
import sys
import json
import pandas as pd
from pathlib import Path
from datetime import datetime

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.data_loader import load_data, load_config, train_test_split_data
from src.data.feature_engineer import engineer_features, get_model_features
from src.models.baseline_model import (
    train_logistic_regression, train_xgboost,
    evaluate_on_test, get_feature_importance, save_model,
)
from src.fairness.fairness_metrics import run_fairness_audit
from src.causal.did_analysis import run_did_analysis
from src.causal.iv_analysis import run_iv_analysis
from src.bayesian.bayesian_audit import run_bayesian_audit
from src.visualization.plot_utils import generate_all_plots

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("outputs/audit_log.txt", mode="w"),
    ],
)
logger = logging.getLogger(__name__)


def generate_recommendations(
    fairness_results: dict,
    causal_results: dict,
    bayesian_results: dict,
    feature_importance,
    proxy_features: list,
) -> str:
    """Generate written recommendations based on audit findings."""
    lines = []
    lines.append("=" * 70)
    lines.append("CONSUMER LENDING MODEL AUDIT — RECOMMENDATIONS REPORT")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * 70)

    # Fairness findings
    lines.append("\n1. FAIRNESS FINDINGS")
    lines.append("-" * 40)
    total_violations = 0
    for attr, results in fairness_results.items():
        n_violations = len(results["violations"])
        total_violations += n_violations
        lines.append(f"\n  {attr}: {n_violations} violation(s)")
        for v in results["violations"]:
            lines.append(f"    - {v}")
        lines.append(f"    Demographic Parity gap: {results['demographic_parity_gap']:.4f}")
        lines.append(f"    TPR gap: {results['tpr_gap']:.4f}")
        lines.append(f"    FPR gap: {results['fpr_gap']:.4f}")

    # Causal findings
    lines.append("\n2. CAUSAL INFERENCE FINDINGS")
    lines.append("-" * 40)
    if "did" in causal_results:
        did = causal_results["did"].get("did", {})
        lines.append(f"  DiD estimate: {did.get('did_coefficient', 'N/A'):.4f}")
        lines.append(f"  DiD p-value: {did.get('did_pvalue', 'N/A'):.4f}")
    if "iv" in causal_results:
        iv = causal_results["iv"].get("iv", {})
        lines.append(f"  IV income coefficient: {iv.get('income_coefficient', 'N/A'):.4f}")
        lines.append(f"  IV income p-value: {iv.get('income_pvalue', 'N/A'):.4f}")

    # Proxy features
    lines.append("\n3. PROXY FEATURE ANALYSIS")
    lines.append("-" * 40)
    if proxy_features:
        lines.append(f"  {len(proxy_features)} feature(s) flagged as potential proxies:")
        for f in proxy_features:
            lines.append(f"    - {f}")
        lines.append("  These features correlate with protected attributes but may not")
        lines.append("  causally predict default. Consider removing or debiasing.")
    else:
        lines.append("  No proxy features detected.")

    # Bayesian findings
    lines.append("\n4. BAYESIAN UNCERTAINTY")
    lines.append("-" * 40)
    for key, result in bayesian_results.items():
        if "gap_mean" in result:
            lines.append(f"  {key}:")
            lines.append(f"    Gap mean: {result['gap_mean']:.4f}")
            lines.append(f"    95% HDI: [{result['gap_hdi_lower']:.4f}, {result['gap_hdi_upper']:.4f}]")
            lines.append(f"    P(gap > 0): {result['prob_gap_positive']:.4f}")
            lines.append(f"    P(DI < 0.80): {result['prob_di_violation']:.4f}")

    # Recommendations
    lines.append("\n5. RECOMMENDATIONS")
    lines.append("-" * 40)
    rec_num = 1

    if total_violations > 0:
        lines.append(f"  {rec_num}. RECALIBRATE MODEL THRESHOLDS")
        lines.append("     Apply group-specific threshold optimization to equalize FPR/TPR")
        lines.append("     across income and geographic segments while maintaining overall AUC.")
        rec_num += 1

    if proxy_features:
        lines.append(f"  {rec_num}. REMOVE OR DEBIAS PROXY FEATURES")
        lines.append(f"     Features {proxy_features} should be:")
        lines.append("     a) Removed if they don't improve AUC by > 0.01")
        lines.append("     b) Residualized against protected attributes if retained")
        rec_num += 1

    lines.append(f"  {rec_num}. IMPLEMENT FAIRNESS CONSTRAINTS IN TRAINING")
    lines.append("     Use constrained optimization (e.g., fairlearn) to enforce")
    lines.append("     equalized odds or demographic parity during model training.")
    rec_num += 1

    lines.append(f"  {rec_num}. ESTABLISH ONGOING MONITORING")
    lines.append("     Deploy fairness metric dashboards and set alert thresholds")
    lines.append("     for disparate impact ratio < 0.80 on all production models.")
    rec_num += 1

    lines.append(f"  {rec_num}. CONDUCT PERIODIC CAUSAL AUDITS")
    lines.append("     Repeat DiD/IV analysis quarterly to detect drift in causal")
    lines.append("     relationships between features and outcomes.")
    rec_num += 1

    lines.append("\n" + "=" * 70)
    lines.append("END OF REPORT")
    lines.append("=" * 70)

    return "\n".join(lines)


def identify_proxy_features(
    feature_importance, df, config, top_n: int = 12
) -> list:
    """
    Identify features that correlate with protected attributes
    more strongly than they correlate with the outcome.
    """
    proxies = []
    if feature_importance is None or feature_importance.empty:
        return proxies

    top_features = feature_importance.head(top_n)["feature"].tolist()

    for feat in top_features:
        if feat not in df.columns:
            continue
        if not pd.api.types.is_numeric_dtype(df[feat]):
            continue

        # Correlation with income
        if "annual_inc" in df.columns:
            corr_income = abs(df[feat].corr(df["annual_inc"]))
        else:
            corr_income = 0

        # Correlation with outcome
        if "default" in df.columns:
            corr_default = abs(df[feat].corr(df["default"]))
        else:
            corr_default = 0

        # Flag if more correlated with protected attr than outcome
        if corr_income > corr_default * 1.5 and corr_income > 0.1:
            proxies.append(feat)

    logger.info(f"Proxy features identified: {proxies}")
    return proxies


def main():
    parser = argparse.ArgumentParser(description="Consumer Lending Fairness Audit")
    parser.add_argument("--config", default="config/config.yaml", help="Config file path")
    parser.add_argument("--use-sample-data", action="store_true", help="Use synthetic data")
    parser.add_argument("--skip-bayesian", action="store_true", help="Skip Bayesian analysis")
    parser.add_argument("--skip-causal", action="store_true", help="Skip causal analysis")
    args = parser.parse_args()

    logger.info("=" * 70)
    logger.info("CONSUMER LENDING MODEL AUDIT & FAIRNESS ANALYSIS")
    logger.info("=" * 70)

    # Load config
    config_path = PROJECT_ROOT / args.config
    config = load_config(str(config_path))
    output_dir = config["output"]["plots_dir"]
    reports_dir = config["output"]["reports_dir"]

    if args.use_sample_data:
        config["data"]["raw_path"] = None

    # Ensure output dirs exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    Path(reports_dir).mkdir(parents=True, exist_ok=True)
    Path("outputs").mkdir(parents=True, exist_ok=True)

    # ========== STEP 1: Load Data ==========
    logger.info("\n[STEP 1/7] Loading data...")
    df = load_data(config)
    logger.info(f"Loaded {df.shape[0]} records with {df.shape[1]} columns")

    # ========== STEP 2: Feature Engineering ==========
    logger.info("\n[STEP 2/7] Engineering features...")
    df = engineer_features(df, config)

    # ========== STEP 3: Train/Test Split + Model Training ==========
    logger.info("\n[STEP 3/7] Training baseline models...")
    train_df, test_df = train_test_split_data(df, config)

    X_train, y_train = get_model_features(train_df, config)
    X_test, y_test = get_model_features(test_df, config)

    # Logistic Regression
    lr_model, lr_cv_metrics = train_logistic_regression(X_train, y_train, config)
    lr_test_metrics = evaluate_on_test(lr_model, X_test, y_test, "Logistic Regression")

    # XGBoost
    xgb_model, xgb_cv_metrics = train_xgboost(X_train, y_train, config)
    if xgb_model:
        xgb_test_metrics = evaluate_on_test(xgb_model, X_test, y_test, "XGBoost")

    # Feature importance (use logistic regression for interpretability)
    feature_importance = get_feature_importance(lr_model, list(X_train.columns), "logistic")
    logger.info(f"\nTop 10 features:\n{feature_importance.head(10).to_string()}")

    # Save models
    save_model(lr_model, "outputs/models/logistic_regression.pkl")
    if xgb_model:
        save_model(xgb_model, "outputs/models/xgboost.pkl")

    # ========== STEP 4: Fairness Audit ==========
    logger.info("\n[STEP 4/7] Running fairness audit...")
    y_pred = lr_test_metrics["predictions"]
    y_probs = lr_test_metrics["probabilities"]

    fairness_results = run_fairness_audit(
        y_test.values, y_pred, test_df, config
    )

    # ========== STEP 5: Causal Inference ==========
    causal_results = {}
    if not args.skip_causal:
        logger.info("\n[STEP 5/7] Running causal analysis...")
        try:
            did_results = run_did_analysis(df, config)
            causal_results["did"] = did_results
        except Exception as e:
            logger.warning(f"DiD analysis failed: {e}")

        try:
            iv_results = run_iv_analysis(df, config)
            causal_results["iv"] = iv_results
        except Exception as e:
            logger.warning(f"IV analysis failed: {e}")
    else:
        logger.info("\n[STEP 5/7] Skipping causal analysis (--skip-causal)")

    # ========== STEP 6: Bayesian Analysis ==========
    bayesian_results = {}
    if not args.skip_bayesian:
        logger.info("\n[STEP 6/7] Running Bayesian analysis...")
        try:
            bayesian_results = run_bayesian_audit(
                y_test.values, y_pred, test_df,
                X_train, y_train, config
            )
        except Exception as e:
            logger.warning(f"Bayesian analysis failed: {e}")
    else:
        logger.info("\n[STEP 6/7] Skipping Bayesian analysis (--skip-bayesian)")

    # ========== STEP 7: Visualizations & Report ==========
    logger.info("\n[STEP 7/7] Generating visualizations and report...")

    # Identify proxy features
    proxy_features = identify_proxy_features(feature_importance, df, config)

    # Generate all plots
    did_for_plots = causal_results.get("did")
    plot_paths = generate_all_plots(
        fairness_results, y_test.values, y_probs, test_df,
        did_for_plots, bayesian_results,
        feature_importance, proxy_features, output_dir,
    )

    # Generate recommendations report
    report = generate_recommendations(
        fairness_results, causal_results, bayesian_results,
        feature_importance, proxy_features,
    )

    report_path = Path(reports_dir) / "audit_recommendations.txt"
    with open(report_path, "w") as f:
        f.write(report)
    logger.info(f"Report saved to {report_path}")

    # Print report to console
    print("\n" + report)

    # Save metrics summary as JSON
    metrics_summary = {
        "logistic_regression": {k: v for k, v in lr_test_metrics.items()
                                 if k not in ["probabilities", "predictions"]},
        "xgboost": {k: v for k, v in (xgb_test_metrics if xgb_model else {}).items()
                     if k not in ["probabilities", "predictions"]},
        "fairness_violations": {
            attr: len(r["violations"]) for attr, r in fairness_results.items()
        },
        "proxy_features": proxy_features,
    }
    metrics_path = Path(reports_dir) / "metrics_summary.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics_summary, f, indent=2, default=str)

    logger.info(f"\nAudit complete. {len(plot_paths)} plots saved to {output_dir}")
    logger.info(f"Report saved to {reports_dir}")


if __name__ == "__main__":
    main()
