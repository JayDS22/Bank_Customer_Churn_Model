#!/usr/bin/env python3
"""
Run Fairness Metrics Only
=========================

Quick fairness audit without causal or Bayesian analysis.

Usage:
    python scripts/run_fairness_only.py
    python scripts/run_fairness_only.py --use-sample-data
"""

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.data_loader import load_data, load_config, train_test_split_data
from src.data.feature_engineer import engineer_features, get_model_features
from src.models.baseline_model import train_logistic_regression, evaluate_on_test
from src.fairness.fairness_metrics import run_fairness_audit
from src.visualization.plot_utils import plot_fairness_dashboard, plot_disparate_impact_heatmap

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Fairness Metrics Only")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--use-sample-data", action="store_true")
    args = parser.parse_args()

    config = load_config(str(PROJECT_ROOT / args.config))
    if args.use_sample_data:
        config["data"]["raw_path"] = None

    output_dir = config["output"]["plots_dir"]
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Load & engineer
    df = load_data(config)
    df = engineer_features(df, config)
    train_df, test_df = train_test_split_data(df, config)

    # Train model
    X_train, y_train = get_model_features(train_df, config)
    X_test, y_test = get_model_features(test_df, config)
    model, _ = train_logistic_regression(X_train, y_train, config)
    test_metrics = evaluate_on_test(model, X_test, y_test, "Logistic Regression")

    # Fairness audit
    fairness_results = run_fairness_audit(
        y_test.values, test_metrics["predictions"], test_df, config
    )

    # Plots
    plot_fairness_dashboard(fairness_results, output_dir)
    plot_disparate_impact_heatmap(fairness_results, output_dir)

    logger.info("Fairness audit complete.")


if __name__ == "__main__":
    main()
