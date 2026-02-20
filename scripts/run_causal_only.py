#!/usr/bin/env python3
"""
Run Causal Analysis Only
========================

DiD and IV analysis standalone.

Usage:
    python scripts/run_causal_only.py
    python scripts/run_causal_only.py --use-sample-data
"""

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.data_loader import load_data, load_config
from src.data.feature_engineer import engineer_features
from src.causal.did_analysis import run_did_analysis
from src.causal.iv_analysis import run_iv_analysis

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Causal Analysis Only")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--use-sample-data", action="store_true")
    parser.add_argument("--skip-iv", action="store_true", help="Skip IV analysis")
    args = parser.parse_args()

    config = load_config(str(PROJECT_ROOT / args.config))
    if args.use_sample_data:
        config["data"]["raw_path"] = None

    # Load & engineer
    df = load_data(config)
    df = engineer_features(df, config)

    # DiD
    logger.info("Running Difference-in-Differences...")
    did_results = run_did_analysis(df, config)

    # IV
    if not args.skip_iv:
        logger.info("Running Instrumental Variables (2SLS)...")
        iv_results = run_iv_analysis(df, config)

    logger.info("Causal analysis complete.")


if __name__ == "__main__":
    main()
