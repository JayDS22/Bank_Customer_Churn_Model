"""
Baseline Model Training for Consumer Lending Fairness Audit.

Trains Logistic Regression and XGBoost classifiers with stratified CV,
producing evaluation metrics and feature importances for audit.
"""

import logging
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from typing import Dict, Tuple, Optional

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score,
    recall_score, f1_score, classification_report,
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)


def train_logistic_regression(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    config: dict,
) -> Tuple[Pipeline, Dict]:
    """
    Train a Logistic Regression model with standardization.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training features.
    y_train : pd.Series
        Training labels.
    config : dict
        Model configuration.

    Returns
    -------
    Tuple[Pipeline, Dict]
        (trained pipeline, cross-validated metrics)
    """
    logger.info("Training Logistic Regression...")

    lr_config = config["model"]["logistic"]
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(
            C=lr_config["C"],
            max_iter=lr_config["max_iter"],
            class_weight=lr_config["class_weight"],
            solver=lr_config["solver"],
            random_state=config["data"]["random_state"],
        )),
    ])

    # Cross-validated predictions for fair evaluation
    cv = StratifiedKFold(
        n_splits=config["model"]["cv_folds"],
        shuffle=True,
        random_state=config["data"]["random_state"],
    )

    cv_probs = cross_val_predict(pipe, X_train, y_train, cv=cv, method="predict_proba")[:, 1]
    cv_preds = (cv_probs >= 0.5).astype(int)

    metrics = {
        "auc_roc": roc_auc_score(y_train, cv_probs),
        "accuracy": accuracy_score(y_train, cv_preds),
        "precision": precision_score(y_train, cv_preds, zero_division=0),
        "recall": recall_score(y_train, cv_preds, zero_division=0),
        "f1": f1_score(y_train, cv_preds, zero_division=0),
    }

    # Fit on full training data
    pipe.fit(X_train, y_train)

    logger.info(f"Logistic Regression CV AUC: {metrics['auc_roc']:.4f}")
    logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")
    logger.info(f"  Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}")

    return pipe, metrics


def train_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    config: dict,
) -> Tuple[object, Dict]:
    """
    Train an XGBoost classifier.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training features.
    y_train : pd.Series
        Training labels.
    config : dict
        Model configuration.

    Returns
    -------
    Tuple[object, Dict]
        (trained model, cross-validated metrics)
    """
    try:
        from xgboost import XGBClassifier
    except ImportError:
        logger.warning("XGBoost not installed. Skipping XGBoost model.")
        return None, {}

    logger.info("Training XGBoost...")

    xgb_config = config["model"]["xgboost"]

    # Auto-compute scale_pos_weight
    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    scale_pos = neg_count / max(pos_count, 1)

    model = XGBClassifier(
        n_estimators=xgb_config["n_estimators"],
        max_depth=xgb_config["max_depth"],
        learning_rate=xgb_config["learning_rate"],
        subsample=xgb_config["subsample"],
        colsample_bytree=xgb_config["colsample_bytree"],
        scale_pos_weight=scale_pos,
        eval_metric=xgb_config["eval_metric"],
        random_state=xgb_config["random_state"],
        use_label_encoder=False,
    )

    cv = StratifiedKFold(
        n_splits=config["model"]["cv_folds"],
        shuffle=True,
        random_state=config["data"]["random_state"],
    )

    cv_probs = cross_val_predict(model, X_train, y_train, cv=cv, method="predict_proba")[:, 1]
    cv_preds = (cv_probs >= 0.5).astype(int)

    metrics = {
        "auc_roc": roc_auc_score(y_train, cv_probs),
        "accuracy": accuracy_score(y_train, cv_preds),
        "precision": precision_score(y_train, cv_preds, zero_division=0),
        "recall": recall_score(y_train, cv_preds, zero_division=0),
        "f1": f1_score(y_train, cv_preds, zero_division=0),
    }

    model.fit(X_train, y_train)

    logger.info(f"XGBoost CV AUC: {metrics['auc_roc']:.4f}")
    logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")
    logger.info(f"  Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}")

    return model, metrics


def evaluate_on_test(
    model, X_test: pd.DataFrame, y_test: pd.Series, model_name: str = "Model"
) -> Dict:
    """
    Evaluate a trained model on the test set.

    Parameters
    ----------
    model : sklearn estimator
        Trained model or pipeline.
    X_test : pd.DataFrame
        Test features.
    y_test : pd.Series
        Test labels.
    model_name : str
        Name for logging.

    Returns
    -------
    Dict
        Test set metrics and prediction arrays.
    """
    probs = model.predict_proba(X_test)[:, 1]
    preds = (probs >= 0.5).astype(int)

    metrics = {
        "auc_roc": roc_auc_score(y_test, probs),
        "accuracy": accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds, zero_division=0),
        "recall": recall_score(y_test, preds, zero_division=0),
        "f1": f1_score(y_test, preds, zero_division=0),
        "probabilities": probs,
        "predictions": preds,
    }

    logger.info(f"\n{model_name} Test Set Results:")
    logger.info(f"  AUC-ROC: {metrics['auc_roc']:.4f}")
    logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")
    logger.info(f"  Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}")

    return metrics


def get_feature_importance(model, feature_names: list, model_type: str = "logistic") -> pd.DataFrame:
    """
    Extract feature importances from a trained model.

    Parameters
    ----------
    model : estimator
        Trained model (Pipeline or raw estimator).
    feature_names : list
        Feature column names.
    model_type : str
        'logistic' or 'xgboost'.

    Returns
    -------
    pd.DataFrame
        Sorted feature importance table.
    """
    if model_type == "logistic":
        # Extract from pipeline
        lr = model.named_steps["lr"] if hasattr(model, "named_steps") else model
        importance = np.abs(lr.coef_[0])
    elif model_type == "xgboost":
        importance = model.feature_importances_
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    fi = pd.DataFrame({
        "feature": feature_names[:len(importance)],
        "importance": importance,
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    return fi


def save_model(model, path: str) -> None:
    """Save model to disk."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    logger.info(f"Model saved to {path}")
