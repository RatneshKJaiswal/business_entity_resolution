"""
K-Fold Cross Validation and Hyperparameter Tuning for Entity Resolution.
Uses GroupKFold on source1_entity_id to ensure zero data leakage between folds.
"""

from pathlib import Path
from typing import Dict, List, Tuple, Any
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
from sklearn.metrics import log_loss, roc_auc_score
from tqdm import tqdm

from ..config import MODEL_SAVE_PATH, RANDOM_STATE, N_JOBS
from .train_lgbm import FEATURE_COLS

# Curated candidate parameter configurations for GBDT tuning
PARAM_GRID = [
    {
        "name": "Config-1 (Standard Balanced)",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "max_depth": 6,
        "subsample": 0.80,
        "colsample_bytree": 0.80,
        "min_child_samples": 20,
        "reg_alpha": 0.05,
        "reg_lambda": 0.10,
    },
    {
        "name": "Config-2 (Deeper & High Capacity)",
        "learning_rate": 0.04,
        "num_leaves": 45,
        "max_depth": 7,
        "subsample": 0.85,
        "colsample_bytree": 0.75,
        "min_child_samples": 30,
        "reg_alpha": 0.10,
        "reg_lambda": 0.50,
    },
    {
        "name": "Config-3 (Conservative / High Regularization)",
        "learning_rate": 0.05,
        "num_leaves": 25,
        "max_depth": 5,
        "subsample": 0.75,
        "colsample_bytree": 0.85,
        "min_child_samples": 50,
        "reg_alpha": 0.50,
        "reg_lambda": 1.00,
    },
    {
        "name": "Config-4 (Fast Gradient with Feature Subsampling)",
        "learning_rate": 0.07,
        "num_leaves": 31,
        "max_depth": 6,
        "subsample": 0.80,
        "colsample_bytree": 0.70,
        "min_child_samples": 25,
        "reg_alpha": 0.01,
        "reg_lambda": 0.10,
    },
    {
        "name": "Config-5 (Wide Trees & Heavy Subsample)",
        "learning_rate": 0.035,
        "num_leaves": 63,
        "max_depth": 8,
        "subsample": 0.70,
        "colsample_bytree": 0.80,
        "min_child_samples": 40,
        "reg_alpha": 0.20,
        "reg_lambda": 0.80,
    },
]


def evaluate_config_cv(
    df_features: pd.DataFrame,
    params: Dict[str, Any],
    n_splits: int = 5,
    target_col: str = "label",
    scale_pos_weight: float = 3.0
) -> Tuple[float, float]:
    """
    Evaluates a single hyperparameter configuration using GroupKFold cross-validation.
    Groups are defined by source1_entity_id to ensure zero entity leakage across folds.
    
    Returns:
        (mean_oof_logloss, mean_oof_auc)
    """
    X = df_features[FEATURE_COLS]
    y = df_features[target_col]
    groups = df_features["source1_entity_id"]

    gkf = GroupKFold(n_splits=n_splits)
    oof_predictions = np.zeros(len(df_features))

    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups=groups)):
        X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
        X_vl, y_vl = X.iloc[val_idx], y.iloc[val_idx]

        clf = lgb.LGBMClassifier(
            objective="binary",
            boosting_type="gbdt",
            n_estimators=500,
            scale_pos_weight=scale_pos_weight,
            n_jobs=N_JOBS,
            random_state=RANDOM_STATE + fold,
            importance_type="gain",
            **{k: v for k, v in params.items() if k != "name"}
        )

        clf.fit(
            X_tr,
            y_tr,
            eval_set=[(X_vl, y_vl)],
            callbacks=[lgb.early_stopping(stopping_rounds=20, verbose=False)]
        )

        oof_predictions[val_idx] = clf.predict_proba(X_vl)[:, 1]

    oof_loss = log_loss(y, oof_predictions)
    oof_auc = roc_auc_score(y, oof_predictions)
    return oof_loss, oof_auc


def run_kfold_hyperparameter_tuning(
    df_features: pd.DataFrame,
    n_splits: int = 5,
    n_trials: int = 4,
    target_col: str = "label",
    model_path: Path = MODEL_SAVE_PATH,
    scale_pos_weight: float = 3.0
) -> Tuple[lgb.LGBMClassifier, Dict[str, Any], Dict[str, float]]:
    """
    Runs K-Fold Cross Validation across candidate hyperparameter configurations,
    selects the top performing configuration, fits the final model, and saves it.
    """
    print(f"\n[K-Fold CV & Hyperparameter Tuning]")
    print(f"  • K-Fold Splits: {n_splits} folds (GroupKFold by source1_entity_id)")
    print(f"  • Configurations to Evaluate: {min(n_trials, len(PARAM_GRID))}")
    print(f"  • scale_pos_weight: {scale_pos_weight:.2f}")
    print(f"  • Total Samples: {len(df_features):,} pairs across {df_features['source1_entity_id'].nunique():,} unique S1 entities\n")

    candidates = PARAM_GRID[:n_trials]
    best_loss = float("inf")
    best_config = candidates[0]
    best_auc = 0.0
    cv_results = []

    for i, config in enumerate(candidates, 1):
        cfg_name = config["name"]
        print(f"  [{i}/{len(candidates)}] Testing {cfg_name}...")
        loss, auc = evaluate_config_cv(
            df_features=df_features,
            params=config,
            n_splits=n_splits,
            target_col=target_col,
            scale_pos_weight=scale_pos_weight
        )
        print(f"      -> Out-Of-Fold Log Loss: {loss:.5f} | ROC AUC: {auc:.5f}")
        cv_results.append({"name": cfg_name, "loss": loss, "auc": auc, "params": config})

        if loss < best_loss:
            best_loss = loss
            best_auc = auc
            best_config = config

    print(f"\n✓ Winning Hyperparameter Configuration: {best_config['name']}")
    print(f"  Best OOF Log Loss: {best_loss:.5f} | Best ROC AUC: {best_auc:.5f}")
    print(f"  Hyperparameters: { {k: v for k, v in best_config.items() if k != 'name'} }\n")

    # Fit final model on full feature set using the winning hyperparameters
    print(f"Fitting final LightGBM model on all training pairs using winning parameters...")
    X_full = df_features[FEATURE_COLS]
    y_full = df_features[target_col]

    final_clf = lgb.LGBMClassifier(
        objective="binary",
        boosting_type="gbdt",
        n_estimators=500,
        scale_pos_weight=scale_pos_weight,
        n_jobs=N_JOBS,
        random_state=RANDOM_STATE,
        importance_type="gain",
        **{k: v for k, v in best_config.items() if k != "name"}
    )
    final_clf.fit(X_full, y_full)

    model_path.parent.mkdir(parents=True, exist_ok=True)
    final_clf.booster_.save_model(str(model_path))
    print(f"Final tuned model saved to {model_path}")

    metrics = {
        "best_oof_logloss": best_loss,
        "best_oof_auc": best_auc,
        "n_splits": n_splits,
        "winning_config": best_config["name"]
    }
    return final_clf, best_config, metrics

