"""
Training module for LightGBM entity resolution classifier.
Uses is_unbalance for class imbalance and group-aware train/val splitting.
"""

from pathlib import Path
from typing import List, Tuple
import lightgbm as lgb
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from ..config import MODEL_SAVE_PATH, RANDOM_STATE, N_JOBS

FEATURE_COLS = [
    "jw_sim",
    "token_sort",
    "token_set",
    "lev_ratio",
    "exact_match",
    "partial_ratio",
    "wratio",
    "token_exact",
    "name_jaccard",
    "trigram_jaccard",
    "name_len_ratio",
    "token_containment",
    "token_count_diff",
    "addr_token_sort",
    "addr_token_set",
    "addr_lev_ratio",
    "addr_jaccard",
    "addr_missing",
    "postal_match",
    "num_match",
    "combined_score",
    "name_addr_geom"
]


def train_lightgbm_model(
    df_features: pd.DataFrame,
    target_col: str = "label",
    model_path: Path = MODEL_SAVE_PATH,
    params: dict = None
) -> lgb.LGBMClassifier:
    """
    Trains a LightGBM classifier on extracted candidate pair features.
    Uses GroupShuffleSplit to prevent entity-level data leakage.
    """
    X = df_features[FEATURE_COLS]
    y = df_features[target_col]
    groups = df_features["source1_entity_id"]

    # Group-aware split: no S1 entity appears in both train and val
    gss = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=RANDOM_STATE)
    train_idx, val_idx = next(gss.split(X, y, groups=groups))
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    base_params = {
        "objective": "binary",
        "boosting_type": "gbdt",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "max_depth": 6,
        "n_estimators": 500,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "scale_pos_weight": 3.0,
        "n_jobs": N_JOBS,
        "random_state": RANDOM_STATE,
        "importance_type": "gain"
    }
    if params:
        base_params.update({k: v for k, v in params.items() if k != "name"})

    clf = lgb.LGBMClassifier(**base_params)

    clf.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)]
    )

    model_path.parent.mkdir(parents=True, exist_ok=True)
    clf.booster_.save_model(str(model_path))
    print(f"Model saved successfully to {model_path}")
    return clf


def load_lightgbm_model(model_path: Path = MODEL_SAVE_PATH) -> lgb.Booster:
    """
    Loads a saved LightGBM booster model.
    """
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found at {model_path}")
    return lgb.Booster(model_file=str(model_path))
