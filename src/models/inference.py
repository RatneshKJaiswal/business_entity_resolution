"""
Inference module for scoring candidate pairs with trained model.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb

from ..config import MODEL_SAVE_PATH
from .train_lgbm import FEATURE_COLS, load_lightgbm_model


def predict_pair_probabilities(
    df_pair_features: pd.DataFrame,
    model: lgb.Booster = None,
    model_path: Path = MODEL_SAVE_PATH
) -> pd.DataFrame:
    """
    Appends 'match_prob' column to the candidate pair features.
    """
    if df_pair_features.empty:
        df_copy = df_pair_features.copy()
        df_copy["match_prob"] = []
        return df_copy

    if model is None:
        model = load_lightgbm_model(model_path)

    X = df_pair_features[FEATURE_COLS]
    probs = model.predict(X)

    df_result = df_pair_features[["source1_entity_id", "candidate_entity_id"]].copy()
    df_result["match_prob"] = probs
    return df_result
