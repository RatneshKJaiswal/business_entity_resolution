"""
Validation runner performing entity-level holdout split, validation scoring, and error analysis.
"""

from pathlib import Path
from typing import Dict, List, Set, Tuple
import gc
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from ..config import (
    TRAIN_DIR,
    MODEL_SAVE_PATH,
    MAX_CANDIDATES_PER_S1,
    RANDOM_STATE,
    N_JOBS
)
from ..utils.io_utils import load_source_dataframe, load_ground_truth
from ..utils.evaluator import compute_macro_f05, compute_entity_f05
from ..blocking.candidate_builder import CandidateBuilder
from ..features.feature_pipeline import FeaturePipeline
from ..postprocessing.threshold_optimizer import find_optimal_threshold
from ..postprocessing.matcher import select_matches_from_scored_pairs
from .train_lgbm import train_lightgbm_model
from .inference import predict_pair_probabilities


def run_entity_validation_pipeline(
    train_dir: Path = TRAIN_DIR,
    val_fraction: float = 0.20,
    sample_size: int = 100_000
) -> Dict[str, float]:
    """
    Executes a true entity-level validation workflow:
    1. Loads train records and cuts an 80/20 train/validation split on Source 1 entities.
    2. Trains the LightGBM classifier on the training split.
    3. Runs candidate blocking & feature extraction on the held-out validation entities.
    4. Evaluates predictions using the official Macro F_0.5 metric.
    5. Optimizes the decision threshold tau.
    """
    print(f"\n=======================================================")
    print(f"ENTITY-LEVEL VALIDATION PIPELINE (val_fraction={val_fraction:.0%})")
    print(f"=======================================================")

    s1_path = train_dir / "train_source1.tsv"
    s2_path = train_dir / "train_source2.tsv"
    s3_path = train_dir / "train_source3.tsv"
    gt_path = train_dir / "train_ground_truth.tsv"

    print("\n[Step 1] Loading Ground Truth and Source records...")
    gt_map = load_ground_truth(gt_path)

    # Load S1 records
    df_s1 = load_source_dataframe(s1_path, nrows=sample_size)
    print(f"  Loaded {len(df_s1)} Source 1 records.")

    # Cut entity-level train and validation splits (stratified by country)
    df_s1_train, df_s1_val = train_test_split(
        df_s1,
        test_size=val_fraction,
        random_state=RANDOM_STATE,
        stratify=df_s1["country"]
    )
    print(f"  Split: {len(df_s1_train)} Train entities, {len(df_s1_val)} Validation entities.")

    # Load S2 and S3 query records
    df_s2 = load_source_dataframe(s2_path, nrows=sample_size * 2)
    df_s3 = load_source_dataframe(s3_path, nrows=sample_size * 2)
    df_target = pd.concat([df_s2, df_s3], ignore_index=True)
    del df_s2, df_s3
    gc.collect()
    print(f"  Loaded {len(df_target)} target query records (S2 + S3).")

    # [Step 2] Candidate generation & training feature extraction
    print("\n[Step 2] Building training pairs for LightGBM...")
    builder = CandidateBuilder(top_k=MAX_CANDIDATES_PER_S1)
    
    train_feature_dfs = []
    val_feature_dfs = []
    val_cand_mapping = {}

    for country in ["US", "India"]:
        df_s1_tr_c = df_s1_train[df_s1_train["country"] == country]
        df_s1_vl_c = df_s1_val[df_s1_val["country"] == country]
        df_target_c = df_target[df_target["country"] == country]

        if df_target_c.empty:
            continue

        print(f"  Indexing {country} targets ({len(df_target_c)} records)...")
        builder.index_target_records(df_target_c, country)
        target_ids_set = set(df_target_c["entity_id"])

        # Train partition candidates
        print(f"  Generating candidates for {country} train entities ({len(df_s1_tr_c)})...")
        cand_map_tr = builder.generate_candidates(df_s1_tr_c, country)

        # Inject ground truth positives for training
        for s1_id, cand_list in cand_map_tr.items():
            true_matches = gt_map.get(s1_id, set())
            for tm in true_matches:
                if tm in target_ids_set and tm not in cand_list:
                    cand_list.append(tm)

        feat_df_tr = FeaturePipeline.extract_features_for_pairs(
            cand_map_tr, df_s1_tr_c, df_target_c, country=country
        )
        if not feat_df_tr.empty:
            feat_df_tr["label"] = feat_df_tr.apply(
                lambda r: 1 if r["candidate_entity_id"] in gt_map.get(r["source1_entity_id"], set()) else 0,
                axis=1
            )
            train_feature_dfs.append(feat_df_tr)

        # Validation partition candidates (NO ground truth injection, simulate real test set!)
        print(f"  Generating candidates for {country} validation entities ({len(df_s1_vl_c)})...")
        cand_map_vl = builder.generate_candidates(df_s1_vl_c, country)
        val_cand_mapping.update(cand_map_vl)

        feat_df_vl = FeaturePipeline.extract_features_for_pairs(
            cand_map_vl, df_s1_vl_c, df_target_c, country=country
        )
        if not feat_df_vl.empty:
            val_feature_dfs.append(feat_df_vl)

    # Train LightGBM model
    print("\n[Step 3] Fitting LightGBM classifier on training split...")
    full_train_df = pd.concat(train_feature_dfs, ignore_index=True)
    positives = full_train_df[full_train_df["label"] == 1]
    negatives = full_train_df[full_train_df["label"] == 0]
    neg_sample = negatives.sample(min(len(negatives), len(positives) * 5), random_state=RANDOM_STATE)
    balanced_df = pd.concat([positives, neg_sample], ignore_index=True)

    clf = train_lightgbm_model(balanced_df, target_col="label", model_path=MODEL_SAVE_PATH)

    # [Step 4] Validation Inference
    print("\n[Step 4] Scoring validation candidate pairs...")
    full_val_df = pd.concat(val_feature_dfs, ignore_index=True)
    val_scored_df = predict_pair_probabilities(full_val_df, model=clf.booster_)

    # [Step 5] Threshold search for Macro F_0.5
    val_s1_ids = df_s1_val["entity_id"].tolist()
    best_tau, best_f05 = find_optimal_threshold(
        df_scored_val_pairs=val_scored_df,
        val_s1_ids=val_s1_ids,
        ground_truth=gt_map,
        threshold_range=(0.60, 0.90, 0.02)
    )

    # [Step 6] Error Analysis on Validation Set
    print("\n[Step 5] Detailed Error Analysis at optimal tau = {:.2f}:".format(best_tau))
    best_preds = select_matches_from_scored_pairs(
        df_scored_pairs=val_scored_df,
        all_s1_ids=val_s1_ids,
        threshold=best_tau,
        singleton_threshold=best_tau - 0.05
    )

    singleton_correct = 0
    singleton_total = 0
    tp_total = 0
    fp_total = 0
    fn_total = 0

    for s1_id in val_s1_ids:
        gt_set = gt_map.get(s1_id, set())
        pred_set = best_preds.get(s1_id, set())
        
        if len(gt_set) == 0:
            singleton_total += 1
            if len(pred_set) == 0:
                singleton_correct += 1
            else:
                fp_total += len(pred_set)
        else:
            tp = len(pred_set & gt_set)
            tp_total += tp
            fp_total += len(pred_set - gt_set)
            fn_total += len(gt_set - pred_set)

    singleton_acc = singleton_correct / singleton_total if singleton_total > 0 else 0.0
    precision = tp_total / (tp_total + fp_total) if (tp_total + fp_total) > 0 else 0.0
    recall = tp_total / (tp_total + fn_total) if (tp_total + fn_total) > 0 else 0.0

    print(f"  * Validation Entities:        {len(val_s1_ids)}")
    print(f"  * Best Macro F_0.5:           {best_f05:.4f}")
    print(f"  * Non-singleton Precision:    {precision:.4f} (tp={tp_total}, fp={fp_total})")
    print(f"  * Non-singleton Recall:       {recall:.4f} (tp={tp_total}, fn={fn_total})")
    print(f"  * Singleton Accuracy:         {singleton_acc:.4f} ({singleton_correct}/{singleton_total} correct empty predictions)")
    print("=======================================================\n")

    return {
        "best_tau": best_tau,
        "macro_f05": best_f05,
        "precision": precision,
        "recall": recall,
        "singleton_accuracy": singleton_acc
    }
