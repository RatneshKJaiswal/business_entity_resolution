"""
Master end-to-end execution pipeline for Business Entity Resolution.
When running training (--train):
- Automatically performs an 80/20 entity-level train/validation split
- Trains LightGBM on the 80% train split
- Evaluates Macro F_0.5 and tunes threshold tau on the 20% validation split
- Saves model and optimal threshold metadata
"""

import argparse
import json
from pathlib import Path
import sys
import time
import gc

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import (
    TRAIN_DIR,
    TEST_DIR,
    OUTPUT_DIR,
    MATCHING_RESULTS_PATH,
    CANDIDATE_PAIRS_PATH,
    MODEL_SAVE_PATH,
    MODEL_METADATA_PATH,
    MATCH_THRESHOLD,
    MAX_CANDIDATES_PER_S1,
    RANDOM_STATE,
)
from src.utils.io_utils import (
    load_source_dataframe,
    load_ground_truth,
    write_submission_tsv
)
from src.blocking.candidate_builder import CandidateBuilder, FALLBACK_COUNTRY
from src.features.feature_pipeline import FeaturePipeline
from src.models.train_lgbm import train_lightgbm_model, load_lightgbm_model
from src.models.tuner import run_kfold_hyperparameter_tuning
from src.models.inference import predict_pair_probabilities
from src.postprocessing.threshold_optimizer import find_optimal_threshold
from src.postprocessing.matcher import select_matches_from_scored_pairs
from src.postprocessing.tsv_formatter import save_matching_results


def _get_country_partitions(df):
    """
    Returns list of country partitions including a FALLBACK for missing countries.
    Entities with NaN/empty country go into FALLBACK_COUNTRY partition.
    """
    countries = sorted(df["country"].fillna("").unique().tolist())
    partitions = []
    for c in countries:
        if c and c.strip():
            partitions.append(c)
        # Empty/NaN countries will be handled as FALLBACK
    # Check if any entities have missing country
    missing_count = df["country"].fillna("").eq("").sum()
    if missing_count > 0:
        partitions.append(FALLBACK_COUNTRY)
    return partitions


def _filter_by_country(df, country):
    """Filter DataFrame by country, handling FALLBACK for missing."""
    if country == FALLBACK_COUNTRY:
        return df[df["country"].fillna("").eq("")]
    return df[df["country"] == country]


def run_training_stage(
    train_dir: Path,
    sample_size: int = 250_000,
    val_fraction: float = 0.20,
    tune_hyperparams: bool = False,
    cv_folds: int = 5,
    n_trials: int = 4
) -> float:
    """
    Loads training data, cuts an 80/20 entity-level train/val split,
    trains LightGBM on the 80% split (with optional K-Fold CV & tuning),
    tunes threshold on the 20% validation split, and saves model weights + threshold metadata.
    """
    mode_desc = f"with {cv_folds}-Fold CV Tuning" if tune_hyperparams else "Standard Training"
    sample_desc = f"{sample_size:,} entities" if sample_size else "All Entities (Full Dataset)"
    print(f"\n=================================================================")
    print(f"TRAINING STAGE ({mode_desc}, Sample: {sample_desc}, Train/Val Split: {1-val_fraction:.0%}/{val_fraction:.0%})")
    print(f"=================================================================")
    
    s1_path = train_dir / "train_source1.tsv"
    s2_path = train_dir / "train_source2.tsv"
    s3_path = train_dir / "train_source3.tsv"
    gt_path = train_dir / "train_ground_truth.tsv"

    # 1. Load ground truth and reference S1
    print("\n[Step 1/5] Loading Ground Truth and Reference Source 1...")
    gt_map = load_ground_truth(gt_path)

    # Load S1 — if sampling, load all first then random sample (not head nrows)
    if sample_size is None or sample_size <= 0 or sample_size >= 2_200_000:
        df_s1 = load_source_dataframe(s1_path)
        print(f"  Loaded {len(df_s1):,} Source 1 records (100% full dataset).")
    else:
        df_s1_full = load_source_dataframe(s1_path)
        df_s1 = df_s1_full.sample(n=min(sample_size, len(df_s1_full)), random_state=RANDOM_STATE)
        del df_s1_full
        gc.collect()
        print(f"  Randomly sampled {len(df_s1):,} Source 1 records.")

    # 2. Perform 80/20 entity-level split stratified by country
    df_s1["_country_fill"] = df_s1["country"].fillna("__NA__")
    df_s1_train, df_s1_val = train_test_split(
        df_s1,
        test_size=val_fraction,
        random_state=RANDOM_STATE,
        stratify=df_s1["_country_fill"]
    )
    df_s1.drop(columns=["_country_fill"], inplace=True)
    df_s1_train = df_s1_train.drop(columns=["_country_fill"], errors="ignore")
    df_s1_val = df_s1_val.drop(columns=["_country_fill"], errors="ignore")
    print(f"  Entity Split: {len(df_s1_train):,} Train entities (80%), {len(df_s1_val):,} Validation entities (20%).")

    # 3. Load Query records (S2 + S3)
    print("\n[Step 2/5] Loading Source 2 and Source 3 query records...")
    df_s2 = load_source_dataframe(s2_path)
    df_s3 = load_source_dataframe(s3_path)
    df_target = pd.concat([df_s2, df_s3], ignore_index=True)
    del df_s2, df_s3
    gc.collect()
    print(f"  Loaded {len(df_target):,} target records for blocking.")

    # 4. Generate candidate pairs & extract features for train and val splits
    print("\n[Step 3/5] Candidate Blocking and Feature Extraction...")
    builder = CandidateBuilder(top_k=MAX_CANDIDATES_PER_S1)
    
    # Determine country partitions from the combined data
    all_countries_s1 = set(df_s1["country"].fillna("").unique())
    all_countries_target = set(df_target["country"].fillna("").unique())
    known_countries = sorted({c for c in (all_countries_s1 | all_countries_target) if c and c.strip()})
    
    # Check if any target entities have missing country
    has_fallback = df_target["country"].fillna("").eq("").any()
    
    partitions = known_countries[:]
    if has_fallback:
        partitions.append(FALLBACK_COUNTRY)
    
    print(f"  Detected country partitions: {partitions}")

    train_feature_dfs = []
    val_feature_dfs = []
    all_cand_map_val = {}

    for country in partitions:
        df_s1_tr_c = _filter_by_country(df_s1_train, country)
        df_s1_vl_c = _filter_by_country(df_s1_val, country)
        df_target_c = _filter_by_country(df_target, country)

        if df_target_c.empty:
            continue

        print(f"\n  === Processing partition: {country} ===")
        print(f"  Indexing {country} targets ({len(df_target_c):,} records)...")
        builder.index_target_records(df_target_c, country)
        target_ids_set = set(df_target_c["entity_id"])

        # Train partition candidates
        if not df_s1_tr_c.empty:
            print(f"  Generating candidates for {country} train entities ({len(df_s1_tr_c):,})...")
            cand_map_tr = builder.generate_candidates(df_s1_tr_c, country)

            # Inject ground truth true positives if missing from blocking
            for s1_id, cand_list in cand_map_tr.items():
                true_matches = gt_map.get(s1_id, set())
                for tm in true_matches:
                    if tm in target_ids_set and tm not in cand_list:
                        cand_list.append(tm)

            # EFFICIENT TRAINING NEGATIVE PRUNING:
            # Keep ALL true positives, and keep up to 8 hard negative candidates per S1 entity.
            # This slashes training pairs from 35M to ~10M, preventing MemoryError and speeding up feature extraction 4x
            for s1_id, cand_list in cand_map_tr.items():
                true_matches = gt_map.get(s1_id, set())
                pos = [c for c in cand_list if c in true_matches]
                neg = [c for c in cand_list if c not in true_matches]
                if len(neg) > 30:
                    neg = neg[:30]
                cand_map_tr[s1_id] = pos + neg

            feat_df_tr = FeaturePipeline.extract_features_for_pairs(
                cand_map_tr, df_s1_tr_c, df_target_c, country=country if country != FALLBACK_COUNTRY else ""
            )
            if not feat_df_tr.empty:
                feat_df_tr["label"] = feat_df_tr.apply(
                    lambda r: 1 if r["candidate_entity_id"] in gt_map.get(r["source1_entity_id"], set()) else 0,
                    axis=1
                )
                train_feature_dfs.append(feat_df_tr)

        # Validation partition candidates (simulate unseen test set, NO ground truth injection)
        if not df_s1_vl_c.empty:
            print(f"  Generating candidates for {country} validation entities ({len(df_s1_vl_c):,})...")
            cand_map_vl = builder.generate_candidates(df_s1_vl_c, country)
            all_cand_map_val.update(cand_map_vl)
            feat_df_vl = FeaturePipeline.extract_features_for_pairs(
                cand_map_vl, df_s1_vl_c, df_target_c, country=country if country != FALLBACK_COUNTRY else ""
            )
            if not feat_df_vl.empty:
                val_feature_dfs.append(feat_df_vl)

        del df_s1_tr_c, df_s1_vl_c, df_target_c
        gc.collect()

    # Measure Blocker Recall on the held-out validation entities
    val_s1_set = set(df_s1_val["entity_id"])
    total_val_gt = 0
    retrieved_val_gt = 0
    for sid in val_s1_set:
        gt_set = gt_map.get(sid, set())
        total_val_gt += len(gt_set)
        cand_set = set(all_cand_map_val.get(sid, []))
        retrieved_val_gt += len(gt_set & cand_set)

    if total_val_gt > 0:
        val_blocker_recall = retrieved_val_gt / total_val_gt
        print(f"\n  ---------------------------------------------------------------")
        print(f"  BLOCKER QUALITY: Candidate Recall = {val_blocker_recall:.2%} ({retrieved_val_gt:,}/{total_val_gt:,} true matches in top-{MAX_CANDIDATES_PER_S1})")
        print(f"  ---------------------------------------------------------------\n")

    # 5. Fit LightGBM on the 80% train partition
    print("\n[Step 4/5] Training LightGBM classifier on the 80% training partition...")
    if not train_feature_dfs:
        print("ERROR: No training features extracted. Check blocking and data.")
        sys.exit(1)

    full_train_df = pd.concat(train_feature_dfs, ignore_index=True)
    del train_feature_dfs
    gc.collect()

    pos_count = int((full_train_df["label"] == 1).sum())
    neg_count = len(full_train_df) - pos_count
    dynamic_spw = min(max(neg_count / max(pos_count, 1), 1.0), 5.0)
    ratio_str = f"1:{neg_count // max(pos_count, 1)}"
    print(f"  Extracted pairs: {pos_count:,} positive matches, {neg_count:,} negative candidates (ratio {ratio_str}).")
    print(f"  Dynamic scale_pos_weight: {dynamic_spw:.2f}")
    balanced_df = full_train_df

    if tune_hyperparams:
        clf, best_config, cv_metrics = run_kfold_hyperparameter_tuning(
            df_features=balanced_df,
            n_splits=cv_folds,
            n_trials=n_trials,
            target_col="label",
            model_path=MODEL_SAVE_PATH,
            scale_pos_weight=dynamic_spw
        )
    else:
        clf = train_lightgbm_model(balanced_df, target_col="label", model_path=MODEL_SAVE_PATH,
                                   params={"scale_pos_weight": dynamic_spw})
        best_config = None
        cv_metrics = None

    # 6. Evaluate on 20% validation partition & tune threshold tau for Macro F_0.5
    print("\n[Step 5/5] Evaluating on the 20% validation split & optimizing threshold tau...")
    if not val_feature_dfs:
        print("WARNING: No validation features extracted. Using default threshold.")
        best_tau = MATCH_THRESHOLD
        best_f05 = 0.0
    else:
        full_val_df = pd.concat(val_feature_dfs, ignore_index=True)
        val_scored_df = predict_pair_probabilities(full_val_df, model=clf.booster_)

        val_s1_ids = df_s1_val["entity_id"].tolist()
        best_tau, best_f05 = find_optimal_threshold(
            df_scored_val_pairs=val_scored_df,
            val_s1_ids=val_s1_ids,
            ground_truth=gt_map
        )

        # Compute detailed validation report
        best_preds = select_matches_from_scored_pairs(
            df_scored_pairs=val_scored_df,
            all_s1_ids=val_s1_ids,
            threshold=best_tau
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

        print("-----------------------------------------------------------------")
        print("VALIDATION RESULTS SUMMARY (Held-out 20% Split):")
        print(f"  * Validation Entities:        {len(val_s1_ids):,}")
        print(f"  * Optimal Match Threshold:    tau = {best_tau:.2f}")
        print(f"  * Best Validation Macro F_0.5: {best_f05:.4f}")
        print(f"  * Non-singleton Precision:    {precision:.4f} (TP={tp_total}, FP={fp_total})")
        print(f"  * Non-singleton Recall:       {recall:.4f} (TP={tp_total}, FN={fn_total})")
        print(f"  * Singleton Accuracy:         {singleton_acc:.4f} ({singleton_correct}/{singleton_total} singletons correctly empty)")
        print("-----------------------------------------------------------------")

    # Save metadata so prediction stage uses this optimal threshold automatically
    metadata = {
        "best_tau": best_tau,
        "best_macro_f05": best_f05,
        "precision": precision if 'precision' in dir() else 0.0,
        "recall": recall if 'recall' in dir() else 0.0,
        "singleton_accuracy": singleton_acc if 'singleton_acc' in dir() else 0.0,
        "val_sample_size": len(df_s1_val),
        "total_sample_size": sample_size or len(df_s1),
        "cv_tuning_performed": tune_hyperparams,
        "winning_hyperparams": {k: v for k, v in best_config.items() if k != "name"} if best_config else None,
        "cv_metrics": cv_metrics
    }
    with open(MODEL_METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"Optimal threshold and tuning configuration saved to {MODEL_METADATA_PATH}")

    return best_tau


def run_prediction_stage(
    test_dir: Path,
    output_dir: Path,
    threshold: float = None
) -> None:
    """
    Runs full inference pipeline on the test set:
    Blocking -> Feature Extraction -> LightGBM Prediction -> TSV Outputs
    """
    # Load optimal threshold from metadata if not explicitly provided
    if threshold is None:
        if MODEL_METADATA_PATH.exists():
            try:
                with open(MODEL_METADATA_PATH, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                    threshold = float(meta.get("best_tau", MATCH_THRESHOLD))
                    print(f"Loaded calibrated threshold from validation: tau = {threshold:.2f}")
            except Exception:
                threshold = MATCH_THRESHOLD
        else:
            threshold = MATCH_THRESHOLD

    print(f"\n=================================================================")
    print(f"PREDICTION STAGE: Loading test data from {test_dir}...")
    print(f"=================================================================")
    
    s1_path = test_dir / "test_source1.tsv"
    s2_path = test_dir / "test_source2.tsv"
    s3_path = test_dir / "test_source3.tsv"

    print("  Loading test Source 1 records...")
    df_s1 = load_source_dataframe(s1_path)
    all_s1_ids = df_s1["entity_id"].tolist()
    print(f"  Total test Source 1 entities: {len(all_s1_ids):,}")

    print("  Loading test Source 2 and Source 3 records...")
    df_s2 = load_source_dataframe(s2_path)
    df_s3 = load_source_dataframe(s3_path)
    df_target = pd.concat([df_s2, df_s3], ignore_index=True)
    del df_s2, df_s3
    gc.collect()
    print(f"  Total test target query records (S2 + S3): {len(df_target):,}")

    # Load trained model
    print(f"  Loading trained model from {MODEL_SAVE_PATH}...")
    model = load_lightgbm_model(MODEL_SAVE_PATH)

    builder = CandidateBuilder(top_k=MAX_CANDIDATES_PER_S1)
    all_candidate_mapping = {}
    all_scored_pairs = []

    # Determine country partitions dynamically
    partitions = _get_country_partitions(pd.concat([df_s1, df_target], ignore_index=True))
    print(f"  Detected country partitions: {partitions}")

    for country in partitions:
        df_s1_c = _filter_by_country(df_s1, country)
        df_target_c = _filter_by_country(df_target, country)
        if df_s1_c.empty:
            continue

        print(f"\n  Processing partition: {country} ({len(df_s1_c):,} S1 entities, {len(df_target_c):,} targets)...")
        
        if not df_target_c.empty:
            print(f"    [a] Building inverted index for {country}...")
            builder.index_target_records(df_target_c, country)

        print(f"    [b] Generating candidate pairs...")
        cand_map_c = builder.generate_candidates(df_s1_c, country)
        all_candidate_mapping.update(cand_map_c)

        # Memory-efficient chunked feature extraction and scoring (50k S1 entities per chunk)
        all_cand_ids = {cid for cids in cand_map_c.values() for cid in cids}
        df_target_for_features = df_target[df_target["entity_id"].isin(all_cand_ids)]
        effective_country = country if country != FALLBACK_COUNTRY else ""

        CHUNK_SIZE = 50_000
        s1_items = list(cand_map_c.items())
        total_chunks = (len(s1_items) + CHUNK_SIZE - 1) // CHUNK_SIZE
        print(f"    [c/d] Extracting features & scoring in {total_chunks} chunks of up to {CHUNK_SIZE:,} entities...")

        for chunk_idx, chunk_start in enumerate(range(0, len(s1_items), CHUNK_SIZE), 1):
            chunk_cand_map = dict(s1_items[chunk_start:chunk_start + CHUNK_SIZE])
            chunk_s1_ids = set(chunk_cand_map.keys())
            chunk_df_s1 = df_s1_c[df_s1_c["entity_id"].isin(chunk_s1_ids)]

            chunk_feat_df = FeaturePipeline.extract_features_for_pairs(
                chunk_cand_map, chunk_df_s1, df_target_for_features, country=effective_country
            )
            if not chunk_feat_df.empty:
                scored_chunk = predict_pair_probabilities(chunk_feat_df, model=model)
                # Drop pairs with prob < 0.20 to keep memory tiny
                scored_chunk = scored_chunk[scored_chunk["match_prob"] >= 0.05]
                if not scored_chunk.empty:
                    all_scored_pairs.append(scored_chunk)
                del chunk_feat_df, scored_chunk
                gc.collect()

        del df_s1_c, df_target_c, df_target_for_features, cand_map_c
        gc.collect()

    # 1. Save candidate_pairs.tsv (blocking output)
    candidate_output_path = output_dir / "candidate_pairs.tsv"
    print(f"\n  Writing candidate pairs to {candidate_output_path}...")
    builder.save_candidate_pairs(
        candidate_mapping=all_candidate_mapping,
        output_path=candidate_output_path,
        required_s1_order=all_s1_ids
    )

    # 2. Select final matches using threshold
    print(f"  Applying calibrated threshold (tau={threshold:.2f})...")
    scored_pairs_df = pd.concat(all_scored_pairs, ignore_index=True) if all_scored_pairs else pd.DataFrame()
    matches = select_matches_from_scored_pairs(
        df_scored_pairs=scored_pairs_df,
        all_s1_ids=all_s1_ids,
        threshold=threshold
    )

    # 3. Save matching_results.tsv (final scored output)
    matching_output_path = output_dir / "matching_results.tsv"
    print(f"  Writing final matching results to {matching_output_path}...")
    save_matching_results(
        matches=matches,
        required_s1_order=all_s1_ids,
        output_path=matching_output_path
    )
    
    # Print summary statistics
    non_empty = sum(1 for v in matches.values() if v)
    singletons = sum(1 for v in matches.values() if not v)
    print(f"\n  PREDICTION SUMMARY:")
    print(f"    Total S1 entities:  {len(matches):,}")
    print(f"    Matched (non-empty): {non_empty:,}")
    print(f"    Singletons (empty):  {singletons:,}")
    print("  Inference complete!")


def main():
    parser = argparse.ArgumentParser(description="End-to-End Business Entity Resolution Pipeline")
    parser.add_argument("--train", action="store_true", help="Run model training (with automatic 80/20 train/val split and threshold tuning)")
    parser.add_argument("--predict", action="store_true", help="Run test inference and produce output TSVs")
    parser.add_argument("--sample-size", "--train-sample-size", dest="sample_size", type=int, default=250_000, help="Number of S1 train rows to sample (default: 250,000)")
    parser.add_argument("--whole-data", "--all-data", dest="whole_data", action="store_true", help="Train on all records in the training dataset (all 2.2M S1 and 10.3M targets)")
    parser.add_argument("--val-fraction", type=float, default=0.20, help="Fraction of train S1 entities for validation (default: 0.20 for 80/20 split)")
    parser.add_argument("--tune", action="store_true", help="Enable K-Fold Cross Validation and hyperparameter tuning")
    parser.add_argument("--cv-folds", type=int, default=5, help="Number of folds for GroupKFold Cross Validation (default: 5)")
    parser.add_argument("--n-trials", type=int, default=4, help="Number of hyperparameter configurations to test (default: 4)")
    parser.add_argument("--train-dir", type=Path, default=TRAIN_DIR, help="Path to train dataset directory")
    parser.add_argument("--test-dir", type=Path, default=TEST_DIR, help="Path to test dataset directory")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR, help="Path to save output TSV files")
    parser.add_argument("--threshold", type=float, default=None, help="Optional manual match threshold override")
    args = parser.parse_args()

    t0 = time.time()
    effective_sample_size = None if args.whole_data else args.sample_size

    # If neither train nor predict specified, run both end-to-end
    run_train = args.train or (not args.train and not args.predict)
    run_pred = args.predict or (not args.train and not args.predict)

    if run_train:
        best_tau = run_training_stage(
            train_dir=args.train_dir,
            sample_size=effective_sample_size,
            val_fraction=args.val_fraction,
            tune_hyperparams=args.tune,
            cv_folds=args.cv_folds,
            n_trials=args.n_trials
        )
        if args.threshold is None:
            args.threshold = best_tau

    if run_pred:
        run_prediction_stage(
            test_dir=args.test_dir,
            output_dir=args.output_dir,
            threshold=args.threshold
        )

    print(f"\nEntire execution completed in {time.time() - t0:.2f} seconds.")


if __name__ == "__main__":
    import multiprocessing as mp
    mp.freeze_support()
    main()
