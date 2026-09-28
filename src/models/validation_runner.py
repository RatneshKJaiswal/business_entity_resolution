"""
Validation runner performing entity-level holdout split, validation scoring, and error analysis.
"""

import argparse
from collections import defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Dict, List, Optional, Set, Tuple
import gc
import pandas as pd
from sklearn.model_selection import train_test_split

from ..preprocessing.address_parser import extract_numbers_and_postal
from ..config import (
    TRAIN_DIR,
    MAX_CANDIDATES_PER_S1,
    RANDOM_STATE
)
from ..preprocessing.text_cleaner import get_name_tokens, strip_legal_suffixes
from ..utils.io_utils import load_source_dataframe, load_ground_truth
from ..utils.evaluator import compute_macro_f05
from ..blocking.candidate_builder import CandidateBuilder, prune_training_candidates
from ..blocking.inverted_index_blocker import (
    BLOCKING_CHANNELS,
    DEFAULT_BLOCKING_CHANNELS
)
from ..features.feature_pipeline import FeaturePipeline
from ..postprocessing.threshold_optimizer import find_optimal_threshold
from ..postprocessing.matcher import select_matches_from_scored_pairs
from .train_lgbm import label_candidate_pairs, train_lightgbm_model
from .inference import predict_pair_probabilities


FALLBACK_COUNTRY = "__UNKNOWN__"


def _country_partitions(df: pd.DataFrame) -> List[str]:
    countries = {
        country for country in df["country"].fillna("").unique()
        if country and country.strip()
    }
    partitions = sorted(countries)
    if df["country"].fillna("").eq("").any():
        partitions.append(FALLBACK_COUNTRY)
    return partitions


def _filter_country(df: pd.DataFrame, country: str) -> pd.DataFrame:
    if country == FALLBACK_COUNTRY:
        return df[df["country"].fillna("").eq("")]
    return df[df["country"] == country]


def _report_blocking_miss_signatures(
    val_s1_ids: List[str],
    ground_truth: Dict[str, Set[str]],
    candidates: Dict[str, List[str]],
    df_s1: pd.DataFrame,
    df_target: pd.DataFrame
) -> None:
    """Summarize name and address evidence for true pairs absent from blocking."""
    missed_pairs = [
        (s1_id, target_id)
        for s1_id in val_s1_ids
        for target_id in ground_truth.get(s1_id, set()) - set(candidates.get(s1_id, []))
    ]
    if not missed_pairs:
        print("  * Blocking miss signatures:   no missed true pairs.")
        return

    needed_s1 = {s1_id for s1_id, _ in missed_pairs}
    needed_targets = {target_id for _, target_id in missed_pairs}
    s1_records = (
        df_s1[df_s1["entity_id"].isin(needed_s1)]
        .set_index("entity_id")[["business_name", "business_address", "country"]]
        .to_dict("index")
    )
    target_records = (
        df_target[df_target["entity_id"].isin(needed_targets)]
        .set_index("entity_id")[["business_name", "business_address", "country"]]
        .to_dict("index")
    )
    name_cache = {}
    address_cache = {}
    by_country = defaultdict(lambda: {
        "missed_pairs": 0,
        "exact_core_name": 0,
        "name_token_overlap": 0,
        "postal_match": 0,
        "address_number_overlap": 0
    })

    def name_features(entity_id, record, country):
        key = (entity_id, country)
        if key not in name_cache:
            core = strip_legal_suffixes(record.get("business_name", "") or "", country)
            name_cache[key] = (core, get_name_tokens(core))
        return name_cache[key]

    def address_features(entity_id, record, country):
        key = (entity_id, country)
        if key not in address_cache:
            address = record.get("business_address", "") or ""
            address_cache[key] = extract_numbers_and_postal(address, country)
        return address_cache[key]

    for s1_id, target_id in missed_pairs:
        s1_record = s1_records.get(s1_id)
        target_record = target_records.get(target_id)
        if s1_record is None or target_record is None:
            continue

        country = s1_record.get("country", "") or FALLBACK_COUNTRY
        stats = by_country[country]
        stats["missed_pairs"] += 1

        s1_core, s1_tokens = name_features(s1_id, s1_record, country)
        target_core, target_tokens = name_features(target_id, target_record, country)
        if s1_core and s1_core == target_core:
            stats["exact_core_name"] += 1
        if s1_tokens & target_tokens:
            stats["name_token_overlap"] += 1

        s1_numbers, s1_postal = address_features(s1_id, s1_record, country)
        target_numbers, target_postal = address_features(target_id, target_record, country)
        if s1_postal and s1_postal == target_postal:
            stats["postal_match"] += 1
        if s1_numbers & target_numbers:
            stats["address_number_overlap"] += 1

    print("  * Blocking miss signatures (among true pairs not retrieved):")
    for country in sorted(by_country):
        stats = by_country[country]
        total = stats["missed_pairs"]
        print(
            f"    {country}: missed={total:,}, "
            f"exact_core_name={stats['exact_core_name'] / total:.3f}, "
            f"name_token_overlap={stats['name_token_overlap'] / total:.3f}, "
            f"postal_match={stats['postal_match'] / total:.3f}, "
            f"address_number_overlap={stats['address_number_overlap'] / total:.3f}"
        )


def run_entity_validation_pipeline(
    train_dir: Path = TRAIN_DIR,
    val_fraction: float = 0.20,
    sample_size: Optional[int] = 100_000,
    threshold_range: Optional[Tuple[float, float, float]] = None,
    run_ablations: bool = False
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
    df_s1 = load_source_dataframe(s1_path)
    if sample_size is not None and sample_size > 0 and sample_size < len(df_s1):
        df_s1 = df_s1.sample(n=sample_size, random_state=RANDOM_STATE)
    print(f"  Loaded {len(df_s1)} Source 1 records.")

    # Cut entity-level train and validation splits (stratified by country).
    df_s1["_country_fill"] = df_s1["country"].fillna("").replace("", FALLBACK_COUNTRY)
    country_counts = df_s1["_country_fill"].value_counts()
    stratify = df_s1["_country_fill"] if country_counts.min() >= 2 else None
    df_s1_train, df_s1_val = train_test_split(
        df_s1,
        test_size=val_fraction,
        random_state=RANDOM_STATE,
        stratify=stratify
    )
    df_s1_train = df_s1_train.drop(columns=["_country_fill"])
    df_s1_val = df_s1_val.drop(columns=["_country_fill"])
    print(f"  Split: {len(df_s1_train)} Train entities, {len(df_s1_val)} Validation entities.")

    # Use the complete target sets so validation blocking matches inference.
    df_s2 = load_source_dataframe(s2_path)
    df_s3 = load_source_dataframe(s3_path)
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
    ablation_channels = {
        f"without_{channel}": DEFAULT_BLOCKING_CHANNELS - {channel}
        for channel in sorted(DEFAULT_BLOCKING_CHANNELS)
    } if run_ablations else {}
    if run_ablations:
        ablation_channels["with_prefix"] = BLOCKING_CHANNELS
    ablation_candidate_maps = {name: {} for name in ablation_channels}
    if run_ablations:
        ablation_candidate_maps["cap_300"] = {}
        ablation_candidate_maps["cap_500"] = {}
    partitions = _country_partitions(pd.concat([df_s1, df_target], ignore_index=True))

    for country in partitions:
        df_s1_tr_c = _filter_country(df_s1_train, country)
        df_s1_vl_c = _filter_country(df_s1_val, country)
        df_target_c = _filter_country(df_target, country)

        if df_target_c.empty:
            continue

        effective_country = "" if country == FALLBACK_COUNTRY else country
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
        cand_map_tr = prune_training_candidates(cand_map_tr, gt_map)

        feat_df_tr = FeaturePipeline.extract_features_for_pairs(
            cand_map_tr, df_s1_tr_c, df_target_c, country=effective_country
        )
        if not feat_df_tr.empty:
            label_candidate_pairs(feat_df_tr, gt_map)
            train_feature_dfs.append(feat_df_tr)

        # Validation partition candidates (NO ground truth injection, simulate real test set!)
        print(f"  Generating candidates for {country} validation entities ({len(df_s1_vl_c)})...")
        cand_map_vl = builder.generate_candidates(df_s1_vl_c, country)
        val_cand_mapping.update(cand_map_vl)
        for name, channels in ablation_channels.items():
            ablation_candidate_maps[name].update(
                builder.generate_candidates(
                    df_s1_vl_c, country, enabled_channels=channels
                )
            )
        if run_ablations:
            cap_500_candidates = builder.generate_candidates(
                df_s1_vl_c, country, top_k=500
            )
            ablation_candidate_maps["cap_500"].update({
                s1_id: candidates[:500]
                for s1_id, candidates in cap_500_candidates.items()
            })

        feat_df_vl = FeaturePipeline.extract_features_for_pairs(
            cand_map_vl, df_s1_vl_c, df_target_c, country=effective_country
        )
        if not feat_df_vl.empty:
            val_feature_dfs.append(feat_df_vl)

    # Train LightGBM model
    print("\n[Step 3] Fitting LightGBM classifier on training split...")
    if not train_feature_dfs:
        raise ValueError("No training candidate features were generated; check the input data and blocking.")
    if not val_feature_dfs:
        raise ValueError("No validation candidate features were generated; check the input data and blocking.")

    full_train_df = pd.concat(train_feature_dfs, ignore_index=True)
    positive_count = int((full_train_df["label"] == 1).sum())
    negative_count = len(full_train_df) - positive_count
    scale_pos_weight = min(max(negative_count / max(positive_count, 1), 1.0), 5.0)

    # Keep validation runs from overwriting the production model artifact.
    with TemporaryDirectory() as temp_dir:
        clf = train_lightgbm_model(
            full_train_df,
            target_col="label",
            model_path=Path(temp_dir) / "validation_model.txt",
            params={"scale_pos_weight": scale_pos_weight}
        )

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
        threshold_range=threshold_range
    )
    s1_country_by_id = dict(zip(
        df_s1_val["entity_id"],
        df_s1_val["country"].fillna("").astype(str)
    ))
    pooled_best_f05 = best_f05
    country_thresholds = {}
    country_labels = sorted(set(s1_country_by_id.values()))
    for country in country_labels:
        country_ids = [
            s1_id for s1_id in val_s1_ids
            if s1_country_by_id[s1_id] == country
        ]
        if len(country_ids) < 1000:
            continue
        country_threshold, country_best_f05 = find_optimal_threshold(
            df_scored_val_pairs=val_scored_df,
            val_s1_ids=country_ids,
            ground_truth=gt_map,
            threshold_range=threshold_range
        )
        country_thresholds[country] = country_threshold
        print(
            f"  Country threshold {country or FALLBACK_COUNTRY}: "
            f"tau={country_threshold:.6f}, Macro F_0.5={country_best_f05:.5f}"
        )

    proposed_threshold_by_s1 = {
        s1_id: country_thresholds.get(s1_country_by_id[s1_id], best_tau)
        for s1_id in val_s1_ids
    }
    country_calibrated_preds = select_matches_from_scored_pairs(
        df_scored_pairs=val_scored_df,
        all_s1_ids=val_s1_ids,
        threshold=best_tau,
        threshold_by_s1=proposed_threshold_by_s1
    )
    country_calibrated_f05 = compute_macro_f05(
        country_calibrated_preds, gt_map, entity_subset=val_s1_ids
    )
    print(
        f"Country-calibrated validation Macro F_0.5: {country_calibrated_f05:.5f} "
        f"(pooled threshold: {pooled_best_f05:.5f})"
    )
    if country_calibrated_f05 > pooled_best_f05:
        threshold_by_s1 = proposed_threshold_by_s1
        best_preds = country_calibrated_preds
        best_f05 = country_calibrated_f05
        print("Using country-specific thresholds: validation score improved.")
    else:
        threshold_by_s1 = None
        country_thresholds = {}
        best_preds = select_matches_from_scored_pairs(
            df_scored_pairs=val_scored_df,
            all_s1_ids=val_s1_ids,
            threshold=best_tau
        )
        best_f05 = pooled_best_f05
        print("Using pooled threshold: country thresholds did not improve validation.")

    if run_ablations:
        ablation_candidate_maps["cap_150"] = {
            s1_id: candidates[:150]
            for s1_id, candidates in val_cand_mapping.items()
        }
        ablation_candidate_maps["cap_300"] = {
            s1_id: candidates[:300]
            for s1_id, candidates in val_cand_mapping.items()
        }
        baseline_pair_count = sum(len(ids) for ids in val_cand_mapping.values())
        baseline_gt_pair_count = sum(
            len(gt_map.get(s1_id, set())) for s1_id in val_s1_ids
        )
        baseline_retrieved_pairs = sum(
            len(gt_map.get(s1_id, set()) & set(val_cand_mapping.get(s1_id, [])))
            for s1_id in val_s1_ids
        )
        baseline_non_singletons = [
            s1_id for s1_id in val_s1_ids if gt_map.get(s1_id, set())
        ]
        baseline_full_coverage_count = sum(
            gt_map[s1_id].issubset(set(val_cand_mapping.get(s1_id, [])))
            for s1_id in baseline_non_singletons
        )
        baseline_candidate_recall = (
            baseline_retrieved_pairs / baseline_gt_pair_count
            if baseline_gt_pair_count else 0.0
        )
        baseline_full_coverage = (
            baseline_full_coverage_count / len(baseline_non_singletons)
            if baseline_non_singletons else 0.0
        )
        print(
            "\n[Blocking Ablations] Baseline: "
            f"pairs={baseline_pair_count:,}, candidate_recall={baseline_candidate_recall:.4f}, "
            f"full_coverage={baseline_full_coverage:.4f}, "
            f"pooled_macroF0.5={pooled_best_f05:.5f}, "
            f"selected_macroF0.5={best_f05:.5f}"
        )
        print(
            "[Blocking Ablations] Each variant removes one enabled retrieval channel, "
            "adds the disabled prefix channel, or tests exact 150/300/500 candidate budgets."
        )
        for variant, candidate_map in ablation_candidate_maps.items():
            pair_count = sum(len(ids) for ids in candidate_map.values())
            true_pairs = sum(
                len(ground_truth_ids & set(candidate_map.get(s1_id, [])))
                for s1_id, ground_truth_ids in (
                    (s1_id, gt_map.get(s1_id, set())) for s1_id in val_s1_ids
                )
            )
            total_pairs = sum(len(gt_map.get(s1_id, set())) for s1_id in val_s1_ids)
            non_singleton_ids = [
                s1_id for s1_id in val_s1_ids if gt_map.get(s1_id, set())
            ]
            fully_covered = sum(
                gt_map[s1_id].issubset(set(candidate_map.get(s1_id, [])))
                for s1_id in non_singleton_ids
            )

            variant_feature_dfs = []
            for country in partitions:
                df_s1_val_country = _filter_country(df_s1_val, country)
                df_target_country = _filter_country(df_target, country)
                if df_s1_val_country.empty or df_target_country.empty:
                    continue
                country_ids = set(df_s1_val_country["entity_id"])
                country_candidate_map = {
                    s1_id: candidate_map.get(s1_id, [])
                    for s1_id in country_ids
                }
                variant_features = FeaturePipeline.extract_features_for_pairs(
                    country_candidate_map,
                    df_s1_val_country,
                    df_target_country,
                    country="" if country == FALLBACK_COUNTRY else country
                )
                if not variant_features.empty:
                    variant_feature_dfs.append(variant_features)

            if variant_feature_dfs:
                variant_features = pd.concat(variant_feature_dfs, ignore_index=True)
                variant_scored = predict_pair_probabilities(
                    variant_features, model=clf.booster_
                )
                variant_tau, variant_f05 = find_optimal_threshold(
                    variant_scored,
                    val_s1_ids,
                    gt_map
                )
            else:
                variant_tau = 1.0
                variant_f05 = compute_macro_f05(
                    {}, gt_map, entity_subset=val_s1_ids
                )

            candidate_recall = true_pairs / total_pairs if total_pairs else 0.0
            coverage = (
                fully_covered / len(non_singleton_ids)
                if non_singleton_ids else 0.0
            )
            print(
                f"  {variant}: pairs={pair_count:,}, "
                f"candidate_recall={candidate_recall:.4f}, "
                f"full_coverage={coverage:.4f}, "
                f"best_macroF0.5={variant_f05:.5f}, "
                f"delta_vs_pooled={variant_f05 - pooled_best_f05:+.5f}, "
                f"tau={variant_tau:.6f}"
            )
            del variant_feature_dfs
            if "variant_features" in locals():
                del variant_features
            if "variant_scored" in locals():
                del variant_scored
            gc.collect()

    target_country_by_id = dict(zip(
        df_target["entity_id"],
        df_target["country"].fillna("").astype(str)
    ))
    country_metrics = defaultdict(lambda: {
        "entities": 0,
        "gt_pairs": 0,
        "retrieved_pairs": 0,
        "fully_covered": 0,
        "non_singletons": 0,
        "singletons": 0,
        "correct_singletons": 0,
        "tp": 0,
        "fp": 0,
        "fn": 0
    })
    total_val_gt = 0
    retrieved_val_gt = 0
    fully_covered_entities = 0
    val_non_singletons = 0
    cross_country_gt_pairs = 0
    missing_gt_target_ids = 0
    for s1_id in val_s1_ids:
        gt_set = gt_map.get(s1_id, set())
        candidate_set = set(val_cand_mapping.get(s1_id, []))
        pred_set = set(best_preds.get(s1_id, set()))
        country = s1_country_by_id.get(s1_id, "") or FALLBACK_COUNTRY
        metrics = country_metrics[country]
        metrics["entities"] += 1
        metrics["gt_pairs"] += len(gt_set)
        metrics["retrieved_pairs"] += len(gt_set & candidate_set)

        for target_id in gt_set:
            target_country = target_country_by_id.get(target_id)
            if target_country is None:
                missing_gt_target_ids += 1
            elif target_country != s1_country_by_id.get(s1_id, ""):
                cross_country_gt_pairs += 1

        total_val_gt += len(gt_set)
        retrieved_val_gt += len(gt_set & candidate_set)
        if gt_set:
            val_non_singletons += 1
            fully_covered_entities += int(gt_set.issubset(candidate_set))
            metrics["non_singletons"] += 1
            metrics["fully_covered"] += int(gt_set.issubset(candidate_set))
            metrics["tp"] += len(pred_set & gt_set)
            metrics["fp"] += len(pred_set - gt_set)
            metrics["fn"] += len(gt_set - pred_set)
        else:
            metrics["singletons"] += 1
            metrics["correct_singletons"] += int(not pred_set)
    blocker_recall = retrieved_val_gt / total_val_gt if total_val_gt else 0.0
    full_candidate_coverage = (
        fully_covered_entities / val_non_singletons if val_non_singletons else 0.0
    )
    oracle_predictions = {
        s1_id: gt_map.get(s1_id, set()) & set(val_cand_mapping.get(s1_id, []))
        for s1_id in val_s1_ids
    }
    candidate_oracle_f05 = compute_macro_f05(
        oracle_predictions, gt_map, entity_subset=val_s1_ids
    )
    print(
        f"  * Candidate Recall:            {blocker_recall:.4f} "
        f"({retrieved_val_gt}/{total_val_gt} true pairs retrieved)"
    )
    print(
        f"  * Full Match-Set Coverage:     {full_candidate_coverage:.4f} "
        f"({fully_covered_entities}/{val_non_singletons} non-singletons fully covered)"
    )
    print(
        f"  * Candidate-oracle Macro F_0.5: {candidate_oracle_f05:.4f} "
        "(perfect classification within generated candidates)"
    )
    print(
        f"  * Cross-country GT pairs:      {cross_country_gt_pairs:,}; "
        f"GT IDs absent from targets: {missing_gt_target_ids:,}"
    )
    _report_blocking_miss_signatures(
        val_s1_ids=val_s1_ids,
        ground_truth=gt_map,
        candidates=val_cand_mapping,
        df_s1=df_s1_val,
        df_target=df_target
    )

    # Error Analysis on Validation Set
    print("\nDetailed Error Analysis at pooled tau = {:.6f}:".format(best_tau))
    for country, country_tau in sorted(country_thresholds.items()):
        print(f"  Threshold override {country or FALLBACK_COUNTRY}: tau={country_tau:.6f}")

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
    print("  * Per-country validation:")
    for country in sorted(country_metrics):
        metrics = country_metrics[country]
        country_ids = [
            s1_id for s1_id in val_s1_ids
            if (s1_country_by_id.get(s1_id, "") or FALLBACK_COUNTRY) == country
        ]
        country_tp = metrics["tp"]
        country_fp = metrics["fp"]
        country_fn = metrics["fn"]
        country_precision = (
            country_tp / (country_tp + country_fp)
            if country_tp + country_fp else 0.0
        )
        country_recall = (
            country_tp / (country_tp + country_fn)
            if country_tp + country_fn else 0.0
        )
        country_singleton_accuracy = (
            metrics["correct_singletons"] / metrics["singletons"]
            if metrics["singletons"] else 0.0
        )
        country_candidate_recall = (
            metrics["retrieved_pairs"] / metrics["gt_pairs"]
            if metrics["gt_pairs"] else 0.0
        )
        country_full_coverage = (
            metrics["fully_covered"] / metrics["non_singletons"]
            if metrics["non_singletons"] else 0.0
        )
        country_f05 = compute_macro_f05(
            best_preds, gt_map, entity_subset=country_ids
        )
        print(
            f"    {country}: n={metrics['entities']:,}, macroF0.5={country_f05:.4f}, "
            f"candidate_recall={country_candidate_recall:.4f}, "
            f"full_coverage={country_full_coverage:.4f}, "
            f"precision={country_precision:.4f}, recall={country_recall:.4f}, "
            f"singleton_acc={country_singleton_accuracy:.4f}"
        )
    print("=======================================================\n")

    return {
        "best_tau": best_tau,
        "macro_f05": best_f05,
        "pooled_macro_f05": pooled_best_f05,
        "precision": precision,
        "recall": recall,
        "singleton_accuracy": singleton_acc,
        "candidate_recall": blocker_recall,
        "full_candidate_coverage": full_candidate_coverage,
        "candidate_oracle_macro_f05": candidate_oracle_f05,
        **{
            f"threshold_{country or 'unknown'}": tau
            for country, tau in country_thresholds.items()
        }
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run entity-level validation and candidate-coverage diagnostics."
    )
    parser.add_argument("--train-dir", type=Path, default=TRAIN_DIR)
    parser.add_argument(
        "--sample-size",
        type=int,
        default=100_000,
        help="Randomly sample this many S1 entities; use 0 for all S1 entities."
    )
    parser.add_argument("--val-fraction", type=float, default=0.20)
    parser.add_argument(
        "--threshold-range",
        type=float,
        nargs=3,
        metavar=("START", "STOP", "STEP"),
        default=None,
        help="Optional threshold sweep (start stop step); default checks distinct observed scores."
    )
    parser.add_argument(
        "--ablations",
        action="store_true",
        help="Measure each blocking channel and compare exact 150-, 300-, and 500-candidate budgets."
    )
    args = parser.parse_args()
    run_entity_validation_pipeline(
        train_dir=args.train_dir,
        val_fraction=args.val_fraction,
        sample_size=args.sample_size,
        threshold_range=tuple(args.threshold_range) if args.threshold_range else None,
        run_ablations=args.ablations
    )


if __name__ == "__main__":
    main()
