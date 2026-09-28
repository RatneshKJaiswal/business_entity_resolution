"""
Threshold optimizer searching for the probability threshold that maximizes Macro F_0.5.
Uses a coarse-to-fine sweep by default, or an explicit caller-supplied sweep.
"""

from typing import Dict, List, Optional, Set, Tuple
import numpy as np
import pandas as pd


def find_optimal_threshold(
    df_scored_val_pairs: pd.DataFrame,
    val_s1_ids: List[str],
    ground_truth: Dict[str, Set[str]],
    threshold_range: Optional[Tuple[float, float, float]] = None,
) -> Tuple[float, float]:
    """
    Sweeps thresholds to maximize Macro F_0.5 on held-out validation entities.
    By default, prints and evaluates the legacy coarse-to-fine threshold schedule.
    A supplied range performs one explicit fixed-step sweep.
    
    Returns:
        (best_threshold, best_f05_score)
    """
    if not val_s1_ids:
        return 1.0, 0.0

    entity_index = {s1_id: index for index, s1_id in enumerate(val_s1_ids)}
    gt_sets = [ground_truth.get(s1_id, set()) for s1_id in val_s1_ids]
    gt_counts = np.asarray([len(ids) for ids in gt_sets], dtype=np.int64)
    entity_count = len(val_s1_ids)
    baseline_score_sum = float(np.count_nonzero(gt_counts == 0))

    valid = df_scored_val_pairs[
        df_scored_val_pairs["source1_entity_id"].isin(entity_index)
        & df_scored_val_pairs["candidate_entity_id"].str.startswith(("S2-", "S3-"))
        & df_scored_val_pairs["match_prob"].notna()
    ][["source1_entity_id", "candidate_entity_id", "match_prob"]]
    valid = valid.groupby(
        ["source1_entity_id", "candidate_entity_id"], as_index=False
    )["match_prob"].max()

    pair_counts = valid.groupby(
        ["source1_entity_id", "match_prob"], sort=False
    ).size().rename("pred_increment").reset_index()
    gt_pair_rows = [
        (s1_id, candidate_id)
        for s1_id in val_s1_ids
        for candidate_id in gt_sets[entity_index[s1_id]]
    ]
    if gt_pair_rows:
        gt_pairs = pd.DataFrame(
            gt_pair_rows,
            columns=["source1_entity_id", "candidate_entity_id"]
        )
        positive_counts = (
            gt_pairs.merge(
                valid,
                on=["source1_entity_id", "candidate_entity_id"],
                how="inner",
                copy=False
            )
            .groupby(["source1_entity_id", "match_prob"], sort=False)
            .size()
            .rename("tp_increment")
            .reset_index()
        )
        pair_counts = pair_counts.merge(
            positive_counts,
            on=["source1_entity_id", "match_prob"],
            how="left",
            copy=False
        )
    else:
        pair_counts["tp_increment"] = 0

    pair_counts["tp_increment"] = pair_counts["tp_increment"].fillna(0).astype(np.int64)
    pair_counts["entity_index"] = pair_counts["source1_entity_id"].map(entity_index).astype(np.int64)
    pair_counts = pair_counts.sort_values("match_prob", ascending=False, kind="mergesort")
    probabilities = pair_counts["match_prob"].to_numpy(dtype=np.float64)
    pair_entity_indices = pair_counts["entity_index"].to_numpy(dtype=np.int64)
    prediction_increments = pair_counts["pred_increment"].to_numpy(dtype=np.int64)
    true_positive_increments = pair_counts["tp_increment"].to_numpy(dtype=np.int64)
    score_denominators = 0.25 * gt_counts

    def sweep(thresholds: np.ndarray, best_tau: float, best_score: float):
        pred_counts = np.zeros(entity_count, dtype=np.int64)
        tp_counts = np.zeros(entity_count, dtype=np.int64)
        row_index = 0
        threshold_scores = {}

        ordered_thresholds = sorted(set(float(value) for value in thresholds))
        for tau in reversed(ordered_thresholds):
            next_index = int(np.searchsorted(-probabilities, -tau, side="right"))
            if next_index > row_index:
                indices = pair_entity_indices[row_index:next_index]
                np.add.at(pred_counts, indices, prediction_increments[row_index:next_index])
                np.add.at(tp_counts, indices, true_positive_increments[row_index:next_index])
                row_index = next_index

            entity_scores = np.divide(
                1.25 * tp_counts,
                score_denominators + pred_counts,
                out=np.zeros(entity_count, dtype=np.float64),
                where=(gt_counts > 0) & (tp_counts > 0)
            )
            singleton_mask = gt_counts == 0
            entity_scores[singleton_mask] = (pred_counts[singleton_mask] == 0)
            score = float(entity_scores.mean())
            threshold_scores[tau] = score

        ordered_results = []
        for tau in ordered_thresholds:
            score = threshold_scores[tau]
            print(f"  tau = {tau:.3f} -> Validation Macro F_0.5: {score:.5f}")
            ordered_results.append((tau, score))
            if score > best_score:
                best_tau, best_score = tau, score

        return best_tau, best_score, ordered_results

    if not len(probabilities):
        best_tau = 1.0
        best_score = baseline_score_sum / entity_count
        print(f"\n-> Optimal Threshold: tau = {best_tau:.6f} with Macro F_0.5 = {best_score:.5f}\n")
        return best_tau, best_score

    best_tau = 0.50
    best_score = -1.0

    if threshold_range is not None:
        start, stop, step = threshold_range
        if step <= 0 or stop < start:
            raise ValueError("threshold_range must have positive step and stop >= start")
        thresholds = np.arange(start, stop + step * 0.5, step)
        print(
            f"\n[Threshold Optimization] Sweeping requested range "
            f"[{start:.3f}, {stop:.3f}] (step={step:.3f})..."
        )
        best_tau, best_score, _ = sweep(thresholds, best_tau, best_score)
    else:
        coarse_thresholds = np.arange(0.10, 0.985, 0.05)
        print("\n[Threshold Optimization] Stage 1: Coarse sweep across probability spectrum...")
        best_tau, best_score, coarse_results = sweep(
            coarse_thresholds, best_tau, best_score
        )

        fine_min = max(0.05, best_tau - 0.06)
        fine_max = min(0.995, best_tau + 0.06)
        fine_thresholds = np.arange(fine_min, fine_max + 1e-5, 0.005)
        coarse_values = {tau for tau, _ in coarse_results}
        fine_thresholds = np.asarray([
            tau for tau in fine_thresholds
            if all(abs(float(tau) - coarse_tau) >= 1e-4 for coarse_tau in coarse_values)
        ])
        print(
            f"\n[Threshold Optimization] Stage 2: Fine sweep in "
            f"[{fine_min:.3f}, {fine_max:.3f}] (step=0.005)..."
        )
        best_tau, best_score, _ = sweep(
            fine_thresholds, best_tau, best_score
        )

    print(f"\n-> Optimal Threshold: tau = {best_tau:.6f} with Macro F_0.5 = {best_score:.5f}\n")
    return best_tau, best_score
