"""
Threshold optimizer searching for the probability threshold that maximizes Macro F_0.5.
Uses two-stage coarse-to-fine search to guarantee finding the global maximum without ceiling clipping.
"""

from typing import Dict, List, Set, Tuple
import numpy as np
import pandas as pd

from ..utils.evaluator import compute_macro_f05
from .matcher import select_matches_from_scored_pairs


def find_optimal_threshold(
    df_scored_val_pairs: pd.DataFrame,
    val_s1_ids: List[str],
    ground_truth: Dict[str, Set[str]],
    threshold_range: Tuple[float, float, float] = None,
) -> Tuple[float, float]:
    """
    Sweeps decision threshold tau to maximize Macro F_0.5 on held-out validation entities.
    Employs a two-stage coarse-to-fine search covering up to 0.995 to avoid boundary truncation.
    
    Returns:
        (best_threshold, best_f05_score)
    """
    print("\n[Threshold Optimization] Stage 1: Coarse sweep across probability spectrum...")

    # Stage 1: Coarse sweep from 0.10 to 0.98 in steps of 0.05
    coarse_thresholds = np.arange(0.10, 0.985, 0.05)
    best_tau = 0.50
    best_score = -1.0

    for tau in coarse_thresholds:
        preds = select_matches_from_scored_pairs(
            df_scored_pairs=df_scored_val_pairs,
            all_s1_ids=val_s1_ids,
            threshold=float(tau)
        )
        score = compute_macro_f05(preds, ground_truth, entity_subset=val_s1_ids)
        print(f"  tau = {tau:.2f} -> Validation Macro F_0.5: {score:.5f}")

        if score > best_score:
            best_score = score
            best_tau = float(tau)

    # Stage 2: Fine sweep around the best coarse threshold (±0.06 in steps of 0.005)
    fine_min = max(0.05, best_tau - 0.06)
    fine_max = min(0.995, best_tau + 0.06)
    fine_thresholds = np.arange(fine_min, fine_max + 1e-5, 0.005)

    print(f"\n[Threshold Optimization] Stage 2: Fine sweep in [{fine_min:.3f}, {fine_max:.3f}] (step=0.005)...")
    for tau in fine_thresholds:
        # Skip if already computed in coarse sweep
        if any(abs(tau - c) < 1e-4 for c in coarse_thresholds):
            continue

        preds = select_matches_from_scored_pairs(
            df_scored_pairs=df_scored_val_pairs,
            all_s1_ids=val_s1_ids,
            threshold=float(tau)
        )
        score = compute_macro_f05(preds, ground_truth, entity_subset=val_s1_ids)
        print(f"  tau = {tau:.3f} -> Validation Macro F_0.5: {score:.5f}")

        if score > best_score:
            best_score = score
            best_tau = float(tau)

    print(f"\n-> Optimal Threshold: tau = {best_tau:.3f} with Macro F_0.5 = {best_score:.5f}\n")
    return best_tau, best_score
