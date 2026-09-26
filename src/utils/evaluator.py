"""
Official Macro F_0.5 metric computation for Entity Resolution.
"""

from typing import Dict, Set, Iterable


def compute_entity_f05(predicted_ids: Set[str], ground_truth_ids: Set[str]) -> float:
    """
    Computes the F_0.5 score for a single Source 1 entity.
    
    Precision-weighted formula:
        F_0.5 = (1.25 * Precision * Recall) / (0.25 * Precision + Recall)
        
    Rules for singletons:
    - If true matches == empty and predicted matches == empty -> score is 1.0
    - If true matches == empty and predicted matches != empty -> score is 0.0
    - If true matches != empty and predicted matches == empty -> score is 0.0
    """
    is_pred_empty = len(predicted_ids) == 0
    is_gt_empty = len(ground_truth_ids) == 0

    if is_gt_empty:
        return 1.0 if is_pred_empty else 0.0

    if is_pred_empty:
        return 0.0

    # True positives
    tp = len(predicted_ids & ground_truth_ids)
    if tp == 0:
        return 0.0

    precision = tp / len(predicted_ids)
    recall = tp / len(ground_truth_ids)

    denom = (0.25 * precision) + recall
    if denom == 0.0:
        return 0.0

    return (1.25 * precision * recall) / denom


def compute_macro_f05(
    predictions: Dict[str, Set[str]],
    ground_truth: Dict[str, Set[str]],
    entity_subset: Iterable[str] = None
) -> float:
    """
    Computes the Macro-Averaged F_0.5 score across all specified S1 entities.
    """
    entities = entity_subset if entity_subset is not None else ground_truth.keys()
    total_score = 0.0
    count = 0

    for s1_id in entities:
        gt_set = ground_truth.get(s1_id, set())
        pred_set = predictions.get(s1_id, set())
        total_score += compute_entity_f05(pred_set, gt_set)
        count += 1

    return total_score / count if count > 0 else 0.0
