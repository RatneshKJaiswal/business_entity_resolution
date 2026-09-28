"""Matcher applying a calibrated threshold to candidates from each target source."""

from typing import Dict, List, Optional, Set
from collections import defaultdict
import pandas as pd


def select_matches_from_scored_pairs(
    df_scored_pairs: pd.DataFrame,
    all_s1_ids: List[str],
    threshold: float = 0.50,
    max_per_source: Optional[int] = None,
    threshold_by_s1: Optional[Dict[str, float]] = None
) -> Dict[str, Set[str]]:
    """
    Applies calibrated decision threshold to select matching pairs.
    Separates candidates by target source (S2 vs S3). By default, all candidates
    above the threshold are retained; an optional per-source limit is available
    for controlled experiments.
    
    Returns:
        Dict mapping each S1 entity ID to a set of matching S2/S3 entity IDs.
    """
    # Group candidates by S1 ID and source prefix
    s2_by_s1 = defaultdict(list)
    s3_by_s1 = defaultdict(list)

    for row in df_scored_pairs.itertuples(index=False):
        s1 = getattr(row, "source1_entity_id")
        cand = getattr(row, "candidate_entity_id")
        prob = getattr(row, "match_prob")

        if cand.startswith("S2-"):
            s2_by_s1[s1].append((cand, prob))
        elif cand.startswith("S3-"):
            s3_by_s1[s1].append((cand, prob))

    matches: Dict[str, Set[str]] = {}

    for s1_id in all_s1_ids:
        selected: Set[str] = set()
        entity_threshold = (
            threshold_by_s1.get(s1_id, threshold)
            if threshold_by_s1 is not None
            else threshold
        )

        # Process S2 candidates
        s2_cands = s2_by_s1.get(s1_id, [])
        if s2_cands:
            s2_cands.sort(key=lambda x: x[1], reverse=True)
            selected_s2 = s2_cands if max_per_source is None else s2_cands[:max_per_source]
            for cand_id, prob in selected_s2:
                if prob >= entity_threshold:
                    selected.add(cand_id)

        # Process S3 candidates
        s3_cands = s3_by_s1.get(s1_id, [])
        if s3_cands:
            s3_cands.sort(key=lambda x: x[1], reverse=True)
            selected_s3 = s3_cands if max_per_source is None else s3_cands[:max_per_source]
            for cand_id, prob in selected_s3:
                if prob >= entity_threshold:
                    selected.add(cand_id)

        matches[s1_id] = selected

    return matches
