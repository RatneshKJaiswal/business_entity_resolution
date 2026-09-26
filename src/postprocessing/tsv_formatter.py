"""
TSV Formatter generating official submission file matching_results.tsv.
"""

from pathlib import Path
from typing import Dict, List, Set

from ..utils.io_utils import write_submission_tsv


def save_matching_results(
    matches: Dict[str, Set[str]],
    required_s1_order: List[str],
    output_path: Path
) -> None:
    """
    Writes out matching_results.tsv strictly following competition guidelines:
    - Tab-separated
    - Header: source1_entity_id \t matched_entity_ids
    - Exactly one row per S1 entity
    - Clean UTF-8 encoding
    """
    write_submission_tsv(
        path=output_path,
        header_col1="source1_entity_id",
        header_col2="matched_entity_ids",
        mapping=matches,
        required_s1_order=required_s1_order
    )
