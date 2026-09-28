"""
Candidate Builder coordinating blocking across countries with parallel execution.
Handles entities with missing countries via a fallback partition.
"""
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import pandas as pd

from ..config import N_JOBS, MAX_CANDIDATES_PER_S1
from ..preprocessing.text_cleaner import strip_legal_suffixes, get_name_tokens
from ..preprocessing.address_parser import extract_postal_code
from ..utils.io_utils import write_submission_tsv
from .inverted_index_blocker import (
    BLOCKING_CHANNELS,
    DEFAULT_BLOCKING_CHANNELS,
    FastInvertedIndexBlocker
)

FALLBACK_COUNTRY = "__UNKNOWN__"


def prune_training_candidates(
    candidate_mapping: Dict[str, List[str]],
    ground_truth: Dict[str, Set[str]],
    max_negatives: int = 30,
    head_negatives: int = 20
) -> Dict[str, List[str]]:
    """Keep all positives and representative high- and lower-ranked negatives."""
    if max_negatives < 0 or head_negatives < 0:
        raise ValueError("Negative candidate limits must be non-negative")

    pruned = {}
    for s1_id, candidates in candidate_mapping.items():
        true_matches = ground_truth.get(s1_id, set())
        positives = [candidate for candidate in candidates if candidate in true_matches]
        negatives = [candidate for candidate in candidates if candidate not in true_matches]

        if len(negatives) > max_negatives:
            head_count = min(head_negatives, max_negatives)
            spread_count = max_negatives - head_count
            selected_negatives = negatives[:head_count]
            if spread_count:
                remaining_count = len(negatives) - head_count
                spread_indices = [
                    head_count
                    + round(i * (remaining_count - 1) / max(spread_count - 1, 1))
                    for i in range(spread_count)
                ]
                selected_negatives.extend(negatives[index] for index in spread_indices)
            negatives = selected_negatives

        pruned[s1_id] = positives + negatives
    return pruned


# Worker function for process pool
_GLOBAL_BLOCKER = None
_GLOBAL_FALLBACK_BLOCKER = None
_GLOBAL_EFFECTIVE_COUNTRY = ""
_GLOBAL_TOP_K = MAX_CANDIDATES_PER_S1
_GLOBAL_ENABLED_CHANNELS = DEFAULT_BLOCKING_CHANNELS


def _init_worker(blocker, fallback_blocker, effective_country, top_k, enabled_channels):
    global _GLOBAL_BLOCKER, _GLOBAL_FALLBACK_BLOCKER, _GLOBAL_EFFECTIVE_COUNTRY
    global _GLOBAL_TOP_K, _GLOBAL_ENABLED_CHANNELS
    _GLOBAL_BLOCKER = blocker
    _GLOBAL_FALLBACK_BLOCKER = fallback_blocker
    _GLOBAL_EFFECTIVE_COUNTRY = effective_country
    _GLOBAL_TOP_K = top_k
    _GLOBAL_ENABLED_CHANNELS = enabled_channels


def _process_s1_batch(batch_records: List[Tuple[str, str, str]]) -> List[Tuple[str, List[str]]]:
    results = []
    for s1_id, name, addr in batch_records:
        cn = strip_legal_suffixes(name, _GLOBAL_EFFECTIVE_COUNTRY)
        tokens = get_name_tokens(cn)
        postal = extract_postal_code(addr, _GLOBAL_EFFECTIVE_COUNTRY)
        candidates = []

        if _GLOBAL_BLOCKER:
            candidates = _GLOBAL_BLOCKER.retrieve_candidates(
                cn, tokens, postal, s1_address=addr, top_k=_GLOBAL_TOP_K,
                enabled_channels=_GLOBAL_ENABLED_CHANNELS
            )

        if _GLOBAL_FALLBACK_BLOCKER:
            fb_candidates = _GLOBAL_FALLBACK_BLOCKER.retrieve_candidates(
                cn, tokens, postal, s1_address=addr, top_k=max(10, _GLOBAL_TOP_K // 5),
                enabled_channels=_GLOBAL_ENABLED_CHANNELS
            )
            seen = set(candidates)
            for c in fb_candidates:
                if c not in seen:
                    candidates.append(c)

        results.append((s1_id, candidates))
    return results


class CandidateBuilder:
    """
    Coordinates blocking partitioned by country and generates candidate pairs.
    """
    def __init__(self, top_k: int = MAX_CANDIDATES_PER_S1):
        self.top_k = top_k
        self.blockers: Dict[str, FastInvertedIndexBlocker] = {}

    def index_target_records(
        self,
        df_target: pd.DataFrame,
        country: str
    ) -> None:
        """
        Indexes target records for a specific country partition.
        """
        blocker = FastInvertedIndexBlocker()
        entity_ids = df_target["entity_id"].tolist()
        names = df_target["business_name"].fillna("").tolist()
        addresses = df_target["business_address"].fillna("").tolist()
        effective_country = country if country and country != FALLBACK_COUNTRY else ""

        print(f"      Tokenizing and building index for {len(entity_ids):,} {country} target records...")
        core_names = [strip_legal_suffixes(n, effective_country) for n in tqdm(names, desc="      Cleaning names", unit="rec", leave=False)]
        token_sets = [get_name_tokens(cn) for cn in tqdm(core_names, desc="      Extracting tokens", unit="rec", leave=False)]
        postal_codes = [extract_postal_code(addr, effective_country) for addr in tqdm(addresses, desc="      Extracting postal", unit="rec", leave=False)]

        blocker.fit(entity_ids, core_names, token_sets, postal_codes, addresses)
        self.blockers[country] = blocker

    def generate_candidates(
        self,
        df_s1: pd.DataFrame,
        country: str,
        enabled_channels: Optional[Set[str]] = None,
        top_k: Optional[int] = None
    ) -> Dict[str, List[str]]:
        """
        Generates candidate IDs for all S1 entities using parallel batch processing.
        """
        blocker = self.blockers.get(country)
        fallback_blocker = self.blockers.get(FALLBACK_COUNTRY) if country != FALLBACK_COUNTRY else None

        s1_ids = df_s1["entity_id"].tolist()
        names = df_s1["business_name"].fillna("").tolist()
        addresses = df_s1["business_address"].fillna("").tolist()
        effective_country = country if country and country != FALLBACK_COUNTRY else ""

        records = list(zip(s1_ids, names, addresses))
        total_records = len(records)

        # Determine batch chunks for multiprocessing
        batch_size = 2000
        batches = [records[i:i + batch_size] for i in range(0, total_records, batch_size)]

        num_workers = min(N_JOBS, 8)
        results: Dict[str, List[str]] = {}
        channels = (
            DEFAULT_BLOCKING_CHANNELS
            if enabled_channels is None
            else frozenset(enabled_channels)
        )
        candidate_limit = self.top_k if top_k is None else top_k
        if candidate_limit <= 0:
            raise ValueError("top_k must be greater than zero")
        unknown_channels = channels - BLOCKING_CHANNELS
        if unknown_channels:
            raise ValueError(f"Unknown blocking channels: {sorted(unknown_channels)}")

        # Fallback to single process if running small batches or 1 CPU
        if num_workers <= 1 or len(batches) <= 1:
            _init_worker(blocker, fallback_blocker, effective_country, candidate_limit, channels)
            for batch in tqdm(batches, desc=f"    [b] Blocking candidates ({country})", ncols=100):
                for s1_id, cands in _process_s1_batch(batch):
                    results[s1_id] = cands
            return results

        with ProcessPoolExecutor(
            max_workers=num_workers,
            initializer=_init_worker,
            initargs=(blocker, fallback_blocker, effective_country, candidate_limit, channels)
        ) as executor:
            for batch_result in tqdm(
                executor.map(_process_s1_batch, batches),
                total=len(batches),
                desc=f"    [b] Blocking candidates ({country} x{num_workers})",
                ncols=100
            ):
                for s1_id, cands in batch_result:
                    results[s1_id] = cands

        return results

    def save_candidate_pairs(
        self,
        candidate_mapping: Dict[str, List[str]],
        output_path: Path,
        required_s1_order: List[str]
    ) -> None:
        """
        Saves candidate map into matching candidate_pairs.tsv.
        """
        set_mapping = {k: set(v) for k, v in candidate_mapping.items()}
        write_submission_tsv(
            path=output_path,
            header_col1="source1_entity_id",
            header_col2="candidate_entity_ids",
            mapping=set_mapping,
            required_s1_order=required_s1_order
        )