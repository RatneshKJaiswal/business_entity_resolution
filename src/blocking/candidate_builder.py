"""
Candidate Builder coordinating blocking across countries with parallel execution.
Handles entities with missing countries via a fallback partition.
"""
from pathlib import Path
from typing import Dict, List, Set, Tuple
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import pandas as pd

from ..config import N_JOBS, MAX_CANDIDATES_PER_S1
from ..preprocessing.text_cleaner import strip_legal_suffixes, get_name_tokens
from ..preprocessing.address_parser import extract_postal_code
from ..utils.io_utils import write_submission_tsv
from .inverted_index_blocker import FastInvertedIndexBlocker

FALLBACK_COUNTRY = "__UNKNOWN__"

# Worker function for process pool
_GLOBAL_BLOCKER = None
_GLOBAL_FALLBACK_BLOCKER = None
_GLOBAL_EFFECTIVE_COUNTRY = ""
_GLOBAL_TOP_K = MAX_CANDIDATES_PER_S1


def _init_worker(blocker, fallback_blocker, effective_country, top_k):
    global _GLOBAL_BLOCKER, _GLOBAL_FALLBACK_BLOCKER, _GLOBAL_EFFECTIVE_COUNTRY, _GLOBAL_TOP_K
    _GLOBAL_BLOCKER = blocker
    _GLOBAL_FALLBACK_BLOCKER = fallback_blocker
    _GLOBAL_EFFECTIVE_COUNTRY = effective_country
    _GLOBAL_TOP_K = top_k


def _process_s1_batch(batch_records: List[Tuple[str, str, str]]) -> List[Tuple[str, List[str]]]:
    results = []
    for s1_id, name, addr in batch_records:
        cn = strip_legal_suffixes(name, _GLOBAL_EFFECTIVE_COUNTRY)
        tokens = get_name_tokens(cn)
        postal = extract_postal_code(addr, _GLOBAL_EFFECTIVE_COUNTRY)
        candidates = []

        if _GLOBAL_BLOCKER:
            candidates = _GLOBAL_BLOCKER.retrieve_candidates(cn, tokens, postal, s1_address=addr, top_k=_GLOBAL_TOP_K)

        if _GLOBAL_FALLBACK_BLOCKER:
            fb_candidates = _GLOBAL_FALLBACK_BLOCKER.retrieve_candidates(
                cn, tokens, postal, s1_address=addr, top_k=max(10, _GLOBAL_TOP_K // 5)
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
        country: str
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

        # Fallback to single process if running small batches or 1 CPU
        if num_workers <= 1 or len(batches) <= 1:
            _init_worker(blocker, fallback_blocker, effective_country, self.top_k)
            for batch in tqdm(batches, desc=f"    [b] Blocking candidates ({country})", ncols=100):
                for s1_id, cands in _process_s1_batch(batch):
                    results[s1_id] = cands
            return results

        with ProcessPoolExecutor(
            max_workers=num_workers,
            initializer=_init_worker,
            initargs=(blocker, fallback_blocker, effective_country, self.top_k)
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