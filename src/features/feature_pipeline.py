"""
Feature pipeline extracting features across candidate pairs using high-speed pre-allocated float32 arrays
and precomputed token/trigram lookups.
Uses single-threaded extraction with precomputed lookups — faster than multiprocessing
because the IPC/serialization overhead exceeds the computation time for pruned datasets.
"""

import os
from typing import Dict, List, Tuple, Any
import numpy as np
import pandas as pd
from tqdm import tqdm

from ..preprocessing.text_cleaner import strip_legal_suffixes, clean_text
from ..preprocessing.address_parser import clean_address, extract_numbers_and_postal
from ..models.train_lgbm import FEATURE_COLS
from .similarity_metrics import compute_pair_features_fast, _char_trigrams

NUM_FEATURES = len(FEATURE_COLS)


def _precompute_entity_lookup(df: pd.DataFrame, country: str, desc: str) -> Dict[str, tuple]:
    """
    Precomputes cleaned text, tokens, trigrams, address tokens for each entity.
    Returns dict mapping entity_id -> (name, core, addr, numbers, postal, toks, tri, addr_toks, core_len)
    """
    lookup = {}
    for row in tqdm(df.itertuples(index=False), total=len(df), desc=desc, unit="rec", ncols=100):
        name = clean_text(getattr(row, "business_name", ""))
        core = strip_legal_suffixes(name, country)
        addr = clean_address(getattr(row, "business_address", ""))
        numbers, postal = extract_numbers_and_postal(addr, country)
        toks = set(core.split())
        tri = _char_trigrams(core)
        addr_toks = set(addr.split()) if addr else set()
        lookup[getattr(row, "entity_id")] = (
            name, core, addr, numbers, postal, toks, tri, addr_toks, len(core)
        )
    return lookup


class FeaturePipeline:
    """
    Extracts tabular features for (s1_id, candidate_id) pairs.
    Uses precomputed lookups for speed and pre-allocated float32 arrays for memory efficiency.
    """

    @staticmethod
    def extract_features_for_pairs(
        candidate_map: Dict[str, List[str]],
        df_s1: pd.DataFrame,
        df_target: pd.DataFrame,
        country: str = ""
    ) -> pd.DataFrame:
        """
        Extracts features for all candidate pairs.
        """
        needed_target_ids = {cid for cids in candidate_map.values() for cid in cids}
        if not needed_target_ids:
            return pd.DataFrame()

        total_pairs = sum(len(cands) for cands in candidate_map.values())
        print(f"      Total pairs to process: {total_pairs:,}")

        # 1. Precompute S1 lookups with progress bar
        s1_lookup = _precompute_entity_lookup(
            df_s1, country,
            desc=f"      [c1] Precomputing S1 ({country})"
        )

        # 2. Precompute needed target lookups with progress bar
        df_target_needed = df_target[df_target["entity_id"].isin(needed_target_ids)]
        target_lookup = _precompute_entity_lookup(
            df_target_needed, country,
            desc=f"      [c2] Precomputing targets ({country})"
        )
        del df_target_needed

        # 3. Extract features into pre-allocated float32 matrix
        mat = np.empty((total_pairs, NUM_FEATURES), dtype=np.float32)
        s1_ids = []
        cand_ids = []
        idx = 0

        s1_get = s1_lookup.get
        tgt_get = target_lookup.get

        items = list(candidate_map.items())
        for s1_id, cand_list in tqdm(items, desc=f"      [c3] Computing features ({country})", unit="S1", ncols=100):
            s1 = s1_get(s1_id)
            if s1 is None:
                continue

            for cand_id in cand_list:
                c2 = tgt_get(cand_id)
                if c2 is None:
                    continue

                mat[idx] = compute_pair_features_fast(
                    s1[0], s1[1], s1[2], s1[3], s1[4], s1[5], s1[6], s1[7], s1[8],
                    c2[0], c2[1], c2[2], c2[3], c2[4], c2[5], c2[6], c2[7], c2[8]
                )
                s1_ids.append(s1_id)
                cand_ids.append(cand_id)
                idx += 1

        # Free lookups
        del s1_lookup, target_lookup

        if idx == 0:
            return pd.DataFrame()

        if idx < total_pairs:
            mat = mat[:idx]

        df = pd.DataFrame(mat, columns=FEATURE_COLS)
        df["source1_entity_id"] = s1_ids
        df["candidate_entity_id"] = cand_ids
        return df
