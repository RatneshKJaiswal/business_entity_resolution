"""
High-Recall Inverted Index Blocker for Business Entity Resolution.
Combines:
1. Exact name tokens with continuous IDF scoring (no arbitrary token dropping).
2. Exact normalized names for strong direct matches.
3. 4-character prefix tokens for spelling variations, acronyms, and compound domain names.
4. Postal-code keys for candidates with incomplete name/address details.
5. Address-number keys for records missing a reliable street or postal key.
6. Compound address keys (postal + street number) for address-level entity linking.
7. Locality/Street keys (street token + street number) when postal code is missing.
Achieves >90% candidate recall across both US and India partitions.
"""

import math
from collections import defaultdict
import heapq
from typing import Dict, List, Optional, Set

ADDR_COMMON = {
    "road", "street", "avenue", "boulevard", "drive", "lane", "place", "court",
    "circle", "highway", "way", "floor", "unit", "building", "suite", "apt",
    "east", "west", "north", "south", "nagar", "marg", "gali", "colony", "cross",
    "main", "near", "opp", "opposite", "behind", "dist", "taluk", "phase", "sector"
}

BLOCKING_CHANNELS = frozenset({
    "core_name",
    "name_tokens",
    "prefix",
    "postal",
    "number",
    "compound_address",
})
DEFAULT_BLOCKING_CHANNELS = BLOCKING_CHANNELS - {"prefix"}


class FastInvertedIndexBlocker:
    """Retrieves candidates from name and compound-address index channels."""
    def __init__(self, top_k: int = 75):
        self.top_k = top_k
        self.total_docs: int = 0
        self.id_list: List[str] = []
        
        # Inverted index channels
        self.index_core_name: Dict[str, List[int]] = defaultdict(list)
        self.index_exact: Dict[str, List[int]] = defaultdict(list)
        self.index_prefix: Dict[str, List[int]] = defaultdict(list)
        self.index_address: Dict[str, List[int]] = defaultdict(list)
        self.index_postal: Dict[str, List[int]] = defaultdict(list)
        self.index_number: Dict[str, List[int]] = defaultdict(list)
        self.doc_freq: Dict[str, int] = defaultdict(int)

    def fit(
        self,
        entity_ids: List[str],
        core_names: List[str],
        token_sets: List[Set[str]],
        postal_codes: Optional[List[str]] = None,
        addresses: Optional[List[str]] = None
    ) -> None:
        """
        Builds the 4-channel inverted index over target records.
        """
        self.id_list = entity_ids
        self.total_docs = len(entity_ids)

        from ..preprocessing.address_parser import extract_numbers_and_postal, clean_address

        for idx in range(self.total_docs):
            toks = token_sets[idx]
            if core_names[idx]:
                self.index_core_name[core_names[idx]].append(idx)
            
            # Channel 1: Exact Name Tokens
            for t in toks:
                self.index_exact[t].append(idx)
                self.doc_freq[t] += 1
                
            # Channel 2: 4-Char Prefixes (for tokens >= 5 chars, handles typos and domain compounds)
            for t in toks:
                if len(t) >= 5:
                    p = t[:4] + "$"
                    self.index_prefix[p].append(idx)
                    self.doc_freq[p] += 1

            # Channel 3 & 4: Compound Address Keys
            if addresses:
                addr = addresses[idx]
                if addr:
                    pc = postal_codes[idx] if postal_codes else ""
                    nums, post = extract_numbers_and_postal(addr)
                    eff_post = post or pc

                    if eff_post:
                        self.index_postal[eff_post].append(idx)

                    for num in nums:
                        self.index_number[num].append(idx)
                    
                    # Channel 3: (Postal + Building Number)
                    if eff_post:
                        for num in nums:
                            self.index_address[f"p_{eff_post}_{num}"].append(idx)
                            
                    # Channel 4: (Street Token + Building Number) when postal is missing or supplementary
                    c_addr = clean_address(addr)
                    sws = [w for w in c_addr.split() if len(w) >= 4 and not w.isdigit() and w not in ADDR_COMMON]
                    for num in nums:
                        for sw in sws[:3]:
                            self.index_address[f"a_{sw}_{num}"].append(idx)

    def retrieve_candidates(
        self,
        s1_core_name: str,
        s1_tokens: Set[str],
        s1_postal: str = "",
        s1_address: str = "",
        top_k: Optional[int] = None,
        enabled_channels: Optional[Set[str]] = None
    ) -> List[str]:
        """
        Multi-channel retrieval accumulating continuous IDF scores across all 4 channels.
        """
        if top_k is None:
            top_k = self.top_k
        channels = DEFAULT_BLOCKING_CHANNELS if enabled_channels is None else enabled_channels
        unknown_channels = channels - BLOCKING_CHANNELS
        if unknown_channels:
            raise ValueError(f"Unknown blocking channels: {sorted(unknown_channels)}")
        if not s1_core_name and not s1_tokens and not s1_postal and not s1_address:
            return []

        candidate_scores: Dict[int, float] = defaultdict(float)
        core_name_scores: Dict[int, float] = defaultdict(float)
        address_scores: Dict[int, float] = defaultdict(float)
        postal_scores: Dict[int, float] = defaultdict(float)
        number_scores: Dict[int, float] = defaultdict(float)
        total_docs_plus_one = self.total_docs + 1
        log = math.log

        # Exact normalized core-name candidates get reserved budget so
        # numerous token-overlap distractors cannot displace direct name matches.
        if "core_name" in channels and s1_core_name:
            postings = self.index_core_name.get(s1_core_name)
            if postings and len(postings) <= 5000:
                for cid in postings:
                    core_name_scores[cid] += 4.0

        # ── Channel 1: Exact Name Tokens with Continuous IDF ──
        if "name_tokens" in channels:
            for t in s1_tokens:
                postings = self.index_exact.get(t)
                if postings:
                    df = self.doc_freq[t]
                    idf = log(total_docs_plus_one / (df + 1))

                    # Common tokens carry little discriminative value. Do not take a
                    # prefix of these postings: that makes recall depend on file order.
                    if df <= 30000:
                        for cid in postings:
                            candidate_scores[cid] += idf

        # ── Channel 2: 4-Character Prefixes (Catches typos & domain compounds) ──
        if "prefix" in channels:
            for t in s1_tokens:
                if len(t) >= 5:
                    p = t[:4] + "$"
                    postings = self.index_prefix.get(p)
                    if postings and len(postings) <= 5000:
                        df = self.doc_freq[p]
                        idf = 0.5 * log(total_docs_plus_one / (df + 1))
                        for cid in postings:
                            candidate_scores[cid] += idf

        # ── Channel 3 & 4: Compound Address Keys (Catches rebrands/aliases) ──
        if s1_address:
            from ..preprocessing.address_parser import extract_numbers_and_postal, clean_address
            nums, post = extract_numbers_and_postal(s1_address)
            eff_post = post or s1_postal

            # Postal-only retrieval catches aliases and records with incomplete
            # names while remaining limited to reasonably selective postal areas.
            if "postal" in channels and eff_post:
                postings = self.index_postal.get(eff_post)
                if postings and len(postings) <= 5000:
                    for cid in postings:
                        postal_scores[cid] += 1.5

            # Number-only retrieval covers address matches where street tokens
            # and postal codes differ, with a strict frequency and slot limit.
            if "number" in channels:
                for num in nums:
                    postings = self.index_number.get(num)
                    if postings and len(postings) <= 5000:
                        idf = log(total_docs_plus_one / (len(postings) + 1))
                        for cid in postings:
                            number_scores[cid] += 1.0 + idf

            # Channel 3: Postal + Number (+3.5 boost)
            if "compound_address" in channels and eff_post:
                for num in nums:
                    postings = self.index_address.get(f"p_{eff_post}_{num}")
                    if postings and len(postings) <= 1000:
                        for cid in postings:
                            address_scores[cid] += 3.5

            # Channel 4: Street Token + Number (+2.5 boost)
            if "compound_address" in channels:
                c_addr = clean_address(s1_address)
                sws = list(dict.fromkeys(
                    w for w in c_addr.split()
                    if len(w) >= 4 and not w.isdigit() and w not in ADDR_COMMON
                ))
                for num in nums:
                    for sw in sws[:3]:
                        postings = self.index_address.get(f"a_{sw}_{num}")
                        if postings and len(postings) <= 1000:
                            for cid in postings:
                                address_scores[cid] += 2.5

        if (
            not candidate_scores
            and not core_name_scores
            and not address_scores
            and not postal_scores
            and not number_scores
        ):
            return []

        # Reserve candidate slots for independent retrieval channels so that
        # one noisy or common name token cannot crowd out other strong evidence.
        combined_scores = dict(candidate_scores)
        for cid, score in core_name_scores.items():
            combined_scores[cid] = combined_scores.get(cid, 0.0) + score
        for cid, score in postal_scores.items():
            combined_scores[cid] = combined_scores.get(cid, 0.0) + score
        for cid, score in number_scores.items():
            combined_scores[cid] = combined_scores.get(cid, 0.0) + score
        for cid, score in address_scores.items():
            combined_scores[cid] = combined_scores.get(cid, 0.0) + score

        core_name_budget = min(top_k // 3, len(core_name_scores))
        core_name_indices = heapq.nlargest(
            core_name_budget, core_name_scores, key=core_name_scores.__getitem__
        )
        selected = set(core_name_indices)
        number_budget = min(top_k // 6, len(number_scores))
        number_indices = heapq.nlargest(
            number_budget,
            (cid for cid in number_scores if cid not in selected),
            key=number_scores.__getitem__
        )
        selected.update(number_indices)
        postal_budget = min(top_k // 6, len(postal_scores))
        postal_indices = heapq.nlargest(
            postal_budget,
            (cid for cid in postal_scores if cid not in selected),
            key=postal_scores.__getitem__
        )
        selected.update(postal_indices)
        address_budget = min(top_k // 3, len(address_scores))
        address_indices = heapq.nlargest(
            address_budget,
            (cid for cid in address_scores if cid not in selected),
            key=combined_scores.__getitem__
        )
        selected.update(address_indices)
        remaining_budget = max(0, top_k - len(selected))
        remaining_indices = heapq.nlargest(
            remaining_budget,
            (cid for cid in combined_scores if cid not in selected),
            key=combined_scores.__getitem__
        )
        selected.update(remaining_indices)

        return [
            self.id_list[idx]
            for idx in sorted(selected, key=lambda cid: (-combined_scores[cid], cid))
        ]