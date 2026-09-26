"""
Pairwise similarity metrics using rapidfuzz and heuristic address matches.
High-speed precomputed token/trigram architecture for 70,000+ pairs/sec.
22 features total (was 18): added partial_ratio, WRatio, token_count_diff, addr_lev_ratio.
"""

from typing import Dict, Any, Set, Tuple
from rapidfuzz import fuzz, distance


def _char_trigrams(text: str) -> Set[str]:
    """Generate character trigrams from text."""
    if len(text) < 3:
        return {text} if text else set()
    return {text[i:i+3] for i in range(len(text) - 2)}


def compute_pair_features_fast(
    name1: str,
    core1: str,
    addr1: str,
    numbers1: Set[str],
    postal1: str,
    toks1: Set[str],
    tri1: Set[str],
    addr_toks1: Set[str],
    len1: int,
    name2: str,
    core2: str,
    addr2: str,
    numbers2: Set[str],
    postal2: str,
    toks2: Set[str],
    tri2: Set[str],
    addr_toks2: Set[str],
    len2: int,
) -> Tuple[float, ...]:
    """
    High-speed pairwise feature computation using precomputed tokens and trigrams.
    Returns 22 features.
    """
    # Name similarities
    jw_sim = distance.JaroWinkler.similarity(core1, core2)
    token_sort = fuzz.token_sort_ratio(name1, name2) / 100.0
    token_set = fuzz.token_set_ratio(name1, name2) / 100.0
    lev_ratio = fuzz.ratio(core1, core2) / 100.0
    exact_match = 1.0 if core1 == core2 and len1 > 0 else 0.0

    # NEW: partial_ratio and WRatio for substring/reordering handling
    partial_ratio = fuzz.partial_ratio(name1, name2) / 100.0
    wratio = fuzz.WRatio(name1, name2) / 100.0

    # Token exact set match
    token_exact = 1.0 if toks1 == toks2 and len(toks1) > 0 else 0.0

    # Token Jaccard on names
    name_jaccard = len(toks1 & toks2) / max(len(toks1 | toks2), 1)

    # Character trigram Jaccard (precomputed)
    trigram_jaccard = len(tri1 & tri2) / max(len(tri1 | tri2), 1)

    # Name length ratio (precomputed lengths)
    max_len = max(len1, len2)
    name_len_ratio = min(len1, len2) / max_len if max_len > 0 else 0.0

    # Token containment
    if toks1 and toks2:
        shorter, longer = (toks1, toks2) if len(toks1) <= len(toks2) else (toks2, toks1)
        token_containment = len(shorter & longer) / len(shorter) if shorter else 0.0
    else:
        token_containment = 0.0

    # NEW: Token count difference (absolute)
    token_count_diff = float(abs(len(toks1) - len(toks2)))

    # Address similarities (handle missing addresses explicitly)
    addr_token_sort = fuzz.token_sort_ratio(addr1, addr2) / 100.0 if addr1 and addr2 else 0.0
    addr_token_set = fuzz.token_set_ratio(addr1, addr2) / 100.0 if addr1 and addr2 else 0.0

    # NEW: Address Levenshtein ratio
    addr_lev_ratio = fuzz.ratio(addr1, addr2) / 100.0 if addr1 and addr2 else 0.0

    # Address tokens Jaccard (precomputed address tokens)
    addr_jaccard = len(addr_toks1 & addr_toks2) / max(len(addr_toks1 | addr_toks2), 1) if (addr_toks1 or addr_toks2) else 0.0

    # Missing address indicator
    if not addr1 and not addr2:
        addr_missing = 1.0
    elif not addr1 or not addr2:
        addr_missing = 0.5
    else:
        addr_missing = 0.0

    # Postal code match
    if postal1 and postal2:
        postal_match = 1.0 if postal1 == postal2 else -1.0
    else:
        postal_match = 0.0

    # Building number match
    if numbers1 and numbers2:
        num_overlap = len(numbers1 & numbers2)
        num_union = len(numbers1 | numbers2)
        num_match = num_overlap / num_union if num_union > 0 else 0.0
    elif not numbers1 and not numbers2:
        num_match = 0.5  # neutral
    else:
        num_match = 0.0

    # Interaction terms
    if addr_missing == 0.0:
        combined_score = token_sort * addr_token_sort
        name_addr_geom = (token_sort * max(addr_token_sort, 0.01)) ** 0.5
    else:
        combined_score = token_sort * (0.85 if token_sort > 0.85 else 0.5)
        name_addr_geom = token_sort

    return (
        jw_sim,
        token_sort,
        token_set,
        lev_ratio,
        exact_match,
        partial_ratio,
        wratio,
        token_exact,
        name_jaccard,
        trigram_jaccard,
        name_len_ratio,
        token_containment,
        token_count_diff,
        addr_token_sort,
        addr_token_set,
        addr_lev_ratio,
        addr_jaccard,
        addr_missing,
        postal_match,
        num_match,
        combined_score,
        name_addr_geom,
    )


def compute_pair_features_tuple(
    name1: str,
    core1: str,
    addr1: str,
    numbers1: Set[str],
    postal1: str,
    name2: str,
    core2: str,
    addr2: str,
    numbers2: Set[str],
    postal2: str,
) -> Tuple[float, ...]:
    """
    Standard tuple wrapper that extracts tokens on the fly when not precomputed.
    """
    toks1 = set(core1.split())
    tri1 = _char_trigrams(core1)
    addr_toks1 = set(addr1.split()) if addr1 else set()
    len1 = len(core1)

    toks2 = set(core2.split())
    tri2 = _char_trigrams(core2)
    addr_toks2 = set(addr2.split()) if addr2 else set()
    len2 = len(core2)

    return compute_pair_features_fast(
        name1, core1, addr1, numbers1, postal1, toks1, tri1, addr_toks1, len1,
        name2, core2, addr2, numbers2, postal2, toks2, tri2, addr_toks2, len2
    )


def compute_pair_features(
    name1: str,
    core1: str,
    addr1: str,
    numbers1: Set[str],
    postal1: str,
    name2: str,
    core2: str,
    addr2: str,
    numbers2: Set[str],
    postal2: str,
) -> Dict[str, float]:
    """
    Dict wrapper around compute_pair_features_tuple.
    """
    from ..models.train_lgbm import FEATURE_COLS
    t = compute_pair_features_tuple(
        name1, core1, addr1, numbers1, postal1,
        name2, core2, addr2, numbers2, postal2
    )
    return dict(zip(FEATURE_COLS, t))
