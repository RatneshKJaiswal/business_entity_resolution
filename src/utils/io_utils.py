"""
Memory-efficient I/O utilities for reading and writing TSV files.
"""

from pathlib import Path
from typing import Dict, Set, Iterator, List, Optional
import pandas as pd


def load_ground_truth(path: Path) -> Dict[str, Set[str]]:
    """
    Loads train_ground_truth.tsv into a dictionary of {s1_id: set(matched_ids)}.
    """
    gt_map: Dict[str, Set[str]] = {}
    with open(path, "r", encoding="utf-8") as f:
        header = f.readline()  # skip header
        for line in f:
            line = line.rstrip("\r\n")
            if not line:
                continue
            parts = line.split("\t", 1)
            s1_id = parts[0].strip()
            if len(parts) > 1 and parts[1].strip():
                gt_map[s1_id] = {mid.strip() for mid in parts[1].split(",") if mid.strip()}
            else:
                gt_map[s1_id] = set()
    return gt_map


def load_source_dataframe(
    path: Path,
    usecols: Optional[List[str]] = None,
    nrows: Optional[int] = None
) -> pd.DataFrame:
    """
    Reads a source TSV file safely into a pandas DataFrame.
    """
    return pd.read_csv(
        path,
        sep="\t",
        usecols=usecols,
        nrows=nrows,
        dtype={
            "entity_id": str,
            "business_name": str,
            "business_address": str,
            "country": str
        },
        na_values=["", "NA", "null"],
        keep_default_na=False,
        encoding="utf-8"
    )


def stream_source_tsv(
    path: Path,
    chunksize: int = 100_000,
    usecols: Optional[List[str]] = None
) -> Iterator[pd.DataFrame]:
    """
    Streams a large source TSV file in manageable chunks.
    """
    yield from pd.read_csv(
        path,
        sep="\t",
        usecols=usecols,
        chunksize=chunksize,
        dtype={
            "entity_id": str,
            "business_name": str,
            "business_address": str,
            "country": str
        },
        na_values=["", "NA", "null"],
        keep_default_na=False,
        encoding="utf-8"
    )


def write_submission_tsv(
    path: Path,
    header_col1: str,
    header_col2: str,
    mapping: Dict[str, Set[str]],
    required_s1_order: Optional[List[str]] = None
) -> None:
    """
    Writes out a strictly compliant TSV submission file.
    Follows every validator rule:
    - Tab-separated
    - Header: header_col1 \t header_col2
    - Comma-separated ID list without spaces or quotes
    - One row per S1 entity
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = required_s1_order if required_s1_order is not None else list(mapping.keys())

    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(f"{header_col1}\t{header_col2}\n")
        for s1_id in keys:
            m_set = mapping.get(s1_id, set())
            m_str = ",".join(sorted(m_set)) if m_set else ""
            f.write(f"{s1_id}\t{m_str}\n")
