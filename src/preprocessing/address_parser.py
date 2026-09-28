"""
Address parser for standardizing street components, numbers, and postal codes.
"""

import re
from typing import Set, Tuple
from .text_cleaner import clean_text

# Street abbreviation dictionary
STREET_MAP = {
    "rd": "road",
    "st": "street",
    "ave": "avenue",
    "av": "avenue",
    "blvd": "boulevard",
    "bd": "boulevard",
    "dr": "drive",
    "pkwy": "parkway",
    "ln": "lane",
    "ct": "court",
    "cir": "circle",
    "hwy": "highway",
    "ste": "unit",
    "apt": "unit",
    "fl": "floor",
    "bldg": "building",
    "str": "street",
    "sq": "square",
    "pl": "place",
}

# Regex for postal codes
POSTAL_INDIA = re.compile(r"\b\d{6}\b")
POSTAL_5DIGIT = re.compile(r"\b\d{5}\b")
NUMBER_PATTERN = re.compile(r"\b\d+[a-zA-Z]?\b")


def clean_address(address: str) -> str:
    """
    Standardizes address tokens and street abbreviations.
    """
    cleaned = clean_text(address)
    tokens = cleaned.split()
    normalized_tokens = [STREET_MAP.get(tok, tok) for tok in tokens]
    return " ".join(normalized_tokens)


def extract_postal_code(address: str, country: str = "") -> str:
    """
    Extracts postal code based on country format (6-digit India, 5-digit US/France).
    """
    if not isinstance(address, str) or not address:
        return ""
    if country == "India":
        m = POSTAL_INDIA.search(address)
        return m.group(0) if m else ""
    elif country in ("US", "France"):
        m = POSTAL_5DIGIT.search(address)
        return m.group(0) if m else ""
    else:
        # Generic: check 6-digit first, then 5-digit
        m6 = POSTAL_INDIA.search(address)
        if m6:
            return m6.group(0)
        m5 = POSTAL_5DIGIT.search(address)
        return m5.group(0) if m5 else ""


def extract_numbers_and_postal(address: str, country: str = "") -> Tuple[Set[str], str]:
    """
    Extracts building/street numbers and postal code.
    Normalizes street numbers by stripping leading zeros (e.g. 0017560 -> 17560).
    Excludes the postal code from the street numbers set.
    """
    if not isinstance(address, str) or not address:
        return set(), ""
    
    postal = extract_postal_code(address, country)
    raw_numbers = NUMBER_PATTERN.findall(address.lower())
    
    # Normalize numbers: include both stripped and raw forms
    all_numbers = set()
    for n in raw_numbers:
        norm = n.lstrip("0") or "0"
        all_numbers.add(norm)
        all_numbers.add(n)
        
    # Exclude postal code from building numbers
    if postal:
        norm_postal = postal.lstrip("0") or "0"
        all_numbers.discard(postal)
        all_numbers.discard(norm_postal)
        
    return all_numbers, postal
