"""
Text cleaner and legal entity suffix normalizer for US, India, and France.
Uses anyascii for robust transliteration of Indic/non-Latin scripts.
Suffixes are anchored to end-of-string to avoid destroying core business names.
"""

import re
import unicodedata
from typing import Set
from anyascii import anyascii

# Regex patterns for common legal suffixes - ANCHORED TO END OF STRING
# Patterns handle both dotted and space-separated forms (e.g., "l.l.c." and "l l c")
US_SUFFIXES = re.compile(
    r"\b(inc|incorporated|corp|corporation|llc|l l c|ltd|limited|lp|llp|pllc|co)\s*$",
    re.IGNORECASE
)

INDIA_SUFFIXES = re.compile(
    r"\b(pvt|private|ltd|limited|llp)\s*$",
    re.IGNORECASE
)

FRANCE_SUFFIXES = re.compile(
    r"\b(sarl|s a r l|sas|s a s|sasu|sa|s a|sci|eurl|snc|ets|etablissements)\s*$",
    re.IGNORECASE
)

ALL_SUFFIXES = re.compile(
    r"\b(inc|incorporated|corp|corporation|llc|l l c|ltd|limited|lp|llp|pllc|co|"
    r"pvt|private|"
    r"sarl|s a r l|sas|s a s|sasu|sa|s a|sci|eurl|snc|ets|etablissements)\s*$",
    re.IGNORECASE
)

# Common business abbreviation dictionary
ABBREVIATIONS = {
    "intl": "international",
    "mfg": "manufacturing",
    "tech": "technology",
    "svcs": "services",
    "svc": "service",
    "mgmt": "management",
    "dept": "department",
    "natl": "national",
    "assn": "association",
    "assoc": "associates",
    "grp": "group",
    "hldgs": "holdings",
    "engg": "engineering",
    "engr": "engineering",
    "govt": "government",
    "indl": "industrial",
    "sys": "systems",
    "soln": "solutions",
    "solns": "solutions",
    "telecom": "telecommunications",
    "pharma": "pharmaceuticals",
    "hosp": "hospital",
    "univ": "university",
    "edu": "education",
    "fin": "financial",
    "mktg": "marketing",
    "acct": "accounting",
    "ins": "insurance",
    "med": "medical",
    "infra": "infrastructure",
    "const": "construction",
    "prop": "properties",
    "dev": "development",
}

DOMAIN_SUFFIX_RE = re.compile(r"\.(com|org|net|co|in|io|biz|us|fr|edu|gov)\b", re.IGNORECASE)
PUNCT_REGEX = re.compile(r"[^\w\s]")
WHITESPACE_REGEX = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """
    Normalizes unicode (transliterating non-Latin scripts to ASCII via anyascii),
    lowercases, replaces & with and, strips domain suffixes, removes punctuation.
    """
    if not text or not isinstance(text, str):
        return ""

    # Normalize unicode and transliterate non-Latin scripts to ASCII
    # anyascii converts Hindi/Bengali/Odia/Telugu/Gujarati to phonetic Latin
    text = anyascii(unicodedata.normalize("NFKC", text))
    text = text.lower()

    # Strip domain/URL suffixes (e.g., primenational.com -> primenational)
    text = DOMAIN_SUFFIX_RE.sub("", text)

    # Strip social handles
    text = text.lstrip("#@")

    # Replace symbols
    text = text.replace("&", " and ")
    text = text.replace("@", " at ")
    text = text.replace("/", " ")
    text = text.replace("-", " ")

    # Remove punctuation
    text = PUNCT_REGEX.sub(" ", text)
    # Collapse whitespace
    text = WHITESPACE_REGEX.sub(" ", text).strip()

    # Expand common abbreviations
    tokens = text.split()
    expanded = [ABBREVIATIONS.get(t, t) for t in tokens]
    text = " ".join(expanded)

    return text


def strip_legal_suffixes(name: str, country: str = "") -> str:
    """
    Removes corporate entity designators (e.g. Inc, LLC, Pvt Ltd, SARL)
    from end of name to yield the core business brand name.
    Applies iteratively to handle stacked suffixes like 'Pvt Ltd'.
    """
    cleaned = clean_text(name)
    if not cleaned:
        return ""

    # Select suffix pattern based on country or apply universal
    if country == "India":
        pattern = INDIA_SUFFIXES
    elif country == "US":
        pattern = US_SUFFIXES
    elif country == "France":
        pattern = FRANCE_SUFFIXES
    else:
        pattern = ALL_SUFFIXES

    # Apply iteratively to handle stacked suffixes like 'Pvt Ltd'
    core = cleaned
    for _ in range(3):  # Max 3 iterations to strip stacked suffixes
        new_core = pattern.sub("", core).strip()
        if new_core == core:
            break
        core = new_core

    core = WHITESPACE_REGEX.sub(" ", core).strip()
    return core if core else cleaned


def get_name_tokens(name: str, min_len: int = 2) -> Set[str]:
    """
    Extracts informative name tokens of minimum length.
    """
    tokens = clean_text(name).split()
    return {t for t in tokens if len(t) >= min_len and not t.isdigit()}
