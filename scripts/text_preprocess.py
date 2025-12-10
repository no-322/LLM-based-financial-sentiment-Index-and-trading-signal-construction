"""
Text preprocessing utilities for news headlines.

Implements:
- lowercase
- remove punctuation, URLs, tickers like $AAPL or AAPL.N
- strip boilerplate (“breaking”, “update”, “exclusive”, etc.)
- collapse whitespace

Usage:
from src.cleaning.text_preprocess import normalize_headline
h = normalize_headline("BREAKING: $AAPL reports earnings — read more at https://example.com")
"""

import re
import string
from typing import List

# Boilerplate tokens to strip if present at start or end of headlines
_BOILERPLATE = [
    r'\bbreaking\b', r'\bupdate\b', r'\bexclusive\b', r'\bjust in\b',
    r'\bvia\b', r'\bnew\b', r'\balert\b', r'\bnews\b'
]

# compiled regexes
_URL_RE = re.compile(r'https?://\S+|www\.\S+')
_TICKER_DOLLAR_RE = re.compile(r'\$[A-Za-z0-9]{1,6}\b')
_TICKER_DOTN_RE = re.compile(r'\b[A-Za-z0-9]{1,6}\.N\b')  # e.g., AAPL.N
_SYMBOLIC_RE = re.compile(r'\b[A-Z]{2,5}\b')  # conservative uppercase ticker removal, used optionally
_WHITESPACE_RE = re.compile(r'\s+')
_PUNCT_RE = re.compile('[%s]' % re.escape(string.punctuation))

def _strip_boilerplate(text: str) -> str:
    """
    Remove common boilerplate words often prepended/appended to headlines.
    This removes whole words matching patterns in _BOILERPLATE.
    """
    t = text
    # remove boilerplate words anywhere (but safe)
    for pat in _BOILERPLATE:
        t = re.sub(pat, ' ', t, flags=re.IGNORECASE)
    return t

def normalize_headline(text: str, aggressive_uppercase_ticker_removal: bool = False) -> str:
    """
    Normalize a headline:
    - lowercase
    - remove URLs
    - remove $TICKER and TICKER.N patterns
    - optionally remove ALL-UPPERCASE short tokens that look like tickers
    - remove punctuation (keeps whitespace)
    - collapse whitespace
    - strip leading/trailing whitespace

    Args:
        text: original headline string
        aggressive_uppercase_ticker_removal: if True, remove OTHER uppercase tokens (like "AAPL")
            Use with care: in some headlines uppercase words are meaningful. Default False.

    Returns:
        normalized text
    """
    if not isinstance(text, str):
        return ''

    t = text.strip()

    # Lowercase
    t = t.lower()

    # Remove URLs
    t = _URL_RE.sub(' ', t)

    # Remove $TICKER like $AAPL
    t = _TICKER_DOLLAR_RE.sub(' ', t)

    # Remove TICKER.N and similar (case-insensitive)
    t = _TICKER_DOTN_RE.sub(' ', t)

    # Optionally remove short all-uppercase tokens (user can enable)
    if aggressive_uppercase_ticker_removal:
        t = _SYMBOLIC_RE.sub(' ', t)

    # Strip boilerplate words (we operate in lowercase)
    t = _strip_boilerplate(t)

    # Remove punctuation
    t = _PUNCT_RE.sub(' ', t)

    # Collapse whitespace
    t = _WHITESPACE_RE.sub(' ', t)

    return t.strip()

def normalize_dataframe_headlines(df, headline_col: str = 'headline', inplace: bool = False):
    """
    Apply normalize_headline to a DataFrame column.
    Returns a DataFrame (inplace if requested) with a new column 'headline_norm'.
    """
    import pandas as pd
    if not inplace:
        df = df.copy()
    df['headline_norm'] = df[headline_col].fillna('').astype(str).map(normalize_headline)
    return df
