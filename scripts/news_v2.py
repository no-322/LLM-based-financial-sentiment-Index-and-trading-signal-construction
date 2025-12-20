#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 10 04:04:02 2025

@author: atharva
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Massive Stocks News downloader.

- Uses Massive /v2/reference/news endpoint.
- Inputs: list of tickers, from/to dates.
- Outputs two CSVs:
  1) massive_news_basic.csv
     columns: news_time, ticker, description, source
  2) massive_news_with_sentiment.csv
     columns: ticker, news_time, description, title, sentiment, sentiment_reasoning
"""

import time

import pandas as pd
import requests
from pathlib import Path
from typing import List, Dict, Any, Set, Tuple

# Massive base URL for ticker news
BASE_URL = "https://api.massive.com/v2/reference/news"
BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_DIR = BASE_DIR / "data"
# Default API key used when script is imported (pipeline fallback); override via env MASSIVE_API_KEY
API_KEY = "IIQfprqQI6vanzkvTyZKDvtPGCv6bsNI"


def fetch_news_for_ticker(
    ticker: str,
    from_date: str,
    to_date: str,
    api_key: str,
    limit_per_page: int = 1000,
) -> List[Dict[str, Any]]:
    """
    Fetch Massive news for a single ticker between two dates.

    Parameters
    ----------
    ticker : str
        Stock symbol, e.g. "AAPL".
    from_date : str
        Start date in 'YYYY-MM-DD' format.
    to_date : str
        End date in 'YYYY-MM-DD' format.
    api_key : str
        Massive API key.
    limit_per_page : int, optional
        Number of results per page (max 1000).

    Returns
    -------
    list of dict
        Raw JSON 'results' objects for the ticker.
    """
    ticker = ticker.upper()

    # Use filter modifiers .gte / .lte on published_utc
    params = {
        "ticker": ticker,
        "published_utc.gte": from_date,
        "published_utc.lte": to_date,
        "order": "asc",
        "sort": "published_utc",
        "limit": limit_per_page,
        "apiKey": api_key,  # auth for first call
    }

    url = BASE_URL
    all_results: List[Dict[str, Any]] = []

    while True:
        # ---- API CALL ----
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        results = data.get("results", []) or []
        all_results.extend(results)

        next_url = data.get("next_url")
        if not next_url:
            # no more pages
            break

        # Respect free-tier rate limit (5 requests/min) → 1 call/min is very safe
        print("  Page complete, waiting 20s before next page...")
        time.sleep(20)

        # For the next page, Massive returns a full URL with cursor.
        # Only `apiKey` will be respected when cursor is present.
        url = next_url
        params = {"apiKey": api_key}

    return all_results


def parse_articles_to_rows(
    articles: List[Dict[str, Any]],
    valid_tickers: Set[str],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Convert a list of Massive news article JSON objects into two DataFrames.

    STRICT version:
    - Only keeps articles where Massive explicitly tagged at least one ticker.
    - Filters tickers to those in `valid_tickers`.
    """
    basic_rows: List[Dict[str, Any]] = []
    sentiment_rows: List[Dict[str, Any]] = []

    for article in articles:
        published_utc = article.get("published_utc")
        description = article.get("description")
        article_url = article.get("article_url")
        title = article.get("title")

        # Article-level tickers (STRICT: require at least one)
        article_tickers = article.get("tickers") or []
        article_tickers = [t.upper() for t in article_tickers if t]

        # If Massive did not tag any tickers, skip this article completely
        if not article_tickers:
            continue

        # ---- First CSV rows ----
        for tk in article_tickers:
            if valid_tickers and tk not in valid_tickers:
                continue
            basic_rows.append(
                {
                    "news_time": published_utc,
                    "ticker": tk,
                    "description": description,
                    "source": article_url,
                }
            )

        # ---- Second CSV rows (sentiment) ----
        insights = article.get("insights") or []
        for insight in insights:
            ticker_insight = insight.get("ticker")
            if ticker_insight:
                ticker_insight = ticker_insight.upper()

            # Filter to requested tickers if supplied
            if ticker_insight and valid_tickers and ticker_insight not in valid_tickers:
                continue

            sentiment_rows.append(
                {
                    "ticker": ticker_insight,
                    "news_time": published_utc,
                    "description": description,
                    "title": title,
                    "sentiment": insight.get("sentiment"),
                    "sentiment_reasoning": insight.get("sentiment_reasoning"),
                }
            )

    df_basic = pd.DataFrame(basic_rows).drop_duplicates()
    df_sentiment = pd.DataFrame(sentiment_rows).drop_duplicates()

    return df_basic, df_sentiment


def build_news_dataframes(
    tickers: List[str],
    from_date: str,
    to_date: str,
    api_key: str,
    limit_per_page: int = 1000,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Download Massive news for a list of tickers and convert to two DataFrames.

    Parameters
    ----------
    tickers : list of str
        Ticker symbols.
    from_date, to_date : str
        'YYYY-MM-DD'.
    api_key : str
        Massive API key.

    Returns
    -------
    df_basic, df_sentiment : DataFrame, DataFrame
    """
    tickers_clean = [t.strip().upper() for t in tickers if t.strip()]
    valid_tickers: Set[str] = set(tickers_clean)
    all_articles: List[Dict[str, Any]] = []

    for i, tk in enumerate(tickers_clean):
        print(f"Fetching news for {tk} from {from_date} to {to_date}...")
        try:
            articles = fetch_news_for_ticker(
                ticker=tk,
                from_date=from_date,
                to_date=to_date,
                api_key=api_key,
                limit_per_page=limit_per_page,
            )
        except requests.RequestException as exc:
            print(f"  ERROR while fetching {tk}: {exc}")
            continue

        print(f"  Retrieved {len(articles)} raw articles for {tk}.")
        all_articles.extend(articles)

        # Extra safety: wait 60s between tickers as well
        if i < len(tickers_clean) - 1:
            print("  Waiting 20s before next ticker to respect rate limits...")
            time.sleep(20)

    df_basic, df_sentiment = parse_articles_to_rows(all_articles, valid_tickers)
    return df_basic, df_sentiment


def save_news_to_csv(
    tickers: List[str],
    from_date: str,
    to_date: str,
    api_key: str,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    limit_per_page: int = 1000,
) -> Tuple[Path, Path]:
    """
    High-level helper: fetch news and save two CSVs.

    Parameters
    ----------
    tickers : list of str
        Ticker symbols.
    from_date, to_date : str
        'YYYY-MM-DD'.
    api_key : str
        Massive API key.
    output_dir : str
        Directory where CSVs should be written.
    limit_per_page : int
        Page size for API calls (max 1000).

    Returns
    -------
    basic_path, sentiment_path : Path
        Paths to the written CSV files.
    """
    df_basic, df_sentiment = build_news_dataframes(
        tickers=tickers,
        from_date=from_date,
        to_date=to_date,
        api_key=api_key,
        limit_per_page=limit_per_page,
    )
    
    sort_cols = ["ticker", "news_time"]
    if not df_basic.empty:
        df_basic = df_basic.sort_values(by=sort_cols)
    if not df_sentiment.empty:
        df_sentiment = df_sentiment.sort_values(by=sort_cols)
        # Drop raw sentiment label; downstream pipeline will compute its own
        if "sentiment" in df_sentiment.columns:
            df_sentiment = df_sentiment.drop(columns=["sentiment"])
    
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    basic_path = out_dir / "massive_news_basic.csv"
    sentiment_path = out_dir / "massive_news_with_sentiment.csv"

    df_basic.to_csv(basic_path, index=False)
    df_sentiment.to_csv(sentiment_path, index=False)

    print(f"Saved {len(df_basic)} rows to {basic_path}")
    print(f"Saved {len(df_sentiment)} rows to {sentiment_path}")

    return basic_path, sentiment_path


if __name__ == "__main__":
    # ====== USER INPUTS ======
    TICKERS = ["AAPL","PLTR","CRWD","LC","JPM","SCHW"]  # add more tickers if you like



    #OUTPUT_DIR = "output"     # e.g. "./data" if you want a subfolder
    # =========================
    FROM_DATE = "2024-01-01"  # inclusive
    TO_DATE = "2025-12-01"    # inclusive

    # Base dir = project root (two levels up, same style as your other script)
    OUTPUT_DIR = DEFAULT_OUTPUT_DIR
# =========================


    save_news_to_csv(
        tickers=TICKERS,
        from_date=FROM_DATE,
        to_date=TO_DATE,
        api_key=API_KEY,
        output_dir=OUTPUT_DIR,
        limit_per_page=1000,
    )
