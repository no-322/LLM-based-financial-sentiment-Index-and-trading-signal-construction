#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
from pathlib import Path

import pandas as pd
import requests

# Prefer env var; fall back to provided default for quick local runs
DEFAULT_API_KEY = "M88D8H99O716X796"
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"



def get_alpha_vantage_news(tickers, limit=50, sort_by="LATEST", api_key=None):
    """
    Fetches news/headlines for given tickers from Alpha Vantage.

    Parameters
    ----------
    tickers : list of str
        List of stock tickers, e.g. ["AAPL", "MSFT"].
    limit : int
        Max number of news articles to return (Alpha Vantage allows up to 50 per call).
    sort_by : str
        "LATEST" or "EARLIEST" (depends on Alpha Vantage API behavior).

    Returns
    -------
    pandas.DataFrame
        DataFrame with headline, source, time, url, summary, etc.
    """
    key = api_key or os.getenv("ALPHAVANTAGE_API_KEY", DEFAULT_API_KEY)
    if not key:
        raise RuntimeError(
            "Alpha Vantage API key missing. "
            "Set ALPHAVANTAGE_API_KEY env var or pass api_key explicitly."
        )

    # Alpha Vantage "NEWS_SENTIMENT" endpoint
    url = "https://www.alphavantage.co/query"

    # Join tickers as a comma-separated string
    tickers_str = ",".join(tickers)

    params = {
        "function": "NEWS_SENTIMENT",
        "tickers": tickers_str,
        "apikey": key,
        "limit": limit,
        "sort": sort_by,
    }

    response = requests.get(url, params=params, timeout=30)

    if response.status_code != 200:
        raise RuntimeError(f"Error: HTTP {response.status_code} - {response.text}")

    data = response.json()

    if "feed" not in data:
        raise RuntimeError(f"No 'feed' field in response. Full response:\n{data}")

    # Parse into DataFrame
    articles = []
    for item in data["feed"]:
        articles.append({
            "ticker_sentiment": ", ".join([t["ticker"] for t in item.get("ticker_sentiment", [])]),
            "headline": item.get("title"),
            "summary": item.get("summary"),
            "source": item.get("source"),
            "time_published": item.get("time_published"),
            "url": item.get("url"),
            "overall_sentiment_score": item.get("overall_sentiment_score"),
            "overall_sentiment_label": item.get("overall_sentiment_label"),
        })

    df = pd.DataFrame(articles)
    return df


if __name__ == "__main__":
    # Example usage:
    tickers = ["AAPL"]
    df_news = get_alpha_vantage_news(tickers, limit=30)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df_news.to_csv(DATA_DIR / "alpha_news.csv", index=False)

    print(df_news.head())
