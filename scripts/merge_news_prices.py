"""
Merge hourly price data with Alpha Vantage news.

Rules:
- News timestamps are aligned to the first available price bar at/after the
  news time using merge_asof (forward). This implicitly handles after-hours
  news rolling into the next session's first bar.
- Price timestamps are not modified; we create a merge key on news.

Outputs:
- data/price_news_hourly.csv : price bars with aggregated news metadata aligned
  on ticker + timestamp.
"""

from pathlib import Path
from typing import Optional

import pandas as pd
from pandas.tseries.offsets import BDay

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

NEWS_PATH = DATA_DIR / "alpha_news_all_tickers.csv"
PRICE_PATH = DATA_DIR / "all_data.csv"
OUTPUT_PATH = DATA_DIR / "price_news_hourly.csv"

def load_news(path: Path = NEWS_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    if df.empty:
        return df

    # Normalize ticker list
    df["ticker"] = (
        df.get("ticker", pd.NA)
        .fillna("")
        .astype(str)
        .replace("", pd.NA)
    )
    if df["ticker"].isna().all() and "ticker_sentiment" in df.columns:
        df["ticker"] = (
            df["ticker_sentiment"]
            .fillna("")
            .astype(str)
            .apply(lambda x: [t.strip() for t in x.split(",") if t.strip()])
        )
        df = df.explode("ticker")

    # Parse published time; handle multiple possible formats/columns
    df["published_at"] = pd.to_datetime(
        df.get("time_published", pd.NA), format="%Y%m%dT%H%M%S", errors="coerce"
    )
    df.loc[df["published_at"].isna(), "published_at"] = pd.to_datetime(
        df.get("time_published", pd.NA), errors="coerce"
    )
    if df["published_at"].isna().all() and "date_time" in df.columns:
        df["published_at"] = pd.to_datetime(df["date_time"], errors="coerce")

    # Ensure expected columns exist
    for col in [
        "headline",
        "source",
        "summary",
        "url",
        "overall_sentiment_score",
        "overall_sentiment_label",
    ]:
        if col not in df.columns:
            df[col] = pd.NA

    # Align timezone with price data (UTC)
    if df["published_at"].dt.tz is None:
        df["published_at"] = df["published_at"].dt.tz_localize(
            "UTC", nonexistent="NaT", ambiguous="NaT"
        )
    else:
        df["published_at"] = df["published_at"].dt.tz_convert("UTC")

    # Drop rows without a ticker or timestamp
    df = df.dropna(subset=["ticker", "published_at"])
    return df


def aggregate_news(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    agg = df.groupby(["ticker", "timestamp_for_merge"]).agg(
        news_count=("headline", "count"),
        headlines=("headline", lambda s: " | ".join(dict.fromkeys(map(str, s)))),
        sources=("source", lambda s: ";".join(sorted(set(filter(pd.notna, s))))),
        summaries=("summary", lambda s: " || ".join(dict.fromkeys(map(str, s)))),
        urls=("url", lambda s: ";".join(dict.fromkeys(map(str, s)))),
        sentiment_score=("overall_sentiment_score", "mean"),
        sentiment_labels=(
            "overall_sentiment_label",
            lambda s: ";".join(sorted(set(filter(pd.notna, s)))),
        ),
    )
    return agg.reset_index()


def merge_price_and_news(
    price_path: Path = PRICE_PATH,
    news_path: Path = NEWS_PATH,
    output_path: Path = OUTPUT_PATH,
) -> Optional[Path]:
    if not price_path.exists():
        print(f"No price file found at {price_path}; skipping merge.")
        return None
    try:
        prices = pd.read_csv(price_path, parse_dates=["timestamp"])
    except pd.errors.EmptyDataError:
        print(f"Price file {price_path} is empty; skipping merge.")
        return None

    if prices.empty or "timestamp" not in prices.columns:
        print(f"Price data unavailable or missing timestamp in {price_path}; skipping merge.")
        return None
    news_df = load_news(news_path)

    if news_df.empty:
        print("No news to merge; writing prices only.")
        merged = prices.copy()
        merged.to_csv(output_path, index=False)
        return output_path

    # Align each news item to the first price bar at/after its timestamp
    price_times = prices[["timestamp"]].drop_duplicates().sort_values("timestamp")
    news_df = news_df.sort_values("published_at")
    aligned = pd.merge_asof(
        news_df,
        price_times,
        left_on="published_at",
        right_on="timestamp",
        direction="forward",
        allow_exact_matches=True,
    )
    aligned = aligned.rename(columns={"timestamp": "timestamp_for_merge"})
    aligned = aligned.dropna(subset=["timestamp_for_merge"])

    news_agg = aggregate_news(aligned)

    merged = prices.merge(
        news_agg,
        how="left",
        left_on=["ticker", "timestamp"],
        right_on=["ticker", "timestamp_for_merge"],
    )
    merged.drop(columns=["timestamp_for_merge"], inplace=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_path, index=False)
    print(f"Saved merged price/news to {output_path}")
    return output_path


if __name__ == "__main__":
    merge_price_and_news()
