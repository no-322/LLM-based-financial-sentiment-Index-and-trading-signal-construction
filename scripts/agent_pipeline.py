import argparse
from datetime import datetime, timedelta
from pathlib import Path
from typing import Union

import pandas as pd
import yaml
import yfinance as yf
import os
from deduplicate import run_dedup_pipeline
from final_data import (
    bucket_news_inplace,
    convert_all_data_gmt_to_est,
    merge_datasets,
)
from news_v2 import save_news_to_csv

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DEFAULT_CONFIG_PATH = BASE_DIR / "config" / "config.yaml"


def load_config(path: Union[Path, str] = DEFAULT_CONFIG_PATH) -> dict:
    """
    Load the YAML config file and return a dictionary.
    
    Args:
        path (str): Path to the YAML file.

    Returns:
        dict: Configuration dictionary with defaults applied.
    """
    config_path = Path(path)
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg


def parse_args():
    """
    Parse command-line arguments for overriding config values.
    """
    parser = argparse.ArgumentParser(description="Run Agentic AI pipeline")

    parser.add_argument(
        "--tickers",
        nargs="+",
        help="List of tickers to override config (e.g. --tickers AAPL MSFT)",
    )
    parser.add_argument(
        "--granularity",
        type=str,
        choices=["h", "d"],
        help="Override granularity: 'h' (hourly) or 'd' (daily)",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        help="Override model name (e.g. 'gpt-4o-mini')",
    )

    return parser.parse_args()


def apply_overrides(config: dict, args):
    """
    Override config values with provided command-line args.

    Args:
        config (dict): default configuration loaded from YAML
        args (Namespace): parsed arguments from parse_args()

    Returns:
        dict: updated configuration
    """

    if args.tickers is not None:
        config["tickers"] = args.tickers

    if args.granularity is not None:
        config["granularity"] = args.granularity

    if args.model_name is not None:
        config["agent"]["model"] = args.model_name

    return config


def scrape_headlines(tickers, granularity) -> pd.DataFrame:
    """
    Fetch headlines for the provided tickers and persist them.
    
    Args:
        tickers (list): Ticker data to extract.
        granularity (str): Frequency of price data. Provided for interface consistency.
    
    Returns:
        pandas.Dataframe: Columns include timestamp, date_time, ticker, headline, source,
            summary, url, and optional sentiment metadata.
    """
    # Use existing extracted news if available; skip refetching
    cached_path = DATA_DIR / "alpha_news_all_tickers.csv"
    if cached_path.exists():
        print(f"Loading cached news from {cached_path}")
        cached_df = pd.read_csv(cached_path)
        return cached_df
    # Prefer env var, but fall back to the default defined in news_v2.py so pipeline runs without manual export
    api_key = os.getenv("MASSIVE_API_KEY") or getattr(__import__("news_v2"), "API_KEY", None)
    if not api_key:
        print("Environment variable MASSIVE_API_KEY not set; cannot fetch Massive news.")
        return pd.DataFrame(
            columns=[
                "timestamp",
                "date_time",
                "ticker",
                "headline",
                "source",
                "summary",
                "url",
                "overall_sentiment_score",
                "overall_sentiment_label",
            ]
        )

    # Set date range aligned with price fetching logic
    now = datetime.now().date()
    if granularity == "h":
        from_date = datetime(2024, 1, 1).date()
    elif granularity == "d":
        from_date = now - timedelta(days=365 * 5)
    else:
        from_date = now - timedelta(days=365)

    print(f"Fetching Massive news from {from_date} to {now}…")
    basic_path, sent_path = save_news_to_csv(
        tickers=tickers,
        from_date=from_date.strftime("%Y-%m-%d"),
        to_date=now.strftime("%Y-%m-%d"),
        api_key=api_key,
        output_dir=DATA_DIR,
        limit_per_page=1000,
    )

    # Load Massive outputs
    df_basic = pd.read_csv(basic_path) if Path(basic_path).exists() else pd.DataFrame()
    df_sent = pd.read_csv(sent_path) if Path(sent_path).exists() else pd.DataFrame()

    # Prefer sentiment file (has titles), fall back to basic
    if not df_sent.empty:
        df = df_sent.rename(
            columns={
                "news_time": "date_time",
                "title": "headline",
                "description": "summary",
                "sentiment": "overall_sentiment_label",
            }
        )
        df["url"] = pd.NA
        df["source"] = pd.NA
        if not df_basic.empty:
            basic_clean = df_basic.rename(
                columns={
                    "news_time": "date_time",
                    "description": "basic_description",
                    "source": "url",
                }
            )
            df = df.merge(
                basic_clean[["ticker", "date_time", "url"]],
                on=["ticker", "date_time"],
                how="left",
                suffixes=("", "_basic"),
            )
            df["url"] = df["url"].fillna(df.get("url_basic"))
            if "url_basic" in df.columns:
                df.drop(columns=["url_basic"], inplace=True)
        df["source"] = df["source"].fillna(df.get("url"))
        # align text fields so downstream dedup/sentiment use Massive reasoning
        df["headline"] = df.get("sentiment_reasoning", df.get("headline"))
        df["summary"] = df.get("sentiment_reasoning", df.get("summary"))
    else:
        df = df_basic.rename(
            columns={
                "news_time": "date_time",
                "description": "headline",
                "source": "url",
            }
        )
        df["summary"] = df.get("headline")
        df["source"] = df.get("url")

    if df.empty:
        print("No headlines returned from Massive.")
        df = pd.DataFrame(
            columns=[
                "timestamp",
                "date_time",
                "ticker",
                "headline",
                "source",
                "summary",
                "url",
                "overall_sentiment_score",
            ]
        )
    else:
        # Normalize tickers and timestamps
        df["ticker"] = df["ticker"].astype(str).str.upper()
        df["date_time"] = pd.to_datetime(df["date_time"], errors="coerce")
        df = df.dropna(subset=["ticker", "date_time"])
        df["timestamp"] = df["date_time"]
        if "overall_sentiment_score" not in df.columns:
            df["overall_sentiment_score"] = pd.NA
        df = df.sort_values("timestamp", ascending=False)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    output_path = DATA_DIR / "alpha_news_all_tickers.csv"
    df.to_csv(output_path, index=False)
    print(f"Saved {len(df)} headlines to {output_path}")
    return df


def deduplicate_headlines(headlines_df: pd.DataFrame) -> pd.DataFrame:
    """
    Run semantic deduplication on fetched headlines and persist results.
    """
    if headlines_df.empty:
        print("No headlines to deduplicate.")
        return pd.DataFrame()

    dedup_df = headlines_df.copy()
    required_cols = ["timestamp", "ticker", "headline", "source", "summary"]
    for col in required_cols:
        if col not in dedup_df.columns:
            dedup_df[col] = pd.NA

    dedup_df = dedup_df[required_cols].rename(columns={"timestamp": "date_time"})
    dedup_df["date_time"] = pd.to_datetime(dedup_df["date_time"], errors="coerce")
    dedup_df = dedup_df.dropna(subset=["date_time"])
    if dedup_df.empty:
        print("No valid timestamps available for deduplication.")
        return pd.DataFrame()
    dedup_df["date_time"] = dedup_df["date_time"].dt.strftime("%Y-%m-%d %H:%M:%S")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    raw_path = DATA_DIR / "news_data.csv"
    dedup_df.to_csv(raw_path, index=False)

    try:
        _, clustered_df = run_dedup_pipeline(
            input_path=raw_path,
            output_path=DATA_DIR / "news_deduped.csv",
        )
        print(f"Deduplicated headlines saved to {DATA_DIR / 'news_deduped.csv'}")
        return clustered_df
    except Exception as exc:
        print(f"Deduplication failed: {exc}")
        return pd.DataFrame()


def clean_headlines(df) -> pd.DataFrame:
    """
    Return cleaned headlines for sentiment analysis
    
    Args:
        pandas.Dataframe: A dataframe with the following columns:
            - date (datetime64): Trading Date
            - ticker (str): Ticker pertaining to the headline
            - headline (str): headline extracted
            - source (str): data source
    
    Returns:
        pandas.Dataframe: A dataframe with the following columns:
            - timestamp_first_seen (datetime64): First observed timestamp for the headline
            - timestamp_last_seen (datetime64: Last observed timestamp for the headline
            - ticker (str): Ticker pertaining to the headline
            - clean_headline (str): cleaned up headline extracted. Shows first occurance
            - source_list (list[str]): list of sources
            - repeat_count (int): number of times the headline was observed
    """
    pass


def run_sentiment(df):
    pass


def build_sentiment_index(df):
    pass


def fetch_price_data(tickers, granularity):
    """
    Return historical price data for a set of tickers

    Args:
        tickers (list[str]): A list of ticker symbols to download.
        granularity (str): Frequency of price data. Supported values:
            - 'h': hourly bars
            - 'd': daily bars

    Returns:
        pandas.DataFrame: A dataframe with the following columns:
            - timestamp (datetime64): Timestamp of the price bar.
            - ticker (str): Ticker symbol.
            - close (float): Adjusted close price for the bar.
            - volume (int): volume of shares traded.
            - returns(float): period returns of the stock
    """
    all_data = pd.DataFrame()
    
    # Define date ranges
    now = datetime.now()
    if granularity == "h":
        start_date = datetime(2024, 1, 1)
        interval = "1h"
    elif granularity == "d":
        start_date = now - timedelta(days=365 * 5)
        interval = "1d"
    else:
        raise ValueError("granularity must be 'h' (hourly) or 'd' (daily)")

    for ticker in tickers:
        print(f"Downloading {granularity.upper()} data for {ticker} ({start_date.date()} → {now.date()})")
        df = yf.download(
            ticker,
            start=start_date.strftime("%Y-%m-%d"),
            end=now.strftime("%Y-%m-%d"),
            interval=interval,
            auto_adjust=True,
            progress=False
        )

        if df.empty:
            print(f"No data returned for {ticker}")
            continue

        # Flatten columns if needed
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = ["_".join(filter(None, col)).strip() for col in df.columns.values]

        # Reset index
        df = df.reset_index()
        df.rename(columns={df.columns[0]: "timestamp"}, inplace=True)

        # Detect Close and Volume columns
        close_col = [c for c in df.columns if "Close" in c][0]
        vol_col = [c for c in df.columns if "Vol" in c][0] if any("Vol" in c for c in df.columns) else None

        # Keep only relevant columns
        columns_to_keep = ["timestamp", close_col]
        if vol_col:
            columns_to_keep.append(vol_col)
        df = df[columns_to_keep]

        # Rename columns
        df.rename(columns={close_col: "close"}, inplace=True)
        if vol_col:
            df.rename(columns={vol_col: "volume"}, inplace=True)
        else:
            df["volume"] = pd.NA  # create column if missing

        # Add ticker column
        df["ticker"] = ticker

        # Compute period returns
        df["returns"] = df["close"].pct_change()

        # Append to all_data
        all_data = pd.concat([all_data, df], ignore_index=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    all_data.to_csv(DATA_DIR / "all_data.csv", index=False)
    return all_data


def generate_signals(df):
    pass


def backtest(df):
    pass


def gpt_summary(results):
    pass


def agent_run(cfg):
    tick = cfg["tickers"]
    gran = cfg["granularity"]
    price_df = fetch_price_data(tick, gran)
    price_df = convert_all_data_gmt_to_est(DATA_DIR / "all_data.csv")
    headlines_df = scrape_headlines(tick, gran)
    deduped_df = deduplicate_headlines(headlines_df)
    news_for_merge = deduped_df.copy()

    if news_for_merge is not None and not news_for_merge.empty:
        news_for_merge = news_for_merge.rename(
            columns={
                "timestamps": "date_time",
                "clean_headline": "headline",
            }
        )
        news_for_merge["timestamp"] = pd.to_datetime(
            news_for_merge["date_time"], errors="coerce"
        )
        news_for_merge["ticker"] = news_for_merge["ticker"].astype(str).str.upper()
        news_for_merge.to_csv(DATA_DIR / "alpha_news_all_tickers.csv", index=False)

    bucket_news_inplace(DATA_DIR / "alpha_news_all_tickers.csv")
    merged_output_file = DATA_DIR / "merged_output.csv"
    merged_df = merge_datasets(
        prices_file=DATA_DIR / "all_data.csv",
        news_file=DATA_DIR / "alpha_news_all_tickers.csv",
        output_file=merged_output_file,
    )
    print(
        f"Fetched {len(price_df)} price rows, {len(headlines_df)} raw headlines, "
        f"and {len(deduped_df)} deduplicated clusters."
    )
    return {
        "prices": price_df,
        "headlines": headlines_df,
        "headlines_deduped": deduped_df,
        "price_news_merged": merged_df,
        "price_news_merged_path": merged_output_file,
    }


def main():
    config = load_config()
    print("Config loaded")

    args = parse_args()
    config = apply_overrides(config, args)
    tick = config["tickers"]
    gran = config["granularity"]
    m_n = config["agent"]["model"]
    print("Using config:")
    print("Tickers:", tick)
    print("Granularity:", gran)
    print("Model:", m_n)

    results = agent_run(config)


if __name__ == "__main__":
    main()
