import argparse
from datetime import datetime, timedelta
from pathlib import Path
from typing import Union

import pandas as pd
import yaml
import yfinance as yf

from alphavantage import get_alpha_vantage_news
from deduplicate import run_dedup_pipeline
from merge_news_prices import merge_price_and_news

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
        pandas.Dataframe: Columns include timestamp, ticker, headline, source,
            summary, url, and Alpha Vantage sentiment metadata.
    """
    print("Fetching headlines from Alpha Vantage…")
    try:
        news_df = get_alpha_vantage_news(tickers)
    except Exception as exc:  # keep pipeline running when API fails
        print(f"Failed to fetch headlines: {exc}")
        news_df = pd.DataFrame()

    if news_df.empty:
        print("No headlines returned from Alpha Vantage.")
        return pd.DataFrame(
            columns=[
                "timestamp",
                "ticker",
                "headline",
                "source",
                "summary",
                "url",
                "overall_sentiment_score",
                "overall_sentiment_label",
            ]
        )

    # Ensure expected column exists
    if "time_published" not in news_df.columns:
        print("Missing time_published in news response; skipping headlines.")
        return pd.DataFrame(
            columns=[
                "timestamp",
                "ticker",
                "headline",
                "source",
                "summary",
                "url",
                "overall_sentiment_score",
                "overall_sentiment_label",
            ]
        )

    # Normalize and explode tickers -> one ticker per row
    news_df["timestamp"] = pd.to_datetime(
        news_df["time_published"], format="%Y%m%dT%H%M%S", errors="coerce"
    )
    news_df["ticker"] = (
        news_df["ticker_sentiment"]
        .fillna("")
        .apply(lambda x: [t.strip() for t in x.split(",") if t.strip()])
    )
    news_df = news_df.explode("ticker")

    # Keep core columns expected by downstream steps
    news_df = news_df[
        [
            "timestamp",
            "ticker",
            "headline",
            "source",
            "summary",
            "url",
            "overall_sentiment_score",
            "overall_sentiment_label",
        ]
    ].sort_values("timestamp", ascending=False)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    output_path = DATA_DIR / "alpha_news_all_tickers.csv"
    news_df.to_csv(output_path, index=False)
    print(f"Saved {len(news_df)} headlines to {output_path}")
    return news_df


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
    headlines_df = scrape_headlines(tick, gran)
    deduped_df = deduplicate_headlines(headlines_df)
    merged_path = merge_price_and_news()
    print(
        f"Fetched {len(price_df)} price rows, {len(headlines_df)} raw headlines, "
        f"and {len(deduped_df)} deduplicated clusters."
    )
    return {
        "prices": price_df,
        "headlines": headlines_df,
        "headlines_deduped": deduped_df,
        "price_news_merged_path": merged_path,
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
