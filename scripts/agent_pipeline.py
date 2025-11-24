import yaml
import argparse
import os
import pandas as pd

def load_config(path="config.yml")->dict:
    """
    Load the YAML config file and return a dictionary.
    
    Args:
        path (str): Path to the YAML file.

    Returns:
        dict: Configuration dictionary with defaults applied.
    """

    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg


def parse_args():
    """
    Parse command-line arguments for overriding config values.
    """
    parser = argparse.ArgumentParser(description="Run Agentic AI pipeline")

    parser.add_argument(
        "--tickers",
        type=list,
        help="List of tickers to override config (e.g. ['AAPL','MSFT'])",
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

    if args.model is not None:
        config["model"] = args.model

    return config

def scrape_headlines(tickers=tick ,granularity=gran)->pd.DataFrame:
    """
    Return headlines from different sources
    
    Args:
        tickers (list): Ticker data to extract
        granularity (str): Frequency of price data. Supported values:
            - 'h': hourly bars
            - 'd': daily barsourly or d for daily
    
    Returns:
        pandas.Dataframe: A dataframe with the following columns:
            - date (datetime64): Trading Date
            - ticker (str): Ticker pertaining to the headline
            - headline (str): headline extracted
            - source (str): data source
    """
    
    pass

def clean_headlines(df)->pd.DataFrame:
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

def fetch_price_data(tickers = tick,granularity=gran):
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
    pass

def generate_signals(df):
    pass

def backtest(df):
    pass

def gpt_summary(results):
    pass

def agent_run():
    
    pass

def main():
    cfg=load_config()
    print("Config loaded")

    args = parse_args()
    config = apply_overrides(config, args)
    tick=cfg['tickers']
    gran = cfg["granularity"]
    m_n = cfg["model_name"]
    print("Using config:")
    print("Tickers:", tick)
    print("Granularity:", gran)
    print("Model:", m_n)

    results=agent_run(cfg)
