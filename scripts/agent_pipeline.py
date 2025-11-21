def scrape_headlines(tickers = ["AAPL", "NET", "PLTR", "JPM", "SCHW", "SOFI"],granularity='h'):
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

def clean_headlines(df):
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

def fetch_price_data(tickers = ["AAPL", "NET", "PLTR", "JPM", "SCHW", "SOFI"]):
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
    """
    pass

def generate_signals(df):
    pass

def backtest(df):
    pass

def gpt_summary(results):
    pass

def agent_run():
    # call everything in order
    pass
