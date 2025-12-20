import pandas as pd
import numpy as np


def generate_signals(daily_df: pd.DataFrame, z_threshold: float = 1.0) -> pd.DataFrame:
    """Generate trading signals for Strategy 1: Sentiment Momentum.

    Trading rules:
    --------------
    Long (buy):
        sentiment_z > +z_threshold
        AND sentiment_change > 0

    Short (sell):
        sentiment_z < -z_threshold
        AND sentiment_change < 0

    Neutral:
        otherwise (signal = 0)

    Parameters
    ----------
    daily_df : pd.DataFrame
        Dataframe from build_sentiment_index(), with at least:
        ['ticker', 'date', 'sentiment_z', 'sentiment_change'].
    z_threshold : float
        Threshold for defining unusually high or low sentiment.

    Returns
    -------
    pd.DataFrame
        Copy of input with an added 'signal_mom' column in {-1, 0, 1}.
    """
    df = daily_df.copy()
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)

    # Initialize signals to neutral
    df["signal_mom"] = 0

    # Long when sentiment is strongly positive and rising
    long_mask = (df["sentiment_z"] > z_threshold) & (df["sentiment_change"] > 0)
    df.loc[long_mask, "signal_mom"] = 1

    # Short when sentiment is strongly negative and falling
    short_mask = (df["sentiment_z"] < -z_threshold) & (df["sentiment_change"] < 0)
    df.loc[short_mask, "signal_mom"] = -1

    return df


if __name__ == "__main__":
    # Example standalone usage (adjust paths as needed)
    from pathlib import Path

    BASE_DIR = Path(__file__).resolve().parent
    DATA_DIR = BASE_DIR / "data"

    in_path = DATA_DIR / "news_deduped_finbert_new.csv"
    out_path = DATA_DIR / "daily_with_signals.csv"

    daily_df = pd.read_csv(in_path)
    daily_df = generate_signals(daily_df)
    daily_df.to_csv(out_path, index=False)
    print(f"Saved daily data with signals to {out_path}")
