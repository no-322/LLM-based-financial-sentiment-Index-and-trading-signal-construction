from pathlib import Path

import pandas as pd

from trading_signals import generate_signals
from backtest import backtest

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

MERGED_FILE = DATA_DIR / "merged_output.csv"  # output from scripts/final_data.py
DAILY_WITH_SIGNALS = DATA_DIR / "daily_with_signals_and_returns.csv"
EQUITY_CURVE = DATA_DIR / "equity_curve.csv"


def build_sentiment_index(merged: pd.DataFrame) -> pd.DataFrame:
    merged = merged.copy()
    merged["date_time"] = pd.to_datetime(merged["date_time"], errors="coerce")
    merged["date"] = merged["date_time"].dt.date

    # Aggregate sentiment per ticker-day
    sent = (
        merged.groupby(["ticker", "date"])["overall_sentiment_score"]
        .mean()
        .reset_index()
        .rename(columns={"overall_sentiment_score": "sentiment_score"})
    )
    sent["sentiment_score"] = sent["sentiment_score"].fillna(0.0)
    sent["sentiment_change"] = (
        sent.groupby("ticker")["sentiment_score"].diff().fillna(0.0)
    )
    sent["sentiment_z"] = sent.groupby("ticker")["sentiment_score"].transform(
        lambda x: (x - x.mean()) / x.std(ddof=0) if x.std(ddof=0) else 0.0
    )

    # Daily price returns from last close of the day
    prices = merged.groupby(["ticker", "date"])["close"].last().reset_index()
    prices["return_1d"] = prices.groupby("ticker")["close"].pct_change()

    daily = sent.merge(prices[["ticker", "date", "return_1d"]], on=["ticker", "date"], how="left")
    return daily


def main():
    # Load merged price/news with sentiment
    merged = pd.read_csv(MERGED_FILE)

    # Build daily sentiment index and signals
    daily_df = build_sentiment_index(merged)
    daily_df = generate_signals(daily_df)

    # Backtest
    results, equity = backtest(daily_df)

    print("\n===== Sentiment Momentum Strategy Results =====")
    for k, v in results.items():
        print(f"{k}: {v}")

    # Save outputs
    daily_df.to_csv(DAILY_WITH_SIGNALS, index=False)
    equity.to_csv(EQUITY_CURVE, header=["equity"])
    print(f"Saved daily data with signals to {DAILY_WITH_SIGNALS}")
    print(f"Saved equity curve to {EQUITY_CURVE}")


if __name__ == "__main__":
    main()
