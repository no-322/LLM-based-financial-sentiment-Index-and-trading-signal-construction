import pandas as pd
import numpy as np


def backtest(daily_df: pd.DataFrame, signal_col: str = "signal_mom"):
    """Backtest the Sentiment Momentum strategy.

    The strategy return for each ticker-day is:
        strategy_ret = signal * next_day_return

    The portfolio return each day is the equal-weighted average
    of all tickers with non-zero signals on that day.

    Parameters
    ----------
    daily_df : pd.DataFrame
        Output of generate_signals(), must contain:
        ['ticker', 'date', 'return_1d', signal_col].
    signal_col : str
        Name of the column containing trading signals.

    Returns
    -------
    results : dict
        Performance metrics (total return, annualized return/vol, Sharpe, max DD).
    equity_curve : pd.Series
        Cumulative equity over time for plotting.
    """
    df = daily_df.copy()
    df = df.sort_values(["date", "ticker"]).reset_index(drop=True)

    # Strategy return per ticker: signal * next-day return
    df["strategy_ret"] = df[signal_col] * df["return_1d"]

    # Aggregate by date
    def _daily_portfolio(group: pd.DataFrame) -> pd.Series:
        active = group[group[signal_col] != 0]
        if len(active) == 0:
            return pd.Series({"port_ret": 0.0})
        return pd.Series({"port_ret": active["strategy_ret"].mean()})

    port = (
        df.groupby("date")
        .apply(_daily_portfolio)
        .reset_index()
        .sort_values("date")
    )

    # Equity curve
    port["equity"] = (1 + port["port_ret"]).cumprod()

    # Performance statistics
    total_return = port["equity"].iloc[-1] - 1
    ann_factor = 252  # approx. trading days per year

    avg_daily = port["port_ret"].mean()
    vol_daily = port["port_ret"].std(ddof=1)

    if len(port) > 1:
        ann_return = (1 + avg_daily) ** ann_factor - 1
        ann_vol = vol_daily * np.sqrt(ann_factor)
    else:
        ann_return = np.nan
        ann_vol = np.nan

    sharpe = ann_return / ann_vol if ann_vol and ann_vol > 0 else np.nan

    # Max drawdown
    rolling_max = port["equity"].cummax()
    drawdown = port["equity"] / rolling_max - 1
    max_dd = drawdown.min()

    results = {
        "Total Return": float(total_return),
        "Annualized Return": float(ann_return) if pd.notna(ann_return) else None,
        "Annualized Volatility": float(ann_vol) if pd.notna(ann_vol) else None,
        "Sharpe Ratio": float(sharpe) if pd.notna(sharpe) else None,
        "Max Drawdown": float(max_dd),
        "N Days": int(len(port)),
    }

    return results, port.set_index("date")["equity"]


if __name__ == "__main__":
    # Example standalone usage (adjust paths as needed)
    from pathlib import Path

    BASE_DIR = Path(__file__).resolve().parent
    DATA_DIR = BASE_DIR / "data"

    in_path = DATA_DIR / "news_deduped_finbert_new.csv"

    daily_df = pd.read_csv(in_path)
    results, equity = backtest(daily_df)

    print("Sentiment Momentum Strategy Results:")
    for k, v in results.items():
        print(f"{k}: {v}")
