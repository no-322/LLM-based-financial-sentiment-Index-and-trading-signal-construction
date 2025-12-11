import pandas as pd
from datetime import timedelta
import numpy as np

# ---------------------------------------------------------
# PART 1 — Convert all_data timestamps (GMT → EST) + cleanup
# ---------------------------------------------------------
def convert_all_data_gmt_to_est(file_path="all_data.csv"):
    df = pd.read_csv(file_path)

    # Capitalize ticker column
    if "ticker" in df.columns:
        df["ticker"] = df["ticker"].astype(str).str.upper()

    # Convert timestamp from GMT → Eastern and remove timezone
    df["timestamp"] = (
        pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        .dt.tz_convert("US/Eastern")
        .dt.tz_localize(None)  # REMOVE timezone
    )

    df.to_csv(file_path, index=False)
    print(f"[OK] all_data.csv converted to EST (no timezone) and tickers capitalized")
    return df


# ------------------------------------------------------------------
# PART 2 — Bucket alpha_news timestamps + fix tickers (uppercase)
# ------------------------------------------------------------------
def bucket_news_inplace(file_path="alpha_news_all_tickers.csv"):
    df = pd.read_csv(file_path)

    # Capitalize ticker column
    if "ticker" in df.columns:
        df["ticker"] = df["ticker"].astype(str).str.upper()

    # Convert to datetime (NO timezone adjustments)
    df["date_time"] = pd.to_datetime(df["date_time"], errors="coerce")

    # Minutes since midnight
    minutes = df["date_time"].dt.hour * 60 + df["date_time"].dt.minute + df["date_time"].dt.second / 60

    # Bucket edges (minutes)
    bucket_edges = [0, 570, 630, 690, 750, 810, 870, 930, 1440]
    # Bucket output times
    bucket_times = [(9,30), (10,30), (11,30), (12,30), (13,30), (14,30), (15,30), (9,30)]

    bucket_idx = np.searchsorted(bucket_edges, minutes, side="right") - 1
    next_day_mask = bucket_idx == 7   # aftermarket → next morning

    # Get hours + minutes for bucket
    hours = [bucket_times[i][0] for i in bucket_idx]
    mins  = [bucket_times[i][1] for i in bucket_idx]

    # Normalize date to midnight, then add bucket time
    df["date_time"] = df["date_time"].dt.normalize()
    df["date_time"] += pd.to_timedelta(hours, unit="h")
    df["date_time"] += pd.to_timedelta(mins, unit="m")

    # Add 1 day for aftermarket bucket
    df.loc[next_day_mask, "date_time"] += timedelta(days=1)

    df.to_csv(file_path, index=False)
    print(f"[OK] alpha_news_all_tickers.csv bucketed and tickers capitalized")
    return df


# ------------------------------------------------------------------
# PART 3 — Merge both datasets (full outer merge)
# ------------------------------------------------------------------
def merge_datasets(
    prices_file="all_data.csv",
    news_file="alpha_news_all_tickers.csv",
    output_file="merged_output.csv"
):
    # Load cleaned files
    df_prices = pd.read_csv(prices_file)
    df_news = pd.read_csv(news_file)

    # Convert timestamps back to datetime and strip timezone info to avoid merge mismatches
    df_prices["timestamp"] = pd.to_datetime(df_prices["timestamp"], errors="coerce").dt.tz_localize(None)
    df_news["date_time"] = pd.to_datetime(df_news["date_time"], errors="coerce").dt.tz_localize(None)

    # FULL OUTER MERGE to keep all rows from both datasets
    merged = pd.merge(
        df_news,
        df_prices,
        how="outer",
        left_on=["date_time", "ticker"],
        right_on=["timestamp", "ticker"],
        suffixes=("", "_price")
    )

    # Remove duplicate timestamp column from prices
    if "timestamp" in merged.columns:
        merged.drop(columns=["timestamp"], inplace=True)

    # Sort data chronologically for each ticker
    merged.sort_values(by=["ticker", "date_time"], inplace=True)
    merged.reset_index(drop=True, inplace=True)

    # ---------------------------------------------------------
    # Forward-fill close, returns, volume from next available row
    # ---------------------------------------------------------
    price_cols = ["close", "returns", "volume"]

    # Group by ticker
    for ticker, group in merged.groupby("ticker", sort=False):
        missing_mask = group[price_cols].isna().all(axis=1)
        if missing_mask.any():
            idx_missing = missing_mask[missing_mask].index
            for i in idx_missing:
                for j in range(i+1, len(group)):
                    if not group.loc[j, price_cols].isna().all():
                        merged.loc[i, price_cols] = group.loc[j, price_cols].values
                        # Update datetime to match next available row
                        merged.loc[i, "date_time"] = group.loc[j, "date_time"]
                        break

    # Save merged dataset
    merged.to_csv(output_file, index=False)
    print(f"[OK] Merged dataset saved as {output_file}")

    # ---------------------------------------------------------
    # Count rows with non-NaN headline, source, summary
    # ---------------------------------------------------------
    non_na_count = merged[["headline", "source", "summary"]].notna().all(axis=1).sum()
    print(f"Number of rows with non-NaN headline, source, summary: {non_na_count}")

    return merged


# ---------------------------------------------------------
# MAIN EXECUTION PIPELINE
# ---------------------------------------------------------
if __name__ == "__main__":
    convert_all_data_gmt_to_est()
    bucket_news_inplace()
    merge_datasets()
