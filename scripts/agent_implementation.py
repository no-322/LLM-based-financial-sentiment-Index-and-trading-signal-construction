#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Agentic trading script using Ollama and FinBERT-based daily features.

What it does:
- Reads a CSV with per-ticker, per-day price and sentiment features.
- For each (ticker, date), builds features using ONLY data up to that date.
- Calls an LLM via Ollama to get:
    {
      "action": "BUY" or "SELL",
      "position_size": float between 0 and 1,
      "reasoning": "short explanation"
    }
- Handles days with missing news: still generates a signal from price + FinBERT daily aggregates.
- Writes an output CSV with signals appended.

Assumed columns in the input CSV:
- 'ticker'
- ONE of: 'date', 'date_time', or 'timestamp_price' (used to build a 'date' column)
- 'close', 'volume', 'returns'
- Optional FinBERT daily sentiment columns:
    'finbert_score',
    'finbert_prob_negative',
    'finbert_prob_neutral',
    'finbert_prob_positive'

Adjust INPUT_CSV, OUTPUT_CSV, MODEL_NAME as needed.
"""

import json
import textwrap
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd
import requests

# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

INPUT_CSV = str(DATA_DIR / "news_deduped_finbert_new.csv")  # pipeline output
OUTPUT_CSV = str(DATA_DIR / "agent_signals_output.csv")
MODEL_NAME = "llama3.1"  # make sure this model is pulled in Ollama
OLLAMA_URL = "http://localhost:11434/api/chat"

LOOKBACK_DAYS = 10  # rolling window for price/sentiment features


# ---------------------------------------------------------------------
# OLLAMA CALL
# ---------------------------------------------------------------------

def call_ollama_chat(model: str, system_prompt: str, user_prompt: str) -> str:
    """
    Call Ollama's /api/chat endpoint with a system + user message.
    Returns the content of the last message as a string.
    """
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "stream": False,
    }

    try:
        resp = requests.post(OLLAMA_URL, json=payload, timeout=120)
        resp.raise_for_status()
        data = resp.json()

        # Newer Ollama returns: {"message": {"role": "assistant", "content": "..."}}
        if "message" in data and "content" in data["message"]:
            return data["message"]["content"]

        # Fallback: if it has "messages" list
        if "messages" in data and data["messages"]:
            return data["messages"][-1].get("content", "")

        raise ValueError("Unexpected Ollama response format")

    except Exception as e:
        print(f"[WARN] Ollama call failed: {e}")
        return ""


# ---------------------------------------------------------------------
# FEATURE ENGINEERING
# ---------------------------------------------------------------------

FINBERT_COLS = [
    "finbert_score",
    "finbert_prob_negative",
    "finbert_prob_neutral",
    "finbert_prob_positive",
]


def build_price_features_for_day(
    price_ticker_df: pd.DataFrame,
    current_idx: int,
    lookback_days: int = LOOKBACK_DAYS,
) -> Dict[str, Any]:
    """
    Build numeric features for a single (ticker, date) using ONLY data
    up to and including current_idx (no look-ahead).

    price_ticker_df: filtered dataframe for a single ticker, sorted by date.
    current_idx: integer positional index in price_ticker_df.
    """

    start_idx = max(0, current_idx - lookback_days + 1)
    window = price_ticker_df.iloc[start_idx:current_idx + 1]
    row = price_ticker_df.iloc[current_idx]

    # Safety: avoid NaNs causing weird JSON
    def safe_float(x: Any, default: float = 0.0) -> float:
        try:
            val = float(x)
            if np.isnan(val):
                return default
            return val
        except Exception:
            return default

    feats: Dict[str, Any] = {
        "close": safe_float(row.get("close", np.nan)),
        "volume": safe_float(row.get("volume", np.nan)),
        "return": safe_float(row.get("returns", np.nan)),

        "rolling_return_mean": safe_float(window["returns"].mean()),
        "rolling_return_std": safe_float(window["returns"].std(ddof=0)),
        "max_close_last_window": safe_float(window["close"].max()),
        "min_close_last_window": safe_float(window["close"].min()),
    }

    # Drawdown from recent max
    max_close = window["close"].max()
    feats["drawdown_from_max"] = safe_float(
        row.get("close", np.nan) / max_close - 1.0 if max_close not in (0, np.nan) else np.nan
    )

    # Add FinBERT daily sentiment columns if they exist
    for col in FINBERT_COLS:
        if col in price_ticker_df.columns:
            feats[col] = safe_float(row.get(col, np.nan))

    return feats


# ---------------------------------------------------------------------
# LLM PROMPTS
# ---------------------------------------------------------------------

SYSTEM_PROMPT = textwrap.dedent("""
You are an algorithmic trading assistant for equities.

You receive, for each ticker and date:
- price_features: numeric features computed ONLY from historical and current data
  up to that date. These include:
  - price and returns (close, volume, return)
  - rolling statistics (rolling_return_mean, rolling_return_std, drawdown_from_max, etc.)
  - when available, FinBERT-based daily sentiment aggregates:
      finbert_score,
      finbert_prob_negative,
      finbert_prob_neutral,
      finbert_prob_positive

Your task:
- Decide whether to take a BUY or SELL position for *today*.
- Decide on a position_size between 0 and 1 (fraction of capital).
- Provide a short natural language reasoning.

Important rules:
- You MUST use only the information passed in the JSON input.
- You MUST treat the features as historical-only (no future information).
- You MUST ALWAYS return a JSON object with exactly these keys:
  {
    "action": "BUY" or "SELL",
    "position_size": float between 0 and 1,
    "reasoning": "short explanation"
  }
- Be conservative on very noisy or ambiguous signals:
  - If sentiment is strongly positive and returns are not extremely overextended,
    you can lean towards BUY with moderate size.
  - If sentiment is strongly negative or there is a sharp drawdown,
    you can lean towards SELL or small/zero exposure.
- Do NOT add any extra keys or commentary outside the JSON.
""")


def build_user_prompt(
    ticker: str,
    trade_date: Any,
    price_features: Dict[str, Any],
) -> str:
    """
    Build the user message describing context and asking for JSON-only output.
    """
    prompt = {
        "ticker": str(ticker),
        "date": str(trade_date),
        "price_features": price_features,
    }

    return textwrap.dedent(f"""
    You are evaluating a trading decision for this equity.

    Input (JSON):
    {json.dumps(prompt, indent=2)}

    Based on this information, respond with a JSON object of the form:
    {{
      "action": "BUY" or "SELL",
      "position_size": float between 0 and 1,
      "reasoning": "short explanation"
    }}

    Do NOT include any text outside the JSON.
    """)


# ---------------------------------------------------------------------
# MAIN PIPELINE
# ---------------------------------------------------------------------

def parse_llm_response(raw_text: str) -> Dict[str, Any]:
    """
    Parse the LLM output as JSON. If parsing fails, return a fallback.
    """
    fallback = {
        "action": "SELL",
        "position_size": 0.0,
        "reasoning": "Fallback decision due to invalid LLM output.",
    }

    if not raw_text:
        return fallback

    # Try to find a JSON substring in the response
    try:
        raw_text = raw_text.strip()
        # If model added extra text, try to locate the first '{' and last '}'
        if not raw_text.startswith("{"):
            start = raw_text.find("{")
            end = raw_text.rfind("}")
            if start != -1 and end != -1:
                raw_text = raw_text[start:end + 1]

        data = json.loads(raw_text)

        action = str(data.get("action", "SELL")).upper()
        if action not in ("BUY", "SELL"):
            action = "SELL"

        pos_size = float(data.get("position_size", 0.0))
        if pos_size < 0:
            pos_size = 0.0
        if pos_size > 1:
            pos_size = 1.0

        reasoning = str(data.get("reasoning", ""))

        return {
            "action": action,
            "position_size": pos_size,
            "reasoning": reasoning,
        }
    except Exception as e:
        print(f"[WARN] Failed to parse LLM JSON: {e}")
        return fallback


def main():
    print(f"[INFO] Reading input CSV: {INPUT_CSV}")
    df = pd.read_csv(INPUT_CSV)
    # Only keep first 5 rows with a non-empty headline
    if "headline" in df.columns:
        df = df[df["headline"].notna() & (df["headline"].astype(str).str.strip() != "")]
    df = df.head(5)

    if "ticker" not in df.columns:
        raise ValueError("Input CSV must have a 'ticker' column.")

    # Build a 'date' column from any available time column
    date_col = None
    for c in ["date", "date_time", "timestamp_price"]:
        if c in df.columns:
            date_col = c
            break

    if date_col is None:
        raise ValueError("Expected one of 'date', 'date_time', or 'timestamp_price' in the CSV.")

    df["date"] = pd.to_datetime(df[date_col]).dt.date

    # Sort by ticker, date (and keep original row order for reference)
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)

    # Prepare containers for results
    llm_raw_outputs = []
    llm_actions = []
    llm_position_sizes = []
    llm_reasonings = []

    # Process one ticker at a time
    tickers = df["ticker"].unique()
    print(f"[INFO] Found {len(tickers)} tickers: {list(tickers)}")

    for ticker in tickers:
        print(f"[INFO] Processing ticker: {ticker}")
        df_ticker = df[df["ticker"] == ticker].copy().reset_index()

        for local_idx, row in df_ticker.iterrows():
            global_idx = row["index"]  # original index in df
            trade_date = row["date"]

            # Build price + FinBERT features using ONLY history up to today
            price_features = build_price_features_for_day(
                df[df["ticker"] == ticker].reset_index(drop=True),
                current_idx=local_idx,
                lookback_days=LOOKBACK_DAYS,
            )

            user_prompt = build_user_prompt(
                ticker=ticker,
                trade_date=trade_date,
                price_features=price_features,
            )

            raw_llm = call_ollama_chat(
                model=MODEL_NAME,
                system_prompt=SYSTEM_PROMPT,
                user_prompt=user_prompt,
            )

            parsed = parse_llm_response(raw_llm)

            # Store results aligned with global row index
            llm_raw_outputs.append((global_idx, raw_llm))
            llm_actions.append((global_idx, parsed["action"]))
            llm_position_sizes.append((global_idx, parsed["position_size"]))
            llm_reasonings.append((global_idx, parsed["reasoning"]))

    # Attach results back to the main dataframe
    df["llm_raw_output"] = ""
    df["llm_action"] = ""
    df["llm_position_size"] = 0.0
    df["llm_reasoning"] = ""

    for idx, val in llm_raw_outputs:
        df.at[idx, "llm_raw_output"] = val
    for idx, val in llm_actions:
        df.at[idx, "llm_action"] = val
    for idx, val in llm_position_sizes:
        df.at[idx, "llm_position_size"] = val
    for idx, val in llm_reasonings:
        df.at[idx, "llm_reasoning"] = val

    print(f"[INFO] Writing output CSV: {OUTPUT_CSV}")
    df.to_csv(OUTPUT_CSV, index=False)
    print("[INFO] Done.")


if __name__ == "__main__":
    main()
