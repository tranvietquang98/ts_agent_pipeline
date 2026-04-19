# Handles data fetching from external APIs, cleaning, and preprocessing
# Ensures time series integrity (dates, missing values, duplicates) before modeling

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import yfinance as yf


@dataclass
class IngestionResult:
    cleaned: pd.DataFrame
    stats: dict


def ingest_yfinance(symbol: str, field: str = "Close", interval: str = "1d", period: str = "2y") -> IngestionResult:
    raw = yf.download(symbol, interval=interval, period=period, auto_adjust=False, progress=False)
    if raw.empty:
        raise ValueError(f"No data returned for {symbol}.")

    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = [c[0] for c in raw.columns]

    df = raw.reset_index()[["Date", field]].rename(columns={"Date": "ds", field: "value"})
    df["ds"] = pd.to_datetime(df["ds"]).dt.tz_localize(None)

    before = len(df)
    duplicate_count = int(df.duplicated(subset=["ds"]).sum())
    df = df.drop_duplicates(subset=["ds"], keep="last").sort_values("ds")

    missing_count = int(df["value"].isna().sum())
    df["value"] = df["value"].ffill().bfill()
    df = df.dropna(subset=["value"]).reset_index(drop=True)

    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["value"]).reset_index(drop=True)

    return IngestionResult(
        cleaned=df,
        stats={
            "rows_before_cleaning": before,
            "rows_after_cleaning": len(df),
            "duplicates_removed": duplicate_count,
            "missing_filled": missing_count,
            "ingested_at": datetime.now(timezone.utc).isoformat(),
        },
    )
