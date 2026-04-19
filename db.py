# Manages persistent storage using SQLite for runs, forecasts, and historical performance

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Iterable

import pandas as pd


class SQLiteStore:
    def __init__(self, path: str | Path, schema_path: str | Path):
        self.path = str(path)
        self.schema_path = str(schema_path)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path)

    def _init_db(self) -> None:
        with self._connect() as conn:
            schema = Path(self.schema_path).read_text(encoding="utf-8")
            conn.executescript(schema)
            conn.commit()

    def upsert_observations(self, df: pd.DataFrame, symbol: str, source: str, ingestion_ts: str) -> int:
        rows = [
            (symbol, str(row.ds.date()), float(row.value), source, ingestion_ts)
            for row in df.itertuples(index=False)
        ]
        with self._connect() as conn:
            conn.executemany(
                """
                INSERT INTO raw_observations(symbol, ds, value, source, ingestion_ts)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(symbol, ds) DO UPDATE SET
                  value = excluded.value,
                  source = excluded.source,
                  ingestion_ts = excluded.ingestion_ts
                """,
                rows,
            )
            conn.commit()
        return len(rows)

    def load_series(self, symbol: str) -> pd.DataFrame:
        with self._connect() as conn:
            df = pd.read_sql_query(
                "SELECT ds, value FROM raw_observations WHERE symbol = ? ORDER BY ds ASC",
                conn,
                params=(symbol,),
                parse_dates=["ds"],
            )
        return df

    def save_model_run(self, payload: dict) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO model_runs(
                    run_id, symbol, created_at, model_name, train_size, holdout_size, mae, rmse, mape, notes
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    payload["run_id"], payload["symbol"], payload["created_at"], payload["model_name"],
                    payload["train_size"], payload["holdout_size"], payload["mae"], payload["rmse"],
                    payload["mape"], payload.get("notes", ""),
                ),
            )
            conn.commit()

    def save_forecasts(self, run_id: str, symbol: str, created_at: str, forecast_df: pd.DataFrame) -> None:
        rows = [
            (
                run_id,
                symbol,
                created_at,
                str(row.ds.date()),
                float(row.yhat),
                None if pd.isna(row.lo80) else float(row.lo80),
                None if pd.isna(row.hi80) else float(row.hi80),
                None if pd.isna(row.lo95) else float(row.lo95),
                None if pd.isna(row.hi95) else float(row.hi95),
            )
            for row in forecast_df.itertuples(index=False)
        ]
        with self._connect() as conn:
            conn.executemany(
                """
                INSERT OR REPLACE INTO forecasts(
                    run_id, symbol, forecast_created_at, target_ds, yhat, lo80, hi80, lo95, hi95
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )
            conn.commit()

    def save_report(self, run_id: str, symbol: str, created_at: str, report_json: dict, report_markdown: str) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO daily_reports(run_id, symbol, created_at, report_json, report_markdown)
                VALUES (?, ?, ?, ?, ?)
                """,
                (run_id, symbol, created_at, json.dumps(report_json, ensure_ascii=False, indent=2), report_markdown),
            )
            conn.commit()

    def get_latest_forecast_for_dates(self, symbol: str, dates: Iterable[pd.Timestamp]) -> pd.DataFrame:
        date_strs = [str(pd.Timestamp(d).date()) for d in dates]
        if not date_strs:
            return pd.DataFrame(columns=["target_ds", "yhat", "forecast_created_at", "run_id"])
        placeholders = ",".join(["?"] * len(date_strs))
        query = f"""
        WITH ranked AS (
            SELECT *,
                   ROW_NUMBER() OVER (PARTITION BY target_ds ORDER BY forecast_created_at DESC) AS rn
            FROM forecasts
            WHERE symbol = ? AND target_ds IN ({placeholders})
        )
        SELECT target_ds, yhat, forecast_created_at, run_id
        FROM ranked
        WHERE rn = 1
        ORDER BY target_ds ASC
        """
        with self._connect() as conn:
            df = pd.read_sql_query(query, conn, params=[symbol, *date_strs], parse_dates=["target_ds", "forecast_created_at"])
        return df

    def append_config_change(self, updated_at: str, key: str, old_value: str, new_value: str, reason: str) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO config_history(updated_at, key, old_value, new_value, reason) VALUES (?, ?, ?, ?, ?)",
                (updated_at, key, old_value, new_value, reason),
            )
            conn.commit()
