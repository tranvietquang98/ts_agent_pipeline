# Orchestrates the end-to-end workflow: data ingestion -> forecasting -> evaluation -> reporting -> config improvement
# Acts as the main entry point and coordinates all components and agents

from __future__ import annotations

import argparse
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from shutil import copyfile

import pandas as pd
import yaml
from dotenv import load_dotenv

from context_search import find_news_context
from db import SQLiteStore
from evaluator import evaluate
from forecasting import fit_and_forecast
from improver import apply_improvements
from ingestion import ingest_yfinance


ROOT = Path(__file__).resolve().parent


def prune_history(history_root: Path, keep: int = 20) -> None:
    if not history_root.exists():
        return
    run_dirs = sorted([p for p in history_root.iterdir() if p.is_dir()], key=lambda p: p.name, reverse=True)
    for old_dir in run_dirs[keep:]:
        import shutil
        shutil.rmtree(old_dir, ignore_errors=True)


def compute_realized_accuracy(store: SQLiteStore, symbol: str, latest_df: pd.DataFrame) -> dict | None:
    relevant_dates = latest_df["ds"].tail(7).tolist()
    prior_fcst = store.get_latest_forecast_for_dates(symbol, relevant_dates)
    if prior_fcst.empty:
        return None
    actual = latest_df[["ds", "value"]].rename(columns={"ds": "target_ds", "value": "actual"})
    merged = actual.merge(prior_fcst, on="target_ds", how="inner")
    if merged.empty:
        return None
    errors = (merged["actual"] - merged["yhat"]).abs()
    ape = (errors / merged["actual"].abs().replace(0, pd.NA)).dropna()
    return {
        "n_realized_points": int(len(merged)),
        "mae": float(errors.mean()),
        "mape": float(ape.mean() * 100) if not ape.empty else None,
        "latest_realized_target": str(merged["target_ds"].max().date()),
    }


def run_pipeline(config_path: str | Path) -> dict:
    load_dotenv()
    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    store = SQLiteStore(ROOT / cfg["storage"]["sqlite_path"], ROOT / "schema.sql")

    symbol = cfg["series"]["symbol"]
    source = cfg["series"]["source"]
    run_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()

    ingest = ingest_yfinance(
        symbol=symbol,
        field=cfg["series"]["field"],
        interval=cfg["series"]["interval"],
        period=cfg["series"]["period"],
    )
    store.upsert_observations(ingest.cleaned, symbol=symbol, source=source, ingestion_ts=now)
    series_df = store.load_series(symbol)

    prior_accuracy = compute_realized_accuracy(store, symbol, series_df)

    artifacts = fit_and_forecast(
        series_df,
        horizon=cfg["forecast"]["horizon"],
        holdout_size=cfg["forecast"]["holdout_size"],
        seasonal_periods=cfg["forecast"].get("seasonal_periods", 5),
        damped_trend=cfg["forecast"].get("damped_trend", True),
        winsorize_returns=cfg["forecast"].get("winsorize_returns", False),
        seasonal=cfg["forecast"].get("seasonal", "add"),
        model_family=cfg["forecast"].get("model_family", "ETS"),
        validation_scheme=cfg.get("validation", {}).get("scheme", "single_holdout"),
        rolling_backtest_windows=cfg.get("validation", {}).get("rolling_backtest_windows", 3),
        xgb=cfg.get("xgb", {}),
        ensemble_components=cfg["forecast"].get("ensemble_components", ["ETS", "naive_last"]),
    )

    news_context = find_news_context(
        symbol=symbol,
        recent_df=series_df.tail(30),
        diagnostics=artifacts.diagnostics,
        metrics=artifacts.holdout_metrics,
        enabled=cfg.get("agent", {}).get("context_search_enabled", True),
        when_days=cfg.get("agent", {}).get("context_search_window_days", 21),
    )

    store.save_model_run(
        {
            "run_id": run_id,
            "symbol": symbol,
            "created_at": now,
            "model_name": str(artifacts.diagnostics.get("model_family", cfg["forecast"].get("model_family", "ETS"))),
            "train_size": len(series_df) - cfg["forecast"]["holdout_size"],
            "holdout_size": cfg["forecast"]["holdout_size"],
            "mae": artifacts.holdout_metrics["mae"],
            "rmse": artifacts.holdout_metrics["rmse"],
            "mape": artifacts.holdout_metrics["mape"],
            "notes": json.dumps(artifacts.diagnostics),
        }
    )
    store.save_forecasts(run_id, symbol, now, artifacts.forecast_df)

    evaluation = evaluate(
        symbol=symbol,
        metrics=artifacts.holdout_metrics,
        diagnostics=artifacts.diagnostics,
        recent_df=series_df.tail(30),
        forecast_df=artifacts.forecast_df,
        prior_accuracy=prior_accuracy,
        news_context={
            "should_search": news_context.should_search,
            "trigger_reasons": news_context.trigger_reasons,
            "summary_points": news_context.summary_points,
            "articles": news_context.articles,
        },
        use_openai_if_available=cfg["agent"].get("use_openai_if_available", True),
    )
    store.save_report(run_id, symbol, now, evaluation.report_json, evaluation.report_markdown)

    new_cfg = apply_improvements(ROOT / config_path, store, evaluation.report_json)

    out_root = ROOT / "outputs" / symbol
    latest_dir = out_root / "latest"
    history_dir = out_root / "history" / f"{datetime.now().strftime('%Y-%m-%d_%H%M%S')}_{run_id[:8]}"
    latest_dir.mkdir(parents=True, exist_ok=True)
    history_dir.mkdir(parents=True, exist_ok=True)
    prune_history(out_root / "history", keep=cfg.get("output", {}).get("keep_history_runs", 20))

    report_json_path = history_dir / "report.json"
    report_md_path = history_dir / "report.md"
    forecast_csv_path = history_dir / "forecast.csv"

    report_json_path.write_text(json.dumps(evaluation.report_json, ensure_ascii=False, indent=2), encoding="utf-8")
    report_md_path.write_text(evaluation.report_markdown, encoding="utf-8")
    artifacts.forecast_df.to_csv(forecast_csv_path, index=False)

    latest_report_json_path = latest_dir / "report.json"
    latest_report_markdown_path = latest_dir / "report.md"
    latest_forecast_csv_path = latest_dir / "forecast.csv"
    copyfile(report_json_path, latest_report_json_path)
    copyfile(report_md_path, latest_report_markdown_path)
    copyfile(forecast_csv_path, latest_forecast_csv_path)

    return {
        "run_id": run_id,
        "symbol": symbol,
        "ingestion_stats": ingest.stats,
        "metrics": artifacts.holdout_metrics,
        "diagnostics": artifacts.diagnostics,
        "prior_accuracy": prior_accuracy,
        "used_openai": evaluation.used_openai,
        "news_context_triggered": news_context.should_search,
        "news_context_summary": news_context.summary_points,
        "report_json_path": str(report_json_path),
        "report_markdown_path": str(report_md_path),
        "forecast_csv_path": str(forecast_csv_path),
        "latest_report_json_path": str(latest_report_json_path),
        "latest_report_markdown_path": str(latest_report_markdown_path),
        "latest_forecast_csv_path": str(latest_forecast_csv_path),
        "updated_config": new_cfg,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    result = run_pipeline(args.config)
    print(json.dumps(result, indent=2, default=str))
