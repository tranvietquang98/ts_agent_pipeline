CREATE TABLE IF NOT EXISTS raw_observations (
    symbol TEXT NOT NULL,
    ds TEXT NOT NULL,
    value REAL NOT NULL,
    source TEXT NOT NULL,
    ingestion_ts TEXT NOT NULL,
    PRIMARY KEY (symbol, ds)
);

CREATE TABLE IF NOT EXISTS forecasts (
    run_id TEXT NOT NULL,
    symbol TEXT NOT NULL,
    forecast_created_at TEXT NOT NULL,
    target_ds TEXT NOT NULL,
    yhat REAL NOT NULL,
    lo80 REAL,
    hi80 REAL,
    lo95 REAL,
    hi95 REAL,
    PRIMARY KEY (run_id, symbol, target_ds)
);

CREATE TABLE IF NOT EXISTS model_runs (
    run_id TEXT PRIMARY KEY,
    symbol TEXT NOT NULL,
    created_at TEXT NOT NULL,
    model_name TEXT NOT NULL,
    train_size INTEGER NOT NULL,
    holdout_size INTEGER NOT NULL,
    mae REAL,
    rmse REAL,
    mape REAL,
    notes TEXT
);

CREATE TABLE IF NOT EXISTS daily_reports (
    run_id TEXT PRIMARY KEY,
    symbol TEXT NOT NULL,
    created_at TEXT NOT NULL,
    report_json TEXT NOT NULL,
    report_markdown TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS config_history (
    updated_at TEXT NOT NULL,
    key TEXT NOT NULL,
    old_value TEXT,
    new_value TEXT,
    reason TEXT
);
