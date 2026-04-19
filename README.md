# Adaptive Time-Series Forecasting Pipeline with AI Evaluation

This project is an end-to-end forecasting pipeline for live daily time-series data. It ingests fresh market data, generates forecasts, evaluates those forecasts with an AI agent, enriches the evaluation with recent news/event context, and writes structured improvement actions back into configuration for future runs.

The goal is not to build the most accurate forecaster possible. The goal is to demonstrate a collaborative workflow across components:

- live data ingestion
- forecasting
- AI-based evaluation
- structured reporting
- automated config updates for the next run

The final system became an adaptive across-runs loop, not a fully autonomous “iterate until optimum” optimizer. Each run performs one evaluation/improvement pass, and the next run uses the updated config.

---

## Architecture

```text
[Live Data Ingestion] -> [Prediction Model] -> [AI Evaluator Agent] -> [Report Output] -> [Improvement Agent]
```

### Implemented components
- Data source: Yahoo Finance via `yfinance`
- Storage: SQLite + JSON/Markdown/CSV outputs
- Forecasting model families:
  - ETS
  - XGB
  - naive last-value
  - moving average
  - ETS+benchmark ensemble
- Evaluation modes:
  - OpenAI structured-output evaluator
  - heuristic fallback evaluator
- Optional context layer: recent news/event search via Google News RSS
- Adaptation layer: config updates written back into `config.yaml`

---

## What the pipeline does

Each run performs these steps:

1. Pull the latest daily series from a public API.
2. Clean and store the data locally.
3. Train a forecasting model and forecast the next 7 business days.
4. Compute:
   - point forecasts
   - 80% prediction intervals
   - 95% prediction intervals
   - holdout metrics: MAE, RMSE, MAPE
   - interval coverage metrics
5. Detect unusual recent behavior and optionally search for recent news/events that may explain it.
6. Run an evaluator agent that:
   - assesses forecast reasonableness
   - diagnoses likely issues
   - contextualizes unusual moves with news
   - recommends whether to trust, monitor, retrain, or adjust
7. Save:
   - a machine-readable JSON report
   - a human-readable Markdown report
   - forecast CSV output
   - run metadata in SQLite
8. Apply supported improvement actions back into `config.yaml` for the next run.

---

## Project structure

```text
.
├── pipeline.py
├── forecasting.py
├── evaluator.py
├── improver.py
├── context_search.py
├── ingestion.py
├── db.py
├── schema.sql
├── config.yaml
├── requirements.txt
├── outputs/
│   └── AAPL/
│       ├── latest/
│       └── history/
└── pipeline.db
```

### File overview

#### `pipeline.py`
Top-level orchestrator. It:
- loads config
- ingests fresh data
- persists observations
- computes realized forecast accuracy if earlier forecasts matured
- runs forecasting
- runs context search
- runs evaluation
- saves reports and forecasts
- applies config improvements
- writes outputs to `outputs/<SYMBOL>/latest` and `outputs/<SYMBOL>/history`

#### `forecasting.py`
Forecast engine. It:
- supports multiple model families
- handles holdout evaluation
- computes forecast intervals
- computes holdout interval coverage
- supports:
  - `seasonal`
  - `damped_trend`
  - `winsorize_returns`
  - `model_family`
  - rolling backtest validation

#### `evaluator.py`
Evaluation layer. It:
- builds a structured prompt
- can call the OpenAI Responses API when a key is available
- falls back to a deterministic heuristic evaluator otherwise
- outputs JSON and Markdown reports
- integrates recent news/event context into the diagnosis
- emits cleaner absolute `set` actions for downstream config updates

#### `improver.py`
Improvement agent. It:
- reads `evaluation.improvement_actions`
- validates allowed parameters
- safely updates nested config values
- treats most hyperparameters as absolute targets rather than numeric deltas
- logs config changes into SQLite
- writes modified config back to disk

#### `context_search.py`
Optional context module. It:
- detects whether unusual behavior justifies a search
- fetches recent articles from Google News RSS
- ranks articles by ticker/company relevance
- returns summary bullets plus structured article metadata

#### `ingestion.py`
Live data ingestion layer. It:
- downloads time-series data from Yahoo Finance
- selects the configured field
- standardizes schema
- removes duplicates
- handles missing values
- returns cleaned data and ingestion stats

#### `db.py`
SQLite access layer. It stores:
- observations
- model runs
- forecasts
- reports
- config changes

---

## Forecasting logic

### Forecast horizon
Default forecast horizon: 7 business days

### Supported model families

#### 1. ETS
Exponential smoothing state-space model using `statsmodels`.

Why it is chosen for this project:
- fast
- interpretable
- supports trend/seasonality variants
- supports prediction intervals naturally

#### 2. XGB
XGBoost regressor with lagged/rolling tabular features.

Why it is chosen for this project:
- strong short-horizon point forecasting potential
- flexible nonlinear feature interactions
- easy to enrich with engineered features

Limitation:
- intervals are approximated from residual dispersion rather than produced natively

#### 3. `naive_last`
Forecasts the last observed value forward.

Useful as a simple benchmark.

#### 4. `moving_average`
Forecasts from a short recent average.

Useful as a basic smoothing benchmark.

#### 5. `ETS+benchmark_ensemble`
Combines ETS and benchmark/model components.

Useful in principle for balancing:
- ETS stability
- XGB responsiveness

---

## Validation and metrics

The pipeline supports:
- single holdout
- rolling backtest

Reported metrics include:
- MAE
- RMSE
- MAPE
- holdout_coverage_80
- holdout_coverage_95
- rolling-backtest mean error metrics
- rolling-backtest mean coverage metrics

---

## AI evaluator behavior

The evaluator is the core “agent” in the system.

It reviews:
- holdout metrics
- rolling-backtest results
- recent observations
- forecast shape
- interval widths and coverage
- anomaly signals
- optional news/event context

It then produces:
- a verdict: `trust`, `monitor`, `retrain`, or `adjust`
- a confidence level
- a short summary
- reasonableness checks
- diagnosis
- recommendations
- improvement actions

### OpenAI mode
If `OPENAI_API_KEY` is set and `use_openai_if_available: true`, the evaluator uses the OpenAI Responses API with structured JSON output.

### Fallback mode
If no key is available, or the API call fails, the evaluator falls back to a deterministic rules-based evaluator. This keeps the pipeline functional even when the LLM path is unavailable.

---

## News / event context layer

This project implements the optional evaluator enhancement: search the web or use a tool to find news/events that may explain unusual patterns.

### When search is triggered
The context search activates when one or more of these are detected:
- high anomaly count
- unusually large recent trend
- large one-day move
- elevated forecast error

### What it does
- queries Google News RSS
- ranks recent articles
- extracts top recent headlines
- feeds that context into the evaluator prompt
- includes a `news_context` section in the final JSON and Markdown reports

### Why it matters
A purely statistical model cannot know whether a move is tied to:
- earnings expectations
- product announcements
- AI-related developments
- analyst target changes
- broader market sentiment

The context layer helps the evaluator distinguish between:
- pure noise
- regime shifts
- event-driven company-specific moves

---

## Improvement loop

The improvement agent reads structured `improvement_actions` and writes supported changes into `config.yaml`.

Examples:
- change `forecast.model_family`
- enable/disable `forecast.winsorize_returns`
- change `forecast.seasonal`
- change `forecast.damped_trend`
- switch validation scheme
- increase rolling backtest windows

### Important design note
This is an across-runs adaptive loop, not an internal optimizer that keeps rerunning until convergence.

This is a deliberate choice for this project, with an upgrade into a feedback-loop system of agents planned in the future.

---

## Output files

Outputs are stored as:

```text
outputs/<SYMBOL>/latest/
outputs/<SYMBOL>/history/<timestamp_runid>/
```

### Generated artifacts
Each run produces:
- `report.json`
- `report.md`
- `forecast.csv`

### Why both latest and history exist
- `latest/` gives quick access to the most recent run
- `history/` preserves an audit trail of previous runs. Currently set to 20 historical runs; number can be adjusted in config.yaml

---

## Initial Observations

The project was tested across multiple model families and multiple adaptive reruns. AAPL was chosen for all runs as an initial benchmark.

### 1. XGB
Observed behavior:
- strongest point-forecast accuracy on some recent holdout windows
- but badly miscalibrated intervals
- and weaker robustness across rolling-backtest windows

Interpretation:
- XGB was promising as a point forecaster
- but current uncertainty estimation was not reliable enough

### 2. ETS+XGB ensemble
Observed behavior:
- plausible compromise in theory
- but in practice often too mean-reverting or under-calibrated
- only became fully assessable after validation and coverage bugs were fixed

Interpretation:
- the ensemble was a useful experimental direction
- but it did not emerge as the most reliable final choice in the current implementation

### 3. ETS
Observed behavior:
- most stable and interpretable family
- generally weaker point accuracy than the best XGB runs
- but more coherent validation behavior
- remaining issue was a tradeoff between:
  - responsiveness to recent event-driven momentum
  - conservative interval calibration

Interpretation:
- ETS emerged as the most practical final model family for this particular test

---

### Human judgment still matters
The evaluator–improver loop worked, but the experiments also showed why final model selection still benefits from human judgment:
- some parameter changes improved responsiveness but hurt stability
- some changes improved robustness but made the model too sluggish
- repeated agent adjustments exposed the tuning tradeoff rather than eliminating it automatically

---

## Installation

### 1. Create a virtual environment
```bash
python -m venv .venv
```

### 2. Activate it

#### PowerShell
```powershell
.venv\Scripts\Activate.ps1
```

#### Command Prompt
```cmd
.venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Optional: set OpenAI API key

#### PowerShell
```powershell
$env:OPENAI_API_KEY="your_api_key_here"
```

#### Command Prompt
```cmd
set OPENAI_API_KEY=your_api_key_here
```

---

## Running the pipeline

```bash
python pipeline.py --config config.yaml
```

---

## Example config

```yaml
series:
  source: yfinance
  symbol: AAPL
  field: Close
  interval: 1d
  period: 2y

storage:
  sqlite_path: pipeline.db

forecast:
  horizon: 7
  holdout_size: 30
  model_family: ETS
  seasonal: none
  seasonal_periods: 5
  damped_trend: false
  winsorize_returns: false
  ensemble_components:
    - ETS
    - XGB

validation:
  scheme: rolling_backtest
  rolling_backtest_windows: 24

xgb:
  lags: [1, 2, 3, 5, 10]
  rolling_windows: [5, 10]
  max_depth: 4
  learning_rate: 0.05
  n_estimators: 300
  subsample: 0.9
  colsample_bytree: 0.9
  min_child_weight: 3
  random_state: 42

agent:
  model: gpt-5.4
  use_openai_if_available: true
```

---

## Key limitations

- ETS is still a relatively simple baseline and can lag sharp repricings
- XGB interval estimates are approximate, not fully probabilistic
- News context quality depends on public RSS results and ranking heuristics
- The evaluator–improver loop is adaptive across runs, not a true internal optimizer
- Realized forecast tracking is still limited until more live forecasts mature

