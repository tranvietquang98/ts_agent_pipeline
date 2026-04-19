# Implements forecasting models and generates predictions with confidence intervals. Computes holdout and rolling backtest metrics for model evaluation.

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.exponential_smoothing.ets import ETSModel
from xgboost import XGBRegressor


@dataclass
class ForecastArtifacts:
    forecast_df: pd.DataFrame
    holdout_metrics: dict[str, Any]
    diagnostics: dict[str, Any]
    holdout_actual_df: pd.DataFrame
    holdout_pred_df: pd.DataFrame


def _mape(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = y_true != 0
    if mask.sum() == 0:
        return None
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0)


def _winsorize_series(values: pd.Series, lower_q: float = 0.02, upper_q: float = 0.98) -> pd.Series:
    lower = values.quantile(lower_q)
    upper = values.quantile(upper_q)
    return values.clip(lower=lower, upper=upper)


def _prepare_values(df: pd.DataFrame, winsorize_returns: bool) -> pd.Series:
    values = df["value"].astype(float).reset_index(drop=True)

    if not winsorize_returns or len(values) < 10:
        return values

    rets = values.pct_change()
    rets_w = _winsorize_series(rets.dropna())
    rets_adj = rets.copy()
    rets_adj.loc[rets_w.index] = rets_w

    rebuilt = [values.iloc[0]]
    for r in rets_adj.iloc[1:]:
        if pd.isna(r):
            rebuilt.append(rebuilt[-1])
        else:
            rebuilt.append(rebuilt[-1] * (1.0 + r))

    return pd.Series(rebuilt, dtype=float)


def _calc_anomaly_count(values: pd.Series, z_thresh: float = 3.0) -> int:
    returns = values.astype(float).pct_change().dropna()
    if len(returns) <= 10:
        return 0
    std = returns.std(ddof=0)
    if std == 0 or pd.isna(std):
        return 0
    z = (returns - returns.mean()) / std
    return int((z.abs() > z_thresh).sum())


def _calc_recent_trend_5d_pct(values: pd.Series) -> float | None:
    if len(values) < 6:
        return None
    start_val = float(values.iloc[-6])
    end_val = float(values.iloc[-1])
    if start_val == 0:
        return None
    return (end_val / start_val - 1.0) * 100.0


def _coverage(actual: pd.Series, lo: np.ndarray, hi: np.ndarray) -> float:
    actual_arr = actual.to_numpy(dtype=float)
    inside = (actual_arr >= lo) & (actual_arr <= hi)
    return float(inside.mean() * 100.0)


def _rolling_origin_splits(n_obs: int, holdout_size: int, windows: int) -> list[tuple[int, int]]:
    splits = []
    if n_obs <= holdout_size + 20:
        return splits

    last_train_end = n_obs - holdout_size
    min_train = max(40, holdout_size)

    candidate_ends = np.linspace(min_train, last_train_end, num=max(windows, 1), dtype=int)
    seen = set()
    for end in candidate_ends:
        end = int(end)
        if end in seen:
            continue
        seen.add(end)
        if end < min_train or end >= n_obs:
            continue
        splits.append((0, end))
    return splits


def _build_xgb_features(df: pd.DataFrame, lags: list[int], rolling_windows: list[int]) -> pd.DataFrame:
    out = df.copy()
    out["ds"] = pd.to_datetime(out["ds"])
    out = out.sort_values("ds").reset_index(drop=True)
    out["value"] = out["value"].astype(float)

    for lag in lags:
        out[f"lag_{lag}"] = out["value"].shift(lag)

    out["ret_1"] = out["value"].pct_change(1)
    out["ret_2"] = out["value"].pct_change(2)
    out["ret_5"] = out["value"].pct_change(5)

    for win in rolling_windows:
        out[f"roll_mean_{win}"] = out["value"].rolling(win).mean()
        out[f"roll_std_{win}"] = out["value"].rolling(win).std(ddof=0)

    out["dow"] = out["ds"].dt.dayofweek
    out["month"] = out["ds"].dt.month
    out["target"] = out["value"]

    return out.dropna().reset_index(drop=True)


def _xgb_feature_cols(lags: list[int], rolling_windows: list[int]) -> list[str]:
    cols = [f"lag_{lag}" for lag in lags]
    cols += ["ret_1", "ret_2", "ret_5"]
    cols += [f"roll_mean_{win}" for win in rolling_windows]
    cols += [f"roll_std_{win}" for win in rolling_windows]
    cols += ["dow", "month"]
    return cols


def _fit_xgb_model(train_feat: pd.DataFrame, xgb_cfg: dict[str, Any], feature_cols: list[str]) -> XGBRegressor:
    model = XGBRegressor(
        n_estimators=int(xgb_cfg.get("n_estimators", 300)),
        max_depth=int(xgb_cfg.get("max_depth", 4)),
        learning_rate=float(xgb_cfg.get("learning_rate", 0.05)),
        subsample=float(xgb_cfg.get("subsample", 0.9)),
        colsample_bytree=float(xgb_cfg.get("colsample_bytree", 0.9)),
        min_child_weight=float(xgb_cfg.get("min_child_weight", 3)),
        objective="reg:squarederror",
        random_state=int(xgb_cfg.get("random_state", 42)),
    )
    model.fit(train_feat[feature_cols], train_feat["target"])
    return model


def _make_feature_row(history_df: pd.DataFrame, lags: list[int], rolling_windows: list[int], next_ds: pd.Timestamp) -> dict[str, float]:
    s = history_df["value"].astype(float).reset_index(drop=True)
    row: dict[str, float] = {}

    for lag in lags:
        row[f"lag_{lag}"] = float(s.iloc[-lag])

    row["ret_1"] = float(s.iloc[-1] / s.iloc[-2] - 1.0) if len(s) >= 2 and s.iloc[-2] != 0 else 0.0
    row["ret_2"] = float(s.iloc[-1] / s.iloc[-3] - 1.0) if len(s) >= 3 and s.iloc[-3] != 0 else 0.0
    row["ret_5"] = float(s.iloc[-1] / s.iloc[-6] - 1.0) if len(s) >= 6 and s.iloc[-6] != 0 else 0.0

    for win in rolling_windows:
        vals = s.iloc[-win:]
        row[f"roll_mean_{win}"] = float(vals.mean())
        row[f"roll_std_{win}"] = float(vals.std(ddof=0)) if len(vals) > 1 else 0.0

    row["dow"] = int(next_ds.dayofweek)
    row["month"] = int(next_ds.month)
    return row


def _recursive_xgb_forecast(
    history_df: pd.DataFrame,
    horizon: int,
    model: XGBRegressor,
    lags: list[int],
    rolling_windows: list[int],
    feature_cols: list[str],
) -> pd.DataFrame:
    hist = history_df.copy().sort_values("ds").reset_index(drop=True)
    last_date = pd.to_datetime(hist["ds"].iloc[-1])
    future_idx = pd.bdate_range(start=last_date + pd.offsets.BDay(1), periods=horizon)

    preds = []
    for next_ds in future_idx:
        row = _make_feature_row(hist, lags, rolling_windows, next_ds)
        X_next = pd.DataFrame([row])[feature_cols]
        yhat = float(model.predict(X_next)[0])
        preds.append({"ds": next_ds, "yhat": yhat})
        hist = pd.concat([hist, pd.DataFrame([{"ds": next_ds, "value": yhat}])], ignore_index=True)

    return pd.DataFrame(preds)


def _fit_and_forecast_ets(
    data: pd.DataFrame,
    horizon: int,
    holdout_size: int,
    seasonal_periods: int,
    damped_trend: bool,
    winsorize_returns: bool,
    seasonal: str | None,
) -> ForecastArtifacts:
    data = data.copy()
    data["ds"] = pd.to_datetime(data["ds"])
    data = data.sort_values("ds").drop_duplicates(subset=["ds"]).reset_index(drop=True)

    train = data.iloc[:-holdout_size].copy()
    test = data.iloc[-holdout_size:].copy()

    seasonal = None if seasonal in [None, "none", "None", False] else "add"
    seasonal_periods_used = seasonal_periods if seasonal == "add" else None

    train_values = _prepare_values(train, winsorize_returns=winsorize_returns)
    test_values = test["value"].astype(float).reset_index(drop=True)

    model = ETSModel(
        train_values,
        error="add",
        trend="add",
        damped_trend=damped_trend,
        seasonal=seasonal,
        seasonal_periods=seasonal_periods_used,
    )
    fitted = model.fit(disp=False)

    pred_holdout = fitted.get_prediction(
        start=len(train_values),
        end=len(train_values) + len(test_values) - 1,
    )

    holdout_sf_95 = pred_holdout.summary_frame(alpha=0.05)
    holdout_sf_80 = pred_holdout.summary_frame(alpha=0.20)

    test_pred = pd.Series(np.asarray(pred_holdout.predicted_mean), index=test["ds"])

    mae = float(mean_absolute_error(test["value"], test_pred.values))
    rmse = float(np.sqrt(mean_squared_error(test["value"], test_pred.values)))
    mape = _mape(test["value"].to_numpy(), test_pred.values)

    cov80 = _coverage(test["value"], np.asarray(holdout_sf_80["pi_lower"]), np.asarray(holdout_sf_80["pi_upper"]))
    cov95 = _coverage(test["value"], np.asarray(holdout_sf_95["pi_lower"]), np.asarray(holdout_sf_95["pi_upper"]))

    full_values = _prepare_values(data, winsorize_returns=winsorize_returns)

    full_model = ETSModel(
        full_values,
        error="add",
        trend="add",
        damped_trend=damped_trend,
        seasonal=seasonal,
        seasonal_periods=seasonal_periods_used,
    )
    full_fitted = full_model.fit(disp=False)

    pred_95 = full_fitted.get_prediction(
        start=len(full_values),
        end=len(full_values) + horizon - 1,
    )
    pred_80 = full_fitted.get_prediction(
        start=len(full_values),
        end=len(full_values) + horizon - 1,
    )

    forecast_mean = np.asarray(pred_95.predicted_mean)
    ci95 = pred_95.summary_frame(alpha=0.05)
    ci80 = pred_80.summary_frame(alpha=0.20)

    last_date = data["ds"].max()
    future_idx = pd.bdate_range(start=last_date + pd.offsets.BDay(1), periods=horizon)

    forecast_df = pd.DataFrame(
        {
            "ds": future_idx,
            "yhat": forecast_mean,
            "lo95": np.asarray(ci95["pi_lower"]),
            "hi95": np.asarray(ci95["pi_upper"]),
            "lo80": np.asarray(ci80["pi_lower"]),
            "hi80": np.asarray(ci80["pi_upper"]),
        }
    )

    holdout_metrics = {
        "mae": mae,
        "rmse": rmse,
        "mape": mape,
        "holdout_size": int(holdout_size),
        "holdout_coverage_80": cov80,
        "holdout_coverage_95": cov95,
    }

    diagnostics = {
        "model_family": "ETS",
        "base_model_family": "ETS",
        "trend": "add",
        "seasonal": "none" if seasonal is None else "add",
        "seasonal_periods": seasonal_periods_used,
        "damped_trend": damped_trend,
        "winsorize_returns": winsorize_returns,
        "n_obs": int(len(data)),
        "train_end": str(train["ds"].max().date()),
        "test_start": str(test["ds"].min().date()),
        "test_end": str(test["ds"].max().date()),
        "recent_trend_5d_pct": _calc_recent_trend_5d_pct(data["value"]),
        "anomaly_count": _calc_anomaly_count(data["value"]),
    }

    holdout_actual_df = test[["ds", "value"]].copy()
    holdout_pred_df = pd.DataFrame(
        {
            "ds": test["ds"].values,
            "yhat": test_pred.values,
        }
    )

    return ForecastArtifacts(
        forecast_df=forecast_df,
        holdout_metrics=holdout_metrics,
        diagnostics=diagnostics,
        holdout_actual_df=holdout_actual_df,
        holdout_pred_df=holdout_pred_df,
    )


def _fit_and_forecast_xgb(
    data: pd.DataFrame,
    horizon: int,
    holdout_size: int,
    winsorize_returns: bool,
    xgb_cfg: dict[str, Any],
) -> ForecastArtifacts:
    data = data.copy()
    data["ds"] = pd.to_datetime(data["ds"])
    data = data.sort_values("ds").drop_duplicates(subset=["ds"]).reset_index(drop=True)

    if winsorize_returns:
        prepared = data.copy()
        prepared["value"] = _prepare_values(prepared, winsorize_returns=True)
    else:
        prepared = data.copy()

    lags = list(xgb_cfg.get("lags", [1, 2, 3, 5, 10]))
    rolling_windows = list(xgb_cfg.get("rolling_windows", [5, 10]))
    feature_cols = _xgb_feature_cols(lags, rolling_windows)

    feat = _build_xgb_features(prepared, lags=lags, rolling_windows=rolling_windows)
    if len(feat) <= holdout_size + 20:
        raise ValueError("Not enough feature rows for XGB holdout evaluation.")

    train_feat = feat.iloc[:-holdout_size].copy()
    test_feat = feat.iloc[-holdout_size:].copy()

    model = _fit_xgb_model(train_feat, xgb_cfg=xgb_cfg, feature_cols=feature_cols)

    test_pred_vals = model.predict(test_feat[feature_cols])

    residuals = train_feat["target"] - model.predict(train_feat[feature_cols])
    sigma = float(np.std(residuals, ddof=0))
    z80 = 1.2815515655446004
    z95 = 1.959963984540054

    lo80 = test_pred_vals - z80 * sigma
    hi80 = test_pred_vals + z80 * sigma
    lo95 = test_pred_vals - z95 * sigma
    hi95 = test_pred_vals + z95 * sigma

    mae = float(mean_absolute_error(test_feat["target"], test_pred_vals))
    rmse = float(np.sqrt(mean_squared_error(test_feat["target"], test_pred_vals)))
    mape = _mape(test_feat["target"].to_numpy(), test_pred_vals)

    cov80 = _coverage(test_feat["target"], lo80, hi80)
    cov95 = _coverage(test_feat["target"], lo95, hi95)

    full_feat = _build_xgb_features(prepared, lags=lags, rolling_windows=rolling_windows)
    full_model = _fit_xgb_model(full_feat, xgb_cfg=xgb_cfg, feature_cols=feature_cols)

    history_for_recursive = prepared[["ds", "value"]].copy()
    forecast_df = _recursive_xgb_forecast(
        history_df=history_for_recursive,
        horizon=horizon,
        model=full_model,
        lags=lags,
        rolling_windows=rolling_windows,
        feature_cols=feature_cols,
    )

    forecast_df["lo80"] = forecast_df["yhat"] - z80 * sigma
    forecast_df["hi80"] = forecast_df["yhat"] + z80 * sigma
    forecast_df["lo95"] = forecast_df["yhat"] - z95 * sigma
    forecast_df["hi95"] = forecast_df["yhat"] + z95 * sigma

    holdout_metrics = {
        "mae": mae,
        "rmse": rmse,
        "mape": mape,
        "holdout_size": int(holdout_size),
        "holdout_coverage_80": cov80,
        "holdout_coverage_95": cov95,
    }

    diagnostics = {
        "model_family": "XGB",
        "base_model_family": "XGB",
        "trend": None,
        "seasonal": None,
        "seasonal_periods": None,
        "damped_trend": None,
        "winsorize_returns": winsorize_returns,
        "n_obs": int(len(data)),
        "train_end": str(test_feat["ds"].min().date()),
        "test_start": str(test_feat["ds"].min().date()),
        "test_end": str(test_feat["ds"].max().date()),
        "recent_trend_5d_pct": _calc_recent_trend_5d_pct(data["value"]),
        "anomaly_count": _calc_anomaly_count(data["value"]),
        "xgb_lags": lags,
        "xgb_rolling_windows": rolling_windows,
        "xgb_max_depth": int(xgb_cfg.get("max_depth", 4)),
        "xgb_n_estimators": int(xgb_cfg.get("n_estimators", 300)),
        "xgb_learning_rate": float(xgb_cfg.get("learning_rate", 0.05)),
        "xgb_residual_sigma": sigma,
    }

    holdout_actual_df = test_feat[["ds", "target"]].rename(columns={"target": "value"}).copy()
    holdout_pred_df = pd.DataFrame(
        {
            "ds": test_feat["ds"].values,
            "yhat": test_pred_vals,
        }
    )

    return ForecastArtifacts(
        forecast_df=forecast_df,
        holdout_metrics=holdout_metrics,
        diagnostics=diagnostics,
        holdout_actual_df=holdout_actual_df,
        holdout_pred_df=holdout_pred_df,
    )


def _fit_and_forecast_naive_last(
    data: pd.DataFrame,
    horizon: int,
    holdout_size: int,
) -> ForecastArtifacts:
    data = data.copy()
    data["ds"] = pd.to_datetime(data["ds"])
    data = data.sort_values("ds").drop_duplicates(subset=["ds"]).reset_index(drop=True)

    train = data.iloc[:-holdout_size].copy()
    test = data.iloc[-holdout_size:].copy()

    last_train = float(train["value"].iloc[-1])
    test_pred_vals = np.repeat(last_train, len(test))

    resid = train["value"].diff().dropna()
    sigma = float(resid.std(ddof=0)) if len(resid) > 1 else 1.0
    z80 = 1.2815515655446004
    z95 = 1.959963984540054

    mae = float(mean_absolute_error(test["value"], test_pred_vals))
    rmse = float(np.sqrt(mean_squared_error(test["value"], test_pred_vals)))
    mape = _mape(test["value"].to_numpy(), test_pred_vals)
    cov80 = _coverage(test["value"], test_pred_vals - z80 * sigma, test_pred_vals + z80 * sigma)
    cov95 = _coverage(test["value"], test_pred_vals - z95 * sigma, test_pred_vals + z95 * sigma)

    last_date = data["ds"].max()
    future_idx = pd.bdate_range(start=last_date + pd.offsets.BDay(1), periods=horizon)
    yhat = np.repeat(float(data["value"].iloc[-1]), horizon)

    forecast_df = pd.DataFrame(
        {
            "ds": future_idx,
            "yhat": yhat,
            "lo80": yhat - z80 * sigma,
            "hi80": yhat + z80 * sigma,
            "lo95": yhat - z95 * sigma,
            "hi95": yhat + z95 * sigma,
        }
    )

    return ForecastArtifacts(
        forecast_df=forecast_df,
        holdout_metrics={
            "mae": mae,
            "rmse": rmse,
            "mape": mape,
            "holdout_size": int(holdout_size),
            "holdout_coverage_80": cov80,
            "holdout_coverage_95": cov95,
        },
        diagnostics={
            "model_family": "naive_last",
            "base_model_family": "naive_last",
            "trend": None,
            "seasonal": None,
            "seasonal_periods": None,
            "damped_trend": None,
            "winsorize_returns": False,
            "n_obs": int(len(data)),
            "train_end": str(train["ds"].max().date()),
            "test_start": str(test["ds"].min().date()),
            "test_end": str(test["ds"].max().date()),
            "recent_trend_5d_pct": _calc_recent_trend_5d_pct(data["value"]),
            "anomaly_count": _calc_anomaly_count(data["value"]),
        },
        holdout_actual_df=test[["ds", "value"]].copy(),
        holdout_pred_df=pd.DataFrame({"ds": test["ds"].values, "yhat": test_pred_vals}),
    )


def _fit_and_forecast_moving_average(
    data: pd.DataFrame,
    horizon: int,
    holdout_size: int,
    window: int = 5,
) -> ForecastArtifacts:
    data = data.copy()
    data["ds"] = pd.to_datetime(data["ds"])
    data = data.sort_values("ds").drop_duplicates(subset=["ds"]).reset_index(drop=True)

    train = data.iloc[:-holdout_size].copy()
    test = data.iloc[-holdout_size:].copy()

    ma_val = float(train["value"].tail(window).mean())
    test_pred_vals = np.repeat(ma_val, len(test))

    resid = train["value"] - train["value"].rolling(window).mean()
    resid = resid.dropna()
    sigma = float(resid.std(ddof=0)) if len(resid) > 1 else 1.0
    z80 = 1.2815515655446004
    z95 = 1.959963984540054

    mae = float(mean_absolute_error(test["value"], test_pred_vals))
    rmse = float(np.sqrt(mean_squared_error(test["value"], test_pred_vals)))
    mape = _mape(test["value"].to_numpy(), test_pred_vals)
    cov80 = _coverage(test["value"], test_pred_vals - z80 * sigma, test_pred_vals + z80 * sigma)
    cov95 = _coverage(test["value"], test_pred_vals - z95 * sigma, test_pred_vals + z95 * sigma)

    last_date = data["ds"].max()
    future_idx = pd.bdate_range(start=last_date + pd.offsets.BDay(1), periods=horizon)
    yhat = np.repeat(float(data["value"].tail(window).mean()), horizon)

    forecast_df = pd.DataFrame(
        {
            "ds": future_idx,
            "yhat": yhat,
            "lo80": yhat - z80 * sigma,
            "hi80": yhat + z80 * sigma,
            "lo95": yhat - z95 * sigma,
            "hi95": yhat + z95 * sigma,
        }
    )

    return ForecastArtifacts(
        forecast_df=forecast_df,
        holdout_metrics={
            "mae": mae,
            "rmse": rmse,
            "mape": mape,
            "holdout_size": int(holdout_size),
            "holdout_coverage_80": cov80,
            "holdout_coverage_95": cov95,
        },
        diagnostics={
            "model_family": "moving_average",
            "base_model_family": "moving_average",
            "trend": None,
            "seasonal": None,
            "seasonal_periods": None,
            "damped_trend": None,
            "winsorize_returns": False,
            "n_obs": int(len(data)),
            "train_end": str(train["ds"].max().date()),
            "test_start": str(test["ds"].min().date()),
            "test_end": str(test["ds"].max().date()),
            "recent_trend_5d_pct": _calc_recent_trend_5d_pct(data["value"]),
            "anomaly_count": _calc_anomaly_count(data["value"]),
            "ma_window": window,
        },
        holdout_actual_df=test[["ds", "value"]].copy(),
        holdout_pred_df=pd.DataFrame({"ds": test["ds"].values, "yhat": test_pred_vals}),
    )


def _fit_and_forecast_ensemble(
    data: pd.DataFrame,
    horizon: int,
    holdout_size: int,
    seasonal_periods: int,
    damped_trend: bool,
    winsorize_returns: bool,
    seasonal: str | None,
    ensemble_components: list[str],
    xgb_cfg: dict[str, Any],
) -> ForecastArtifacts:
    components = []
    for comp in ensemble_components:
        comp_norm = str(comp).strip()
        if comp_norm == "ETS":
            components.append(
                _fit_and_forecast_ets(
                    data=data,
                    horizon=horizon,
                    holdout_size=holdout_size,
                    seasonal_periods=seasonal_periods,
                    damped_trend=damped_trend,
                    winsorize_returns=winsorize_returns,
                    seasonal=seasonal,
                )
            )
        elif comp_norm == "naive_last":
            components.append(_fit_and_forecast_naive_last(data=data, horizon=horizon, holdout_size=holdout_size))
        elif comp_norm == "moving_average":
            components.append(_fit_and_forecast_moving_average(data=data, horizon=horizon, holdout_size=holdout_size))
        elif comp_norm == "XGB":
            components.append(
                _fit_and_forecast_xgb(
                    data=data,
                    horizon=horizon,
                    holdout_size=holdout_size,
                    winsorize_returns=winsorize_returns,
                    xgb_cfg=xgb_cfg,
                )
            )

    if not components:
        raise ValueError("No valid ensemble components provided.")

    forecast_df = components[0].forecast_df[["ds"]].copy()
    for col in ["yhat", "lo80", "hi80", "lo95", "hi95"]:
        forecast_df[col] = np.mean([c.forecast_df[col].to_numpy(dtype=float) for c in components], axis=0)

    actual_vals = components[0].holdout_actual_df["value"].to_numpy(dtype=float)
    holdout_dates = components[0].holdout_actual_df["ds"].values
    component_pred_arrays = [c.holdout_pred_df["yhat"].to_numpy(dtype=float) for c in components]
    holdout_pred_vals = np.mean(component_pred_arrays, axis=0)

    mae = float(mean_absolute_error(actual_vals, holdout_pred_vals))
    rmse = float(np.sqrt(mean_squared_error(actual_vals, holdout_pred_vals)))
    mape = _mape(actual_vals, holdout_pred_vals)

    # Approximate ensemble holdout intervals using average component residual stds.
    # This is still approximate, but much better than always returning null.
    component_resid_stds = []
    for c in components:
        c_actual = c.holdout_actual_df["value"].to_numpy(dtype=float)
        c_pred = c.holdout_pred_df["yhat"].to_numpy(dtype=float)
        component_resid_stds.append(float(np.std(c_actual - c_pred, ddof=0)))

    sigma_ens = float(np.mean(component_resid_stds)) if component_resid_stds else 1.0
    z80 = 1.2815515655446004
    z95 = 1.959963984540054

    lo80 = holdout_pred_vals - z80 * sigma_ens
    hi80 = holdout_pred_vals + z80 * sigma_ens
    lo95 = holdout_pred_vals - z95 * sigma_ens
    hi95 = holdout_pred_vals + z95 * sigma_ens

    cov80 = _coverage(components[0].holdout_actual_df["value"], lo80, hi80)
    cov95 = _coverage(components[0].holdout_actual_df["value"], lo95, hi95)

    holdout_metrics = {
        "mae": mae,
        "rmse": rmse,
        "mape": mape,
        "holdout_size": int(holdout_size),
        "holdout_coverage_80": cov80,
        "holdout_coverage_95": cov95,
    }

    diagnostics = {
        "model_family": "ensemble",
        "base_model_family": "ensemble",
        "ensemble_components": ensemble_components,
        "trend": None,
        "seasonal": seasonal,
        "seasonal_periods": seasonal_periods if seasonal not in [None, "none", "None", False] else None,
        "damped_trend": damped_trend,
        "winsorize_returns": winsorize_returns,
        "n_obs": int(len(data)),
        "recent_trend_5d_pct": _calc_recent_trend_5d_pct(data["value"]),
        "anomaly_count": _calc_anomaly_count(data["value"]),
        "ensemble_holdout_sigma": sigma_ens,
    }

    holdout_actual_df = components[0].holdout_actual_df.copy()
    holdout_pred_df = pd.DataFrame({"ds": holdout_dates, "yhat": holdout_pred_vals})

    return ForecastArtifacts(
        forecast_df=forecast_df,
        holdout_metrics=holdout_metrics,
        diagnostics=diagnostics,
        holdout_actual_df=holdout_actual_df,
        holdout_pred_df=holdout_pred_df,
    )


def _run_rolling_backtest_summary(
    data: pd.DataFrame,
    model_family: str,
    holdout_size: int,
    seasonal_periods: int,
    damped_trend: bool,
    winsorize_returns: bool,
    seasonal: str | None,
    rolling_backtest_windows: int,
    xgb_cfg: dict[str, Any],
    ensemble_components: list[str] | None = None,
) -> dict[str, Any]:
    splits = _rolling_origin_splits(len(data), holdout_size=holdout_size, windows=rolling_backtest_windows)
    if not splits:
        return {"validation_scheme": "single_holdout", "rolling_backtest_windows_used": 0}

    maes = []
    rmses = []
    mapes = []
    cov80s = []
    cov95s = []

    ensemble_components = ensemble_components or ["ETS", "naive_last"]

    for _, train_end in splits:
        sub = data.iloc[: train_end + holdout_size].copy()

        if model_family == "ETS":
            art = _fit_and_forecast_ets(
                data=sub,
                horizon=7,
                holdout_size=holdout_size,
                seasonal_periods=seasonal_periods,
                damped_trend=damped_trend,
                winsorize_returns=winsorize_returns,
                seasonal=seasonal,
            )
        elif model_family == "XGB":
            art = _fit_and_forecast_xgb(
                data=sub,
                horizon=7,
                holdout_size=holdout_size,
                winsorize_returns=winsorize_returns,
                xgb_cfg=xgb_cfg,
            )
        elif model_family in ["ETS+benchmark_ensemble", "ensemble"]:
            art = _fit_and_forecast_ensemble(
                data=sub,
                horizon=7,
                holdout_size=holdout_size,
                seasonal_periods=seasonal_periods,
                damped_trend=damped_trend,
                winsorize_returns=winsorize_returns,
                seasonal=seasonal,
                ensemble_components=ensemble_components,
                xgb_cfg=xgb_cfg,
            )
        else:
            continue

        maes.append(art.holdout_metrics["mae"])
        rmses.append(art.holdout_metrics["rmse"])
        mapes.append(art.holdout_metrics["mape"])

        if art.holdout_metrics.get("holdout_coverage_80") is not None:
            cov80s.append(art.holdout_metrics["holdout_coverage_80"])
        if art.holdout_metrics.get("holdout_coverage_95") is not None:
            cov95s.append(art.holdout_metrics["holdout_coverage_95"])

    return {
        "validation_scheme": "rolling_backtest",
        "rolling_backtest_windows_used": len(maes),
        "rolling_backtest_mae_mean": float(np.mean(maes)) if maes else None,
        "rolling_backtest_rmse_mean": float(np.mean(rmses)) if rmses else None,
        "rolling_backtest_mape_mean": float(np.mean(mapes)) if mapes else None,
        "rolling_backtest_coverage_80_mean": float(np.mean(cov80s)) if cov80s else None,
        "rolling_backtest_coverage_95_mean": float(np.mean(cov95s)) if cov95s else None,
    }


def fit_and_forecast(
    data,
    horizon=7,
    holdout_size=30,
    seasonal_periods=5,
    damped_trend=True,
    winsorize_returns=False,
    seasonal="add",
    model_family="ETS",
    validation_scheme="single_holdout",
    rolling_backtest_windows=3,
    xgb=None,
    ensemble_components=None,
    **kwargs,
):
    data = data.copy()
    data["ds"] = pd.to_datetime(data["ds"])
    data = data.sort_values("ds").drop_duplicates(subset=["ds"]).reset_index(drop=True)

    if len(data) <= holdout_size + 5:
        raise ValueError("Not enough data for holdout evaluation.")

    xgb_cfg = xgb or {}
    ensemble_components = ensemble_components or ["ETS", "naive_last"]

    if model_family == "ETS":
        artifacts = _fit_and_forecast_ets(
            data=data,
            horizon=horizon,
            holdout_size=holdout_size,
            seasonal_periods=seasonal_periods,
            damped_trend=damped_trend,
            winsorize_returns=winsorize_returns,
            seasonal=seasonal,
        )
    elif model_family == "XGB":
        artifacts = _fit_and_forecast_xgb(
            data=data,
            horizon=horizon,
            holdout_size=holdout_size,
            winsorize_returns=winsorize_returns,
            xgb_cfg=xgb_cfg,
        )
    elif model_family == "naive_last":
        artifacts = _fit_and_forecast_naive_last(
            data=data,
            horizon=horizon,
            holdout_size=holdout_size,
        )
    elif model_family == "moving_average":
        artifacts = _fit_and_forecast_moving_average(
            data=data,
            horizon=horizon,
            holdout_size=holdout_size,
            window=int(xgb_cfg.get("moving_average_window", 5)),
        )
    elif model_family in ["ETS+benchmark_ensemble", "ensemble"]:
        artifacts = _fit_and_forecast_ensemble(
            data=data,
            horizon=horizon,
            holdout_size=holdout_size,
            seasonal_periods=seasonal_periods,
            damped_trend=damped_trend,
            winsorize_returns=winsorize_returns,
            seasonal=seasonal,
            ensemble_components=ensemble_components,
            xgb_cfg=xgb_cfg,
        )
    else:
        raise ValueError(f"Unsupported model_family: {model_family}")

    if validation_scheme == "rolling_backtest" and model_family in ["ETS", "XGB", "ETS+benchmark_ensemble", "ensemble"]:
        validation_diag = _run_rolling_backtest_summary(
            data=data,
            model_family=model_family,
            holdout_size=holdout_size,
            seasonal_periods=seasonal_periods,
            damped_trend=damped_trend,
            winsorize_returns=winsorize_returns,
            seasonal=seasonal,
            rolling_backtest_windows=rolling_backtest_windows,
            xgb_cfg=xgb_cfg,
            ensemble_components=ensemble_components,
        )
        artifacts.diagnostics.update(validation_diag)
    else:
        artifacts.diagnostics["validation_scheme"] = "single_holdout"

    return artifacts