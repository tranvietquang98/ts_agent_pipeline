# AI agent that evaluates forecast quality, diagnoses model behavior, and generates structured insights
# Produces recommendations and improvement actions based on metrics, diagnostics, and context

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, model_validator


SUPPORTED_PARAMETERS = {
    "forecast.model_family",
    "forecast.seasonal",
    "forecast.seasonal_periods",
    "forecast.damped_trend",
    "forecast.winsorize_returns",
    "validation.scheme",
    "validation.rolling_backtest_windows"
}

SUPPORTED_MODEL_FAMILIES = {
    "ETS",
    "XGB",
    "naive_last",
    "moving_average",
    "ETS+benchmark_ensemble",
    "ensemble",
}

SUPPORTED_VALIDATION_SCHEMES = {
    "single_holdout",
    "rolling_backtest",
}


class ImprovementAction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    parameter: Literal[
        "forecast.model_family",
        "forecast.seasonal",
        "forecast.seasonal_periods",
        "forecast.damped_trend",
        "forecast.winsorize_returns",
        "validation.scheme",
        "validation.rolling_backtest_windows"
    ]
    action: Literal["set", "increase", "decrease"]
    new_value: str | bool | int | float | None = Field(description="Use an absolute target value in most cases.")
    rationale: str

    @model_validator(mode="after")
    def validate_action_semantics(self):
        # Long-term rule: almost everything should be absolute "set".
        if self.action != "set":
            raise ValueError(
                f"Parameter {self.parameter} must use action='set', not {self.action!r}."
            )
        return self


class AgentEvaluation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    verdict: Literal["trust", "monitor", "retrain", "adjust"]
    confidence: Literal["low", "medium", "high"]
    summary: str
    reasonableness: list[str]
    diagnosis: list[str]
    recommendations: list[str]
    improvement_actions: list[ImprovementAction]


@dataclass
class EvaluationOutput:
    report_json: dict[str, Any]
    report_markdown: str
    used_openai: bool


def _normalize_parameter_value(parameter: str, value: Any) -> Any:
    if parameter == "validation.scheme" and isinstance(value, str):
        v = value.strip().lower()
        if v == "rolling_origin":
            return "rolling_backtest"
        if v in SUPPORTED_VALIDATION_SCHEMES:
            return v

    if parameter == "forecast.model_family" and isinstance(value, str):
        value = value.strip()

    if parameter == "forecast.seasonal":
        if isinstance(value, bool):
            return "add" if value else "none"
        if isinstance(value, str):
            v = value.strip().lower()
            if v in {"true", "add"}:
                return "add"
            if v in {"false", "none", "null"}:
                return "none"

    if isinstance(value, str):
        v = value.strip().lower()
        if v == "true":
            return True
        if v == "false":
            return False
        if v in {"null", "none"}:
            return None

    return value


def _sanitize_improvement_actions(actions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cleaned: list[dict[str, Any]] = []

    for raw in actions or []:
        parameter = raw.get("parameter")
        action = raw.get("action", "set")
        new_value = raw.get("new_value")
        rationale = raw.get("rationale", "")

        if parameter not in SUPPORTED_PARAMETERS:
            continue

        if action not in {"set", "increase", "decrease"}:
            action = "set"

        new_value = _normalize_parameter_value(parameter, new_value)

        if parameter == "forecast.model_family" and isinstance(new_value, str):
            if new_value not in SUPPORTED_MODEL_FAMILIES:
                continue

        if parameter == "validation.scheme" and isinstance(new_value, str):
            if new_value not in SUPPORTED_VALIDATION_SCHEMES:
                continue

        cleaned.append(
            {
                "parameter": parameter,
                "action": action,
                "new_value": new_value,
                "rationale": rationale,
            }
        )

    deduped: dict[str, dict[str, Any]] = {}
    for item in cleaned:
        deduped[item["parameter"]] = item

    return list(deduped.values())


def _build_prompt(
    symbol: str,
    metrics: dict,
    diagnostics: dict,
    recent_df: pd.DataFrame,
    forecast_df: pd.DataFrame,
    prior_accuracy: dict | None,
    news_context: dict | None,
) -> str:
    recent = recent_df.tail(20).to_dict(orient="records")
    fcst = forecast_df.to_dict(orient="records")
    return f"""
You are an evaluation agent for a daily time-series forecasting pipeline.
Assess forecast quality, diagnose issues, and recommend actions.
Return only JSON matching the requested schema.

Series symbol: {symbol}
Holdout metrics: {json.dumps(metrics)}
Diagnostics: {json.dumps(diagnostics)}
Prior realized forecast accuracy: {json.dumps(prior_accuracy or {})}
Recent observations (last ~20 rows): {json.dumps(recent, default=str)}
Forecast horizon rows: {json.dumps(fcst, default=str)}
News/event context: {json.dumps(news_context or {}, default=str)}

Rules:
- Reason about trend consistency, volatility, interval width, and whether the forecast looks plausible.
- Mention likely overfit/underfit/anomaly issues when supported by metrics or diagnostics.
- If recent news/event context is available, use it to explain unusual moves or state that the move may be event-driven.
- Distinguish company-specific context from broad market context where possible.
- Recommend concrete next actions.
- Prefer actions that the current pipeline can support:
  forecast.model_family,
  forecast.seasonal,
  forecast.seasonal_periods,
  forecast.damped_trend,
  forecast.winsorize_returns,
  validation.scheme,
  validation.rolling_backtest_windows.
- IMPORTANT: use action="set" with an ABSOLUTE target value for all scalar hyperparameters.
- IMPORTANT: do NOT use increase/decrease for validation.rolling_backtest_windows, forecast.seasonal_periods, forecast.damped_trend, forecast.winsorize_returns, validation.scheme, or forecast.model_family.
- Use validation.scheme="rolling_backtest" (NOT rolling_origin).
- For forecast.seasonal, use only "add" or "none".
- improvement_actions must be a list of objects with keys: parameter, action, new_value, rationale.
- Use an empty list [] when no improvement actions are needed.
""".strip()


def heuristic_evaluate(
    symbol: str,
    metrics: dict,
    diagnostics: dict,
    recent_df: pd.DataFrame,
    forecast_df: pd.DataFrame,
    prior_accuracy: dict | None,
    news_context: dict | None,
) -> EvaluationOutput:
    mape = metrics.get("mape", 999)
    recent_trend = diagnostics.get("recent_trend_5d_pct")
    anomaly_count = diagnostics.get("anomaly_count", 0)
    coverage_95 = metrics.get("holdout_coverage_95")
    forecast_drift = float((forecast_df["yhat"].iloc[-1] / forecast_df["yhat"].iloc[0] - 1) * 100) if len(forecast_df) > 1 else 0.0

    reasonableness: list[str] = []
    diagnosis: list[str] = []
    recs: list[str] = []
    actions: list[dict[str, Any]] = []

    if recent_trend is not None:
        reasonableness.append(f"Recent 5-day trend is {recent_trend:.2f}%, while 7-step forecast drift is {forecast_drift:.2f}%.")
    if coverage_95 is not None:
        reasonableness.append(f"Holdout 95% interval coverage is {coverage_95:.1f}%.")

    if mape < 2.5:
        verdict, confidence = "trust", "high"
        diagnosis.append("Holdout error is low relative to a noisy daily series.")
    elif mape < 6:
        verdict, confidence = "monitor", "medium"
        diagnosis.append("Holdout error is acceptable but recent behavior should still be monitored.")
    else:
        verdict, confidence = "adjust", "medium"
        diagnosis.append("Holdout error is elevated enough to justify a configuration adjustment.")

    if anomaly_count > 0:
        diagnosis.append(f"Detected {anomaly_count} large return anomalies in the historical series.")
        if not diagnostics.get("winsorize_returns", False):
            recs.append("Consider clipping or winsorizing extreme returns before refitting.")
            actions.append({
                "parameter": "forecast.winsorize_returns",
                "action": "set",
                "new_value": True,
                "rationale": "Large return anomalies may destabilize the level and trend estimate.",
            })

    if diagnostics.get("base_model_family") == "ETS" and abs(forecast_drift) < abs(recent_trend or 0) / 2:
        diagnosis.append("Forecast appears flatter than the recent regime.")
        recs.append("Compare ETS with a simple benchmark ensemble to reduce lag.")
        actions.append({
            "parameter": "forecast.model_family",
            "action": "set",
            "new_value": "ETS+benchmark_ensemble",
            "rationale": "A benchmark ensemble can reduce lag when the market reprices quickly.",
        })

    if diagnostics.get("validation_scheme") != "rolling_backtest":
        recs.append("Use rolling-backtest validation instead of a single holdout split.")
        actions.append({
            "parameter": "validation.scheme",
            "action": "set",
            "new_value": "rolling_backtest",
            "rationale": "Rolling-backtest validation tests robustness across more than one regime window.",
        })
        actions.append({
            "parameter": "validation.rolling_backtest_windows",
            "action": "set",
            "new_value": 5,
            "rationale": "More backtest windows provide more stable model comparisons.",
        })

    context_points = (news_context or {}).get("summary_points", [])
    if context_points and context_points[0] not in {"No news search triggered.", "Context search disabled."}:
        reasonableness.append("Recent news or event context may partially explain unusual price behavior.")

    if prior_accuracy and prior_accuracy.get("mae") is not None:
        reasonableness.append(
            f"Latest realized one-step forecast error MAE is {prior_accuracy['mae']:.4f} based on newly observed data."
        )

    if not recs:
        recs.append("Keep the model active, but continue daily monitoring and retraining as new observations arrive.")

    actions = _sanitize_improvement_actions(actions)

    report_json = {
        "series": symbol,
        "evaluation": {
            "verdict": verdict,
            "confidence": confidence,
            "summary": f"Forecast is {verdict} with {confidence} confidence.",
            "reasonableness": reasonableness,
            "diagnosis": diagnosis,
            "recommendations": recs,
            "improvement_actions": actions,
        },
        "metrics": metrics,
        "diagnostics": diagnostics,
        "prior_realized_accuracy": prior_accuracy,
        "news_context": news_context,
        "forecast_preview": forecast_df.assign(ds=forecast_df["ds"].astype(str)).to_dict(orient="records"),
        "meta": {"used_openai": False},
    }

    md = [
        f"# Daily Forecast Evaluation — {symbol}",
        f"**Verdict:** {verdict}",
        f"**Confidence:** {confidence}",
        "",
        "## Metrics",
        f"- MAE: {metrics.get('mae'):.4f}",
        f"- RMSE: {metrics.get('rmse'):.4f}",
        f"- MAPE: {metrics.get('mape'):.2f}%",
        f"- Holdout 80% coverage: {metrics.get('holdout_coverage_80'):.1f}%" if metrics.get("holdout_coverage_80") is not None else "- Holdout 80% coverage: n/a",
        f"- Holdout 95% coverage: {metrics.get('holdout_coverage_95'):.1f}%" if metrics.get("holdout_coverage_95") is not None else "- Holdout 95% coverage: n/a",
        "",
        "## Diagnosis",
    ]
    md.extend([f"- {x}" for x in diagnosis or ["No major issues flagged."]])
    md.extend(["", "## News / Event Context"])
    md.extend([f"- {x}" for x in (news_context or {}).get("summary_points", ["No news context available."])])
    md.append("")
    md.append("## Recommendations")
    md.extend([f"- {x}" for x in recs])
    return EvaluationOutput(report_json=report_json, report_markdown="\n".join(md), used_openai=False)


def openai_evaluate(
    symbol: str,
    metrics: dict,
    diagnostics: dict,
    recent_df: pd.DataFrame,
    forecast_df: pd.DataFrame,
    prior_accuracy: dict | None,
    news_context: dict | None,
) -> EvaluationOutput:
    from openai import OpenAI

    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    prompt = _build_prompt(symbol, metrics, diagnostics, recent_df, forecast_df, prior_accuracy, news_context)
    schema = AgentEvaluation.model_json_schema()
    schema["additionalProperties"] = False

    response = client.responses.create(
        model=os.environ.get("OPENAI_MODEL", "gpt-5.4"),
        input=prompt,
        text={
            "format": {
                "type": "json_schema",
                "name": "agent_evaluation",
                "strict": True,
                "schema": schema,
            }
        },
    )

    parsed = json.loads(response.output_text)
    parsed["improvement_actions"] = _sanitize_improvement_actions(parsed.get("improvement_actions", []))

    report_json = {
        "series": symbol,
        "evaluation": parsed,
        "metrics": metrics,
        "diagnostics": diagnostics,
        "prior_realized_accuracy": prior_accuracy,
        "news_context": news_context,
        "forecast_preview": forecast_df.assign(ds=forecast_df["ds"].astype(str)).to_dict(orient="records"),
        "meta": {"used_openai": True},
    }
    md = [
        f"# Daily Forecast Evaluation — {symbol}",
        f"**Verdict:** {parsed['verdict']}",
        f"**Confidence:** {parsed['confidence']}",
        "",
        parsed["summary"],
        "",
        "## Reasonableness",
        *[f"- {x}" for x in parsed["reasonableness"]],
        "",
        "## Diagnosis",
        *[f"- {x}" for x in parsed["diagnosis"]],
        "",
        "## News / Event Context",
        *[f"- {x}" for x in (news_context or {}).get("summary_points", ["No news context available."])],
        "",
        "## Recommendations",
        *[f"- {x}" for x in parsed["recommendations"]],
    ]
    return EvaluationOutput(report_json=report_json, report_markdown="\n".join(md), used_openai=True)


def evaluate(
    symbol: str,
    metrics: dict,
    diagnostics: dict,
    recent_df: pd.DataFrame,
    forecast_df: pd.DataFrame,
    prior_accuracy: dict | None,
    news_context: dict | None = None,
    use_openai_if_available: bool = True,
) -> EvaluationOutput:
    if use_openai_if_available and os.environ.get("OPENAI_API_KEY"):
        try:
            return openai_evaluate(symbol, metrics, diagnostics, recent_df, forecast_df, prior_accuracy, news_context)
        except Exception as exc:
            fallback = heuristic_evaluate(symbol, metrics, diagnostics, recent_df, forecast_df, prior_accuracy, news_context)
            fallback.report_json["evaluation"]["fallback_reason"] = f"OpenAI evaluation failed: {exc}"
            return fallback
    return heuristic_evaluate(symbol, metrics, diagnostics, recent_df, forecast_df, prior_accuracy, news_context)