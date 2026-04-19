# Applies evaluator recommendations to update the configuration

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from db import SQLiteStore


# Parameters the improver is allowed to modify, with expected types
ALLOWED_PARAMETERS: dict[str, tuple[type, ...]] = {
    "forecast.model_family": (str,),
    "forecast.seasonal": (str, type(None), bool),
    "forecast.seasonal_periods": (int, type(None)),
    "forecast.damped_trend": (bool,),
    "forecast.winsorize_returns": (bool,),
    "validation.scheme": (str,),
    "validation.rolling_backtest_windows": (int,)
}

# Supported action labels from the evaluator
ALLOWED_ACTIONS = {"set"}

# Optional bounds for numeric parameters
NUMERIC_BOUNDS: dict[str, tuple[int | float | None, int | float | None]] = {
    "validation.rolling_backtest_windows": (1, 100),
    "forecast.seasonal_periods": (1, 365)
}

# Optional allowed value sets for categorical parameters
ALLOWED_VALUE_SETS: dict[str, set[Any]] = {
    "forecast.model_family": {
        "ETS",
        "XGB",
        "naive_last",
        "moving_average",
        "ETS+benchmark_ensemble",
        "ensemble",
    },
    "forecast.seasonal": {"add", "none", None, True, False},
    "validation.scheme": {"single_holdout", "rolling_backtest"},
}


def _normalize_bool_like(value: Any) -> Any:
    if isinstance(value, str):
        v = value.strip().lower()
        if v == "true":
            return True
        if v == "false":
            return False
        if v == "none" or v == "null":
            return None
    return value


def _normalize_seasonal(value: Any) -> Any:
    value = _normalize_bool_like(value)
    if value is True:
        return "add"
    if value is False:
        return "none"
    if isinstance(value, str):
        v = value.strip().lower()
        if v == "add":
            return "add"
        if v == "none":
            return "none"
    return value


def _coerce_type(parameter: str, value: Any) -> Any:
    value = _normalize_bool_like(value)

    if parameter == "forecast.seasonal":
        return _normalize_seasonal(value)

    allowed_types = ALLOWED_PARAMETERS.get(parameter)
    if allowed_types is None:
        raise ValueError(f"Unsupported parameter: {parameter}")

    if value is None or isinstance(value, allowed_types):
        return value

    if bool in allowed_types and isinstance(value, str):
        value2 = _normalize_bool_like(value)
        if isinstance(value2, bool):
            return value2

    if int in allowed_types and isinstance(value, float) and float(value).is_integer():
        return int(value)

    if int in allowed_types and isinstance(value, str):
        return int(value)

    if str in allowed_types:
        return str(value)

    raise TypeError(f"Invalid value type for {parameter}: {type(value)}")


def _apply_bounds(parameter: str, value: Any) -> Any:
    if value is None:
        return value

    bounds = NUMERIC_BOUNDS.get(parameter)
    if bounds is None:
        return value

    lo, hi = bounds
    if isinstance(value, (int, float)):
        if lo is not None:
            value = max(value, lo)
        if hi is not None:
            value = min(value, hi)
    return value


def _validate_allowed_value(parameter: str, value: Any) -> None:
    allowed = ALLOWED_VALUE_SETS.get(parameter)
    if allowed is None:
        return
    if value not in allowed:
        raise ValueError(f"Value {value!r} not allowed for {parameter}")


def _resolve_new_value(parameter: str, action: str, old_value: Any, new_value: Any) -> Any:
    if parameter not in ALLOWED_PARAMETERS:
        raise ValueError(f"Unsupported parameter: {parameter}")
    if action not in ALLOWED_ACTIONS:
        raise ValueError(f"Unsupported action: {action}")

    new_value = _coerce_type(parameter, new_value)

    if action == "set":
        resolved = new_value
    else:
        raise ValueError(f"Unsupported action: {action}")

    resolved = _apply_bounds(parameter, resolved)
    resolved = _coerce_type(parameter, resolved)
    _validate_allowed_value(parameter, resolved)
    return resolved


def apply_improvements(
    config_path: str | Path,
    store: SQLiteStore,
    evaluation_json: dict[str, Any],
) -> dict[str, Any]:
    path = Path(config_path)
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    new_cfg = deepcopy(cfg)
    actions = evaluation_json.get("evaluation", {}).get("improvement_actions", [])
    now = datetime.now(timezone.utc).isoformat()

    skipped_actions: list[dict[str, Any]] = []

    for action in actions:
        parameter = action.get("parameter")
        action_type = action.get("action", "set")
        new_value = action.get("new_value")
        rationale = action.get("rationale", "")

        if not parameter or action_type not in ALLOWED_ACTIONS or parameter not in ALLOWED_PARAMETERS:
            skipped_actions.append(
                {
                    "parameter": parameter,
                    "action": action_type,
                    "new_value": new_value,
                    "reason": "unsupported_parameter_or_action",
                }
            )
            continue

        keys = parameter.split(".")
        cursor_new = new_cfg
        cursor_old = cfg

        for key in keys[:-1]:
            if not isinstance(cursor_new.get(key), dict):
                cursor_new[key] = {}
            cursor_new = cursor_new[key]

            if isinstance(cursor_old, dict) and key in cursor_old and isinstance(cursor_old[key], dict):
                cursor_old = cursor_old[key]
            else:
                cursor_old = {}

        leaf = keys[-1]
        old_value = cursor_old.get(leaf) if isinstance(cursor_old, dict) else None

        try:
            resolved_value = _resolve_new_value(parameter, action_type, old_value, new_value)
        except Exception as exc:
            skipped_actions.append(
                {
                    "parameter": parameter,
                    "action": action_type,
                    "new_value": new_value,
                    "reason": str(exc),
                }
            )
            continue

        cursor_new[leaf] = resolved_value
        store.append_config_change(now, parameter, str(old_value), str(resolved_value), rationale)

    new_yaml = yaml.safe_dump(new_cfg, sort_keys=False)
    if new_yaml != yaml.safe_dump(cfg, sort_keys=False):
        path.write_text(new_yaml, encoding="utf-8")

    # Return config plus debug metadata to make behavior easier to inspect.
    result = deepcopy(new_cfg)
    result["_improver_meta"] = {
        "applied_at": now,
        "n_actions_received": len(actions),
        "n_actions_skipped": len(skipped_actions),
        "skipped_actions": skipped_actions,
    }
    return result