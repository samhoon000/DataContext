"""Statistics for observed columns."""

import math
from typing import Any

import pandas as pd

from ..models.profile import StatisticsContext
from .observations import ColumnObservation


def _snapshot_top_value(value: Any, copies: dict[int, Any]) -> Any:
    """Copy built-in list and dict values without copying arbitrary objects."""
    kind = type(value)
    if kind is not list and kind is not dict:
        return value

    identity = id(value)
    if identity in copies:
        return copies[identity]

    # Register the copy before visiting children so cycles keep their shape.
    if kind is list:
        items: list[Any] = []
        copies[identity] = items
        items.extend(_snapshot_top_value(item, copies) for item in value)
        return items

    entries: dict[Any, Any] = {}
    copies[identity] = entries
    entries.update((key, _snapshot_top_value(item, copies)) for key, item in value.items())
    return entries


def profile_column(obs: ColumnObservation, row_count: int, top_k: int, include_quantiles: bool) -> StatisticsContext:
    s = obs.series
    denom = row_count or 1
    non_null = s.dropna()
    values: dict[str, Any] = {}
    if obs.logical_type == "numeric" and len(non_null):
        numeric = pd.to_numeric(non_null, errors="coerce").dropna()
        if len(numeric):
            def finite_or_none(value: Any) -> float | None:
                result = float(value)
                return result if math.isfinite(result) else None

            # Pandas' mean/median/std/skew emit a RuntimeWarning and can yield
            # NaN when the series holds infinities (e.g. nanops computes
            # (avg - values) ** 2 over inf). Compute the moment statistics on
            # the finite subset so no warning is raised; infinities are simply
            # excluded rather than turning the whole statistic into NaN.
            finite = numeric[numeric.map(math.isfinite)]
            values.update(
                minimum=finite_or_none(finite.min()) if len(finite) else None,
                maximum=finite_or_none(finite.max()) if len(finite) else None,
                mean=finite_or_none(finite.mean()) if len(finite) else None,
                median=finite_or_none(finite.median()) if len(finite) else None,
                standard_deviation=finite_or_none(finite.std()) if len(finite) else None,
            )
            values["zero_count"] = int((numeric == 0).sum())
            values["negative_count"] = int((numeric < 0).sum())
            if include_quantiles:
                qs = finite.quantile([.25, .5, .75]) if len(finite) else {}
                values["quantiles"] = {str(q): finite_or_none(v) for q, v in qs.items()}
            if len(finite) >= 3 and finite.nunique() > 1:
                skew = float(finite.skew())
                values["skewness"] = skew if pd.notna(skew) else None
    elif obs.logical_type == "temporal" and len(non_null):
        values["minimum"] = non_null.min()
        values["maximum"] = non_null.max()
        values["date_range"] = f"{values['minimum']} to {values['maximum']}"
    if len(non_null) and (obs.logical_type in {"text", "categorical", "boolean", "unknown", "duration"} or obs.distinct_count <= top_k * 4):
        try:
            counts = non_null.value_counts(dropna=True)
            counts = counts[counts > 0].head(top_k)
            values["top_values"] = tuple((_snapshot_top_value(k, {}), int(v)) for k, v in counts.items())
            values["top_frequencies"] = tuple(int(v) for v in counts.values)
        except (TypeError, ValueError):
            pass
    return StatisticsContext(
        missing_count=obs.missing_count,
        missing_rate=obs.missing_count / denom,
        distinct_count=obs.distinct_count,
        distinct_rate=obs.distinct_count / denom,
        **values,
    )
