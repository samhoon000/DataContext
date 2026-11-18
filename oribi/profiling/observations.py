"""Per-call memoized observations shared across detectors."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING
from typing import Any

import pandas as pd

from ..exceptions import InvalidInputError

if TYPE_CHECKING:
    from ..api import ProfileOptions


@dataclass
class ColumnObservation:
    name: str
    series: pd.Series
    missing_count: int
    distinct_count: int
    logical_type: str
    stats: dict[str, Any] = field(default_factory=dict)


class AnalysisSession:
    def __init__(self, frame: pd.DataFrame, options: ProfileOptions):
        if not isinstance(frame, pd.DataFrame):
            raise InvalidInputError("inspect() expects a pandas DataFrame")
        if not frame.columns.is_unique:
            raise InvalidInputError("DataFrame column labels must be unique")
        if any(not isinstance(c, str) for c in frame.columns):
            raise InvalidInputError("DataFrame column labels must be strings")
        self.frame = frame
        self.options = options
        self.observations: dict[Any, ColumnObservation] = {}

    def observe(self, name: Any) -> ColumnObservation:
        if name in self.observations:
            return self.observations[name]
        series = self.frame[name]
        if pd.api.types.is_complex_dtype(series.dtype):
            raise InvalidInputError(f"Complex-valued column {name!r} is not supported")
        missing = int(series.isna().sum())
        try:
            distinct = int(series.nunique(dropna=True))
        except (TypeError, ValueError):
            distinct = int(series.dropna().astype(str).nunique())
        if pd.api.types.is_bool_dtype(series.dtype):
            logical = "boolean"
        elif pd.api.types.is_datetime64_any_dtype(series.dtype):
            logical = "temporal"
        elif pd.api.types.is_numeric_dtype(series.dtype):
            logical = "numeric"
        elif isinstance(series.dtype, pd.CategoricalDtype):
            logical = "categorical"
        elif series.dtype == object or pd.api.types.is_string_dtype(series.dtype):
            values = series.dropna()
            kinds = {type(v) for v in values.iloc[:100]}
            logical = "unknown" if len(kinds) > 1 and len({"str" if issubclass(t, str) else t.__name__ for t in kinds}) > 1 else "text"
            if values.empty:
                logical = "unknown"
            elif pd.api.types.infer_dtype(values, skipna=True) in {
                "integer", "floating", "mixed-integer-float"
            }:
                logical = "numeric"
        else:
            logical = "unknown"
        obs = ColumnObservation(str(name), series, missing, distinct, logical)
        self.observations[name] = obs
        return obs
