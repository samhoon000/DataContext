"""Conservative temporal cadence and gap observations."""

from collections import Counter
from dataclasses import replace
from datetime import date, datetime
import re

import pandas as pd

from ..models.evidence import Evidence, Warning

_MIN_UNIQUE_TIMESTAMPS = 4
_MIN_CADENCE_INTERVALS = 2
_MIN_CADENCE_SHARE = 0.6
_ISO_CALENDAR_VALUE = re.compile(
    r"[0-9]{4}-[0-9]{2}-[0-9]{2}"
    r"(?:[T ][0-9]{2}:[0-9]{2}(?::[0-9]{2}(?:\.[0-9]{1,9})?)?"
    r"(?:Z|[+-](?:[01][0-9]|2[0-3]):[0-5][0-9])?)?"
)


def date_like_evidence(series: pd.Series, distinct_count: int) -> Evidence | None:
    """Identify calendar values held in object or string columns."""
    values = series.dropna()
    if values.empty:
        return None
    first = values.iloc[0]
    details = {"observed_count": len(values), "distinct_count": distinct_count}
    if isinstance(first, str):
        if distinct_count < 2 or not _valid_iso_calendar_value(first):
            return None
        if pd.api.types.infer_dtype(values, skipna=False) == "string" and all(
            _valid_iso_calendar_value(value) for value in values.unique()
        ):
            return Evidence(
                "iso_temporal_values",
                f"All {distinct_count} distinct non-missing strings are ISO dates or timestamps",
                details,
            )
        return None
    kind = pd.api.types.infer_dtype(values, skipna=False)
    if kind in {"date", "datetime", "datetime64"} or (
        isinstance(first, date) and all(isinstance(value, date) for value in values)
    ):
        return Evidence(
            "temporal_object_values",
            f"All {len(values)} non-missing values are date or datetime objects",
            details,
        )
    return None


def _valid_iso_calendar_value(value: str) -> bool:
    if _ISO_CALENDAR_VALUE.fullmatch(value) is None:
        return False
    # Python 3.10 requires an explicit offset for the UTC designator.
    iso_value = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        datetime.fromisoformat(iso_value)
    except ValueError:
        return False
    return True


def temporal_gap_warnings(series: pd.Series, column_name: str, *, max_examples: int = 3,
                          max_warnings: int = 10) -> tuple[Warning, ...]:
    """Report possible gaps only when a dominant fixed cadence is supported."""
    timestamps = pd.DatetimeIndex(series.dropna().drop_duplicates().sort_values())
    if len(timestamps) < _MIN_UNIQUE_TIMESTAMPS:
        return ()

    deltas = [timestamps[i] - timestamps[i - 1] for i in range(1, len(timestamps))]
    ranked = Counter(deltas).most_common()
    cadence, dominant_count = ranked[0]
    if len(ranked) > 1 and ranked[1][1] == dominant_count:
        return ()
    if dominant_count < _MIN_CADENCE_INTERVALS or dominant_count / len(deltas) < _MIN_CADENCE_SHARE:
        return ()

    steps_by_interval: list[int] = []
    for delta in deltas:
        steps = delta / cadence
        if steps < 1 or steps != int(steps):
            return ()
        steps_by_interval.append(int(steps))
    if not any(steps > 1 for steps in steps_by_interval):
        return ()

    cadence_text = _format_timedelta(cadence)
    warnings: list[Warning] = []
    gap_count = 0
    for index, steps in enumerate(steps_by_interval):
        missing_count = steps - 1
        if not missing_count:
            continue
        gap_count += 1
        if gap_count > max_warnings:
            continue
        before, after = timestamps[index], timestamps[index + 1]
        examples = [before + cadence * step for step in range(1, min(steps, max_examples + 1))]
        gap_start = before + cadence
        gap_end = after - cadence
        evidence = Evidence(
            "temporal_gap",
            f"{missing_count} timestamp(s) may be absent relative to an inferred {cadence_text} cadence",
            {
                "cadence": cadence_text,
                "gap_start": gap_start.isoformat(),
                "gap_end": gap_end.isoformat(),
                "missing_intervals": missing_count,
                "expected_timestamps": [value.isoformat() for value in examples],
                "examples_omitted": missing_count - len(examples),
                "observed_before": before.isoformat(),
                "observed_after": after.isoformat(),
            },
        )
        warnings.append(Warning(
            "potential_temporal_gap", "low", "Potential temporal gap",
            f"{column_name} has a possible temporal gap relative to an inferred {cadence_text} cadence.",
            (evidence,),
            "The interval may reflect missing observations or a normal pause in data collection.",
            "Confirm the expected sampling schedule before treating the interval as missing data.",
        ))
    if gap_count > max_warnings and warnings:
        last = warnings[-1]
        evidence = last.evidence[0]
        omitted = gap_count - max_warnings
        warnings[-1] = replace(
            last,
            message=f"{last.message} {omitted} additional gap(s) omitted.",
            evidence=(replace(evidence, details={**evidence.details, "gaps_omitted": omitted}),),
        )
    return tuple(warnings)


def _format_timedelta(value: pd.Timedelta) -> str:
    """Format common fixed cadence durations consistently."""
    if value.microseconds or value.nanoseconds:
        return str(value)
    seconds = value.days * 86400 + value.seconds
    for size, unit in ((86400, "day"), (3600, "hour"), (60, "minute"), (1, "second")):
        if seconds % size == 0:
            amount = seconds // size
            return f"{amount} {unit}{'' if amount == 1 else 's'}"
    return str(value)
